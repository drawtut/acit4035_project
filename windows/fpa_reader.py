"""ACIT4035 WT901C485-E foot IMU viewer + loss-aware gait recorder.

200 Hz target, 230400 baud, one or two independent USB-RS485 adapters.

Acquisition is independent of tkinter: each serial-port thread puts every
successful, original Modbus response into a bounded recording queue. A separate
writer thread enriches/saves CSV rows. GUI refresh never drains/blocks logging.

Reference-relative fused yaw is a provisional orientation display, NOT true
trajectory-derived foot progression angle. No device configuration is written.

Needs project-local *optimized* imu_acquisition.py and imu_reader.py.
Example: py fpa_reader.py --sensor left --left COM5 --baud 230400 --poll-hz 200
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from queue import Queue, Empty, Full
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import serial

from imu_reader import LEFT_PORT, RIGHT_PORT, MODBUS_ADDRESS
from imu_acquisition import (
    FIELDS as RAW_FIELDS, REQUEST_TIMEOUT_S, read_modbus_registers,
    decode_sample, try_baud, wait_until,
)

DEFAULT_BAUD = 230400
DEFAULT_HZ = 200.0
CAL_SECONDS = 5.0
FUNCTIONAL_MOVE_SECONDS = 15.0  # aim for >=4 straight steps per foot (>=8 total)
MAX_CAL_GYRO_DPS = 8.0
MIN_CAL_ROWS = 50
UI_REFRESH_MS = 120
STANCE_GYRO_DPS = 18.0  # diagnostic candidate only, not a validated event
STANCE_ACCEL_TOL = 2.2
G = 9.80665
OUTPUT_DIR = Path(__file__).resolve().parent / 'recordings' / 'fpa_trials'
MANIFEST = Path(__file__).resolve().with_name('fpa_calibration_manifest.json')
TOE_SECONDS = 16.0  # >=2 s quiet baseline, then three separated heel-planted toe-up lifts
EXTRA_FIELDS = ['time_s', 'ref_heading_deg', 'stance_candidate']
CSV_FIELDS = RAW_FIELDS + EXTRA_FIELDS
STOP_TOKEN = object()


def wrap_deg(angle):
    """Wrapped signed rotation; +180 and -180 refer to the same heading."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def yaw_heading(row, reference_yaw):
    """Simple reference-relative *fused Euler yaw* (not anatomical FPA).

    Avoids projecting the assumed shoe +Y vector via a potentially ambiguous
    Euler convention under shoe pitch/roll. Gaps from the fusion estimator,
    especially magnetic disturbances, remain possible and are not hidden.
    """
    if row is None or reference_yaw is None:
        return float('nan')
    return wrap_deg(float(row['yaw_deg']) - float(reference_yaw))


def enriched_row(raw, start, reference_yaw):
    data = dict(raw)
    data['time_s'] = float(raw['timestamp_midpoint_monotonic_s']) - start
    data['ref_heading_deg'] = yaw_heading(raw, reference_yaw)
    gyro = np.array([raw[k] for k in ('gx_dps', 'gy_dps', 'gz_dps')], dtype=float)
    accel = np.array([raw[k] for k in ('ax_mps2', 'ay_mps2', 'az_mps2')], dtype=float)
    data['stance_candidate'] = int(
        np.linalg.norm(gyro) < STANCE_GYRO_DPS and
        abs(np.linalg.norm(accel) - G) < STANCE_ACCEL_TOL
    )
    return data


class TrialWriter:
    """Only writer thread owns the output files and derived calculations."""
    def __init__(self, directory, sides, label, requested_hz, baud, ports,
                 record_start, reference, readers, max_queue=40000):
        self.directory = directory
        self.sides = list(sides)
        self.label = label
        self.requested_hz = requested_hz
        self.baud = baud
        self.ports = ports
        self.record_start = record_start
        self.reference = dict(reference)
        self.queue = Queue(maxsize=max_queue)
        self.drop_count = Counter()
        self.count = Counter()
        self.repeat_count = Counter()
        self.prev_motion = {}
        self.first = {}
        self.last = {}
        self.gaps = {side: [] for side in sides}
        self.errors_start = {side: readers[side].error_count for side in sides}
        self.readers = readers
        self.writer_error = None
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True, name='fpa-csv-writer')

    def start(self):
        self.thread.start()
        return self

    def submit(self, side, row):
        try:
            self.queue.put_nowait((side, row))
        except Full:
            self.drop_count[side] += 1

    def close(self, end_time):
        self.end_time = end_time
        # The GUI calls this only after closing the acquisition window under
        # the readers' locks; there will be no later producer submissions.
        self.queue.put(STOP_TOKEN)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError('CSV writer did not terminate; check output directory')
        if self.writer_error:
            raise RuntimeError(f'CSV writer error: {self.writer_error}')
        duration = max(1e-6, end_time - self.record_start)
        report = {
            'trial': self.label,
            'requested_poll_hz': self.requested_hz,
            'baud': self.baud,
            'duration_s': duration,
            'provisional_reference_heading_not_validated_fpa': True,
            'heading_method': 'wrapped fused Euler yaw minus circular-mean reference yaw',
            'raw_sensor_registers_unchanged': True,
            'software_filtering_applied': False,
            'raw_columns': RAW_FIELDS,
            'sensors': {},
        }
        for side in self.sides:
            gaps = self.gaps[side]
            report['sensors'][side] = {
                'port': self.ports[side], 'rows': self.count[side],
                'saved_rows_per_s': self.count[side] / duration,
                'p95_gap_ms': float(np.percentile(gaps, 95)) if gaps else None,
                'max_gap_ms': max(gaps, default=None),
                'identical_neighboring_motion_payloads': self.repeat_count[side],
                'record_queue_drops': self.drop_count[side],
                'communication_errors_during_trial': (
                    self.readers[side].error_count - self.errors_start[side]),
                'skipped_poll_slots_total': self.readers[side].skipped_slots,
                'reference_yaw_deg': self.reference[side],
            }
        (self.directory / 'summary.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )
        return report

    def _write(self):
        files = {}
        try:
            writers = {}
            for side in self.sides:
                fh = (self.directory / f'{side}_raw.csv').open(
                    'w', newline='', encoding='utf-8')
                files[side] = fh
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writers[side] = writer
            while True:
                item = self.queue.get()
                if item is STOP_TOKEN:
                    break
                side, raw = item
                ts = float(raw['timestamp_midpoint_monotonic_s'])
                if ts < self.record_start:
                    continue
                if hasattr(self, 'end_time') and ts > self.end_time:
                    continue
                data = enriched_row(raw, self.record_start, self.reference[side])
                writers[side].writerow(data)
                self.count[side] += 1
                if side in self.last:
                    self.gaps[side].append(1000.0 * (ts - self.last[side]))
                else:
                    self.first[side] = ts
                self.last[side] = ts
                signature = tuple(raw[k] for k in RAW_FIELDS[12:18])
                if self.prev_motion.get(side) == signature:
                    self.repeat_count[side] += 1
                self.prev_motion[side] = signature
        except Exception as exc:
            self.writer_error = f'{type(exc).__name__}: {exc}'
        finally:
            for fh in files.values():
                fh.close()
            self.finished.set()


class FastReader:
    """One polling thread per port, no file I/O and no tkinter operations."""
    def __init__(self, side, port, baud, hz, address):
        self.side, self.port, self.baud, self.hz, self.address = (
            side, port, baud, hz, address)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.writer = None
        self.cal_start = None
        self.cal_until = None
        self.cal_rows = []
        self.latest = None
        self.error = None
        self.ok_count = 0
        self.error_count = 0
        self.skipped_slots = 0
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name=f'{side}-poller')

    def start(self):
        self.thread.start()
        self.ready.wait(timeout=3)
        if self.error:
            raise RuntimeError(self.error)
        return self

    def set_calibration(self, start, until):
        with self.lock:
            self.cal_start, self.cal_until = start, until
            self.cal_rows = []

    def take_calibration(self):
        with self.lock:
            self.cal_start = self.cal_until = None
            return list(self.cal_rows)

    def set_writer(self, writer):
        with self.lock:
            self.writer = writer

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _loop(self):
        try:
            with serial.Serial(
                self.port, baudrate=self.baud, bytesize=8, parity='N',
                stopbits=1, timeout=REQUEST_TIMEOUT_S,
                write_timeout=REQUEST_TIMEOUT_S,
            ) as ser:
                self.ready.set()
                period = 1.0 / self.hz
                next_poll = time.perf_counter()
                anchor = next_poll
                utc_start = datetime.now(timezone.utc)
                while not self.stop_event.is_set():
                    if not wait_until(next_poll, self.stop_event, 'precise'):
                        break
                    try:
                        t0, t1, u, s = read_modbus_registers(
                            ser, self.address, 0x34, 12)
                        raw = decode_sample(self.side, self.ok_count, t0, t1,
                                            u, s, anchor, utc_start,
                                            include_time=False)
                        self.ok_count += 1
                        # A short lock only publishes the pointer and enqueues
                        # raw data. No math, plotting, or disk I/O occurs here.
                        with self.lock:
                            self.latest = raw
                            if (self.cal_start is not None and
                                self.cal_start <= (t0+t1)*.5 <= self.cal_until):
                                self.cal_rows.append(raw)
                            if self.writer is not None:
                                self.writer.submit(self.side, raw)
                        self.error = None
                    except (ValueError, TimeoutError, OSError, serial.SerialException) as exc:
                        self.error_count += 1
                        self.error = f'{type(exc).__name__}: {exc}'
                    next_poll += period
                    now = time.perf_counter()
                    if now > next_poll:
                        skipped = int((now - next_poll) // period) + 1
                        next_poll += skipped * period
                        self.skipped_slots += skipped
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            self.ready.set()


def plot_trial(directory, sides):
    """Plots only after recording; unwrapping affects plots, never raw data."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    specs = [(['ax_mps2', 'ay_mps2', 'az_mps2'], 'Accel (m/s²)'),
             (['gx_dps', 'gy_dps', 'gz_dps'], 'Gyro (°/s)'),
             (['mx_raw', 'my_raw', 'mz_raw'], 'Magnetometer (counts)'),
             (['roll_deg', 'pitch_deg', 'yaw_deg'], 'Fused Euler (°)'),
             (['ref_heading_deg'], 'Yaw vs reference (°)')]
    for side in sides:
        path = directory / f'{side}_raw.csv'
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        stride = max(1, len(rows)//4500)
        shown = rows[::stride]
        t = np.array([float(r['time_s']) for r in shown])
        for idx, (cols, title) in enumerate(specs):
            ax = axes[idx]
            for key in cols:
                y = np.array([float(r[key]) for r in shown])
                if key == 'ref_heading_deg':
                    finite = np.isfinite(y)
                    # Unwrap contiguous valid segments for presentation only.
                    if np.any(finite):
                        y[finite] = np.rad2deg(np.unwrap(np.deg2rad(y[finite])))
                ax.plot(t, y, lw=.85, alpha=.8, label=f'{side} {key}')
            ax.set_ylabel(title)
            ax.grid(alpha=.2)
    for ax in axes:
        ax.legend(fontsize=7, loc='upper right', ncol=3)
    axes[-1].axhline(0, ls='--', c='black', lw=.6)
    axes[-1].set_xlabel('Seconds since recording start')
    fig.suptitle('ACIT4035 unfiltered measurements / provisional fused yaw (NOT FPA)')
    fig.tight_layout()
    plt.show(block=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest():
    if not MANIFEST.is_file():
        return {'schema': 'ACIT4035_READER_CAL_MANIFEST_V1', 'toe': {}, 'foot': {}}
    try:
        obj = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if obj.get('schema') != 'ACIT4035_READER_CAL_MANIFEST_V1':
            raise ValueError('Unexpected calibration manifest schema')
        obj.setdefault('toe', {})
        obj.setdefault('foot', {})
        return obj
    except Exception as exc:
        raise RuntimeError(f'Cannot read calibration manifest {MANIFEST}: {exc}') from exc


def save_manifest(obj):
    temp = MANIFEST.with_name(MANIFEST.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    temp.replace(MANIFEST)


def verified_calibration_entry(entry, side, schema):
    if not isinstance(entry, dict):
        return None
    path = Path(entry.get('path', ''))
    if not path.is_file() or sha256_file(path) != entry.get('sha256'):
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('side') != side or data.get('schema') != schema:
        return None
    return path


class FPAReader(tk.Tk):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sides = ['left', 'right'] if args.sensor == 'both' else [args.sensor]
        self.ports = {'left': args.left, 'right': args.right}
        self.title('ACIT4035 | 200 Hz raw IMU recorder + provisional yaw viewer')
        self.geometry('1050x820')
        self.minsize(950, 680)
        self.readers = {}
        self.reference = {side: None for side in self.sides}
        self.mode = 'CONNECTING'
        self.cal_until = None
        self.writer = None
        self.run_dir = None
        self.functional_stage = None
        self.functional_t0 = None
        self.functional_move_start = None
        self.functional_move_end = None
        self.functional_log = None
        self.toe_side = None
        self.toe_t0 = None
        self.trial_stage = None
        self.simple_straight_trial = False
        self.trial_move_start = None
        self.trial_move_end = None
        self.trial_t0 = None
        self.closing = False
        self.tick_after_id = None
        self.calibration_manifest = read_manifest()
        self.build_ui()
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.after(50, self.connect)

    def build_ui(self):
        self.status = tk.StringVar(value='Connecting...')
        ttk.Label(self, textvariable=self.status, font=('Segoe UI', 11, 'bold'),
                  wraplength=900).pack(pady=12)
        controls = ttk.Frame(self)
        controls.pack(pady=7)
        self.cal_button = ttk.Button(controls, text='CALIBRATE (5 s)',
                                     command=self.calibrate, state='disabled')
        self.cal_button.grid(row=0, column=0, padx=5)
        self.start_button = ttk.Button(controls, text='START RECORDING',
                                       command=self.start_record, state='disabled')
        self.start_button.grid(row=0, column=1, padx=5)
        self.stop_button = ttk.Button(controls, text='STOP + SAVE + PLOT',
                                      command=self.stop_record, state='disabled')
        self.stop_button.grid(row=0, column=2, padx=5)
        ttk.Button(controls, text='RESET REFERENCE', command=self.reset).grid(
            row=0, column=3, padx=5)
        function_controls = ttk.Frame(self)
        function_controls.pack(pady=5)
        self.functional_button = ttk.Button(
            function_controls, text='START FULL FPA CALIBRATION',
            command=self.start_functional, state='disabled')
        self.functional_button.grid(row=0, column=0, padx=8)
        self.steps_done_button = ttk.Button(
            function_controls, text='I FINISHED STEPS (4/FOOT)',
            command=self.finish_four_steps, state='disabled')
        self.steps_done_button.grid(row=0, column=1, padx=8)
        toe_controls = ttk.Frame(self)
        toe_controls.pack(pady=5)
        self.toe_buttons = {}
        for i, side in enumerate(self.sides):
            b = ttk.Button(toe_controls, text=f'TOE LIFTS {side.upper()} (3)',
                           command=lambda side=side: self.start_toe_lifts(side), state='disabled')
            b.grid(row=0, column=i, padx=5)
            self.toe_buttons[side] = b
        self.trial_button = ttk.Button(toe_controls, text='START GUIDED WALKING TRIAL',
                                      command=self.start_guided_trial, state='disabled')
        self.trial_button.grid(row=1, column=0, columnspan=2, pady=4)
        self.walk_done_button = ttk.Button(toe_controls, text='I FINISHED WALKING',
                                          command=self.finish_walking, state='disabled')
        self.walk_done_button.grid(row=1, column=2, padx=5)
        self.straight_button = ttk.Button(toe_controls,text='START STRAIGHT-LINE TRIAL (5 s ZERO ONLY)',
            command=self.start_straight_trial,state='disabled')
        self.straight_button.grid(row=2,column=0,columnspan=3,pady=5)
        self.open_studio_button=ttk.Button(toe_controls,text='OPEN FPA STUDIO (LAST RECORDING)',
            command=self.open_straight_studio)
        self.open_studio_button.grid(row=3,column=0,columnspan=3,pady=3)
        ttk.Label(self,text='Simple straight-line mode: align BOTH shoes parallel to tape, press CALIBRATE (5 s), '
            'then START STRAIGHT-LINE TRIAL. Full toe/PCA calibration is optional for this mode.',
            wraplength=900).pack(pady=2)

        ttk.Label(self, text='Functional calibration: stand 5 s → aim for >=4 straight steps PER FOOT within 15 s → stand 5 s. '
                  'Recording is automatic; calibration produces an experimental axis alignment, NOT FPA.',
                  wraplength=900).pack(pady=3)
        ttk.Label(self, text='Displays fused Euler yaw relative to standing calibration; '
                 'NOT trajectory-derived foot progression angle.').pack(pady=4)
        self.canvas = tk.Canvas(self, bg='white', height=280)
        self.canvas.pack(fill='x', padx=20, pady=8)
        self.info = tk.StringVar(value='Waiting for samples...')
        ttk.Label(self, textvariable=self.info, font=('Consolas', 10),
                  wraplength=900, justify='left').pack(anchor='w', padx=23)
        ttk.Label(self, text='One acquisition thread per IMU and a separate CSV writer. '
                 'GUI refresh does not control the logging rate. Raw counts are preserved.',
                  wraplength=900).pack(pady=9)

    def connect(self):
        try:
            for side in self.sides:
                port = self.ports[side]
                if not try_baud(port, self.args.baud, self.args.address, False):
                    raise RuntimeError(f'{side} ({port}) did not respond at {self.args.baud:,} baud')
                self.readers[side] = FastReader(side, port, self.args.baud,
                                                self.args.poll_hz,
                                                self.args.address).start()
        except Exception as exc:
            for reader in self.readers.values():
                reader.stop()
            messagebox.showerror('Sensor connection failed', str(exc))
            self.destroy()
            return
        self.mode = 'READY'
        self.cal_button.config(state='normal')
        self.functional_button.config(state='normal')
        for button in self.toe_buttons.values(): button.config(state='normal')
        self.trial_button.config(state='normal')
        self.straight_button.config(state='normal')
        self.status.set(f'CONNECTED | {", ".join(self.sides)} | '
                        f'{self.args.baud:,} baud | target {self.args.poll_hz:g} Hz/IMU')
        self.tick_after_id = self.after(UI_REFRESH_MS, self.tick)

    def calibrate(self):
        if self.mode not in ('READY', 'CALIBRATED'):
            return
        self.functional_button.config(state='disabled')
        self.mode = 'CALIBRATING'
        start = time.perf_counter()
        self.cal_until = start + CAL_SECONDS
        for reader in self.readers.values():
            reader.set_calibration(start, self.cal_until)
        self.cal_button.config(state='disabled')
        self.start_button.config(state='disabled')
        self.status.set('CALIBRATING | Stand stationary, toes aligned with floor line')

    def finish_cal(self):
        refs = {}
        details = []
        for side, reader in self.readers.items():
            rows = reader.take_calibration()
            steady = [r for r in rows if np.linalg.norm([
                r['gx_dps'], r['gy_dps'], r['gz_dps']]) < MAX_CAL_GYRO_DPS]
            details.append(f'{side}: {len(steady)}/{len(rows)} stationary')
            if len(steady) < MIN_CAL_ROWS:
                self.mode = 'CALIBRATED' if all(
                    ref is not None for ref in self.reference.values()) else 'READY'
                self.cal_button.config(state='normal')
                self.functional_button.config(state='normal')
                self.start_button.config(state=('normal' if self.mode == 'CALIBRATED'
                                                else 'disabled'))
                self.status.set('CALIBRATION FAILED | ' + ', '.join(details))
                return
            angles = np.deg2rad([r['yaw_deg'] for r in steady])
            refs[side] = float(np.rad2deg(np.arctan2(
                np.sin(angles).mean(), np.cos(angles).mean())))
        self.reference.update(refs)
        self.mode = 'CALIBRATED'
        self.cal_button.config(state='normal')
        self.functional_button.config(state='normal')
        self.start_button.config(state='normal')
        self.trial_button.config(state='normal')
        self.straight_button.config(state='normal')
        for button in self.toe_buttons.values(): button.config(state='normal')
        self.status.set('CALIBRATED | ' + ', '.join(details) +
                        ' | Reference yaw only, not FPA alignment')

    def reset(self):
        if self.mode in ('RECORDING', 'CALIBRATING'):
            messagebox.showwarning('Busy', 'Finish recording or calibration first')
            return
        self.reference = {side: None for side in self.sides}
        self.mode = 'READY'
        self.functional_button.config(state='normal')
        self.trial_button.config(state='normal')
        self.straight_button.config(state='normal')
        for button in self.toe_buttons.values(): button.config(state='normal')
        self.start_button.config(state='disabled')
        self.status.set('Reference reset. Stand toes-forward and recalibrate.')

    def start_toe_lifts(self, side):
        if self.mode not in ('READY', 'CALIBRATED') or side not in self.sides:
            return
        self.start_record(toe_side=side)
        if self.mode != 'RECORDING':
            return
        self.toe_side = side
        self.toe_t0 = self.writer.record_start
        self.status.set(f'TOE CALIBRATION {side.upper()} | Keep foot flat and still for 2 s')

    def start_straight_trial(self):
        """Controlled path-relative FPA mode; requires only explicit standing yaw zero."""
        if self.mode != 'CALIBRATED' or not all(self.reference.get(side) is not None for side in self.sides):
            messagebox.showwarning('Standing zero required',
                'First align both shoes exactly parallel to the straight tape direction, '
                'then click CALIBRATE (5 s) and keep both feet stationary.')
            return
        if not messagebox.askyesno('Confirm 0-degree reference',
            'Were BOTH shoes pointing exactly forward parallel to the tape during the last '
            '5-second CALIBRATE? Will you walk along that tape without turning? '
            'If not, cancel and redo standing calibration.'):
            return
        self.simple_straight_trial=True
        self.start_record(guided=True)
        if self.mode != 'RECORDING':
            self.simple_straight_trial=False
            return
        self.trial_stage='INITIAL_STILL'
        self.trial_t0=self.writer.record_start
        self.trial_move_start=None
        self.trial_move_end=None
        self.status.set('STRAIGHT-LINE 1/3 | Stand still 5s, then walk exactly along tape')

    def open_straight_studio(self):
        directory=self.run_dir
        if not directory or not (directory/'left_raw.csv').exists() or not (directory/'right_raw.csv').exists():
            messagebox.showinfo('No bilateral recording','Finish a bilateral trial first, or open Filter Studio separately.')
            return
        script=Path(__file__).resolve().with_name('fpa_filter_studio.py')
        if not script.exists():
            messagebox.showerror('Studio missing','fpa_filter_studio.py must be in the same project folder.')
            return
        try:
            subprocess.Popen([sys.executable,str(script),'--left',str(directory/'left_raw.csv'),
                '--right',str(directory/'right_raw.csv')],cwd=str(script.parent))
        except Exception as exc:
            messagebox.showerror('Studio launch failed',str(exc))

    def start_guided_trial(self):
        self.simple_straight_trial=False
        if self.mode not in ('READY', 'CALIBRATED'):
            return
        missing = [side for side in self.sides if not verified_calibration_entry(
            self.calibration_manifest['foot'].get(side), side, 'ACIT4035_FPA_CALIBRATION_V2')]
        if missing:
            messagebox.showwarning('Anatomical calibration required',
                'Missing a valid functional calibration for: ' + ', '.join(missing) +
                '. Run toe lifts for each foot, then full FPA calibration. '
                'You can still use START RECORDING for an uncalibrated exploratory trial.')
            return
        if not messagebox.askyesno('Confirm sensor mounting',
                'Are BOTH IMUs still fixed at exactly the same orientation relative to '
                'their shoes as during toe-lift and functional calibration? '
                'If either was moved, cancel and recalibrate.'):
            return
        self.start_record(guided=True)
        if self.mode != 'RECORDING':
            return
        self.trial_stage = 'INITIAL_STILL'
        self.trial_t0 = self.writer.record_start
        self.trial_move_start = None
        self.trial_move_end = None
        self.status.set('GUIDED WALK 1/3 | Stand still for 5 s')

    def finish_walking(self):
        if self.trial_stage != 'WALKING':
            return
        now = time.perf_counter()
        if now - self.trial_move_start < 1.0:
            messagebox.showwarning('Short walk', 'Record at least one second of walking.')
            return
        self.trial_move_end = now
        self.trial_stage = 'FINAL_STILL'
        self.walk_done_button.config(state='disabled')
        self.status.set('GUIDED WALK 3/3 | Stop and stand still for 5 s')

    def run_toe_processing(self, side):
        script = Path(__file__).resolve().with_name('fpa_estimation.py')
        if not script.is_file():
            messagebox.showwarning('Toe calibration not processed',
                'Raw toe-lift CSV saved. Place fpa_estimation.py in the project folder.')
            return
        output = self.run_dir / f'toe_axis_{side}.json'
        cmd = [sys.executable, str(script), 'toe-axis', '--input',
               str(self.run_dir / f'{side}_raw.csv'), '--side', side, '--output', str(output)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            (self.run_dir/f'toe_{side}_log.txt').write_text(
                (proc.stdout or '') + '\n' + (proc.stderr or ''), encoding='utf-8')
            if proc.returncode != 0 or not output.is_file():
                raise RuntimeError((proc.stderr or proc.stdout or 'No output')[-900:])
            data = json.loads(output.read_text(encoding='utf-8'))
            if data.get('schema') != 'ACIT4035_TOE_AXIS_V1' or data.get('side') != side:
                raise ValueError('Unexpected toe calibration output')
            self.calibration_manifest['toe'][side] = dict(path=str(output.resolve()),
                sha256=sha256_file(output), recorded_utc=datetime.now(timezone.utc).isoformat())
            # Old functional calibration is invalid after a new toe-lift session.
            self.calibration_manifest['foot'].pop(side, None)
            save_manifest(self.calibration_manifest)
            messagebox.showinfo('Toe calibration saved',
                f'{side.upper()} toe forward axis saved to:\n{output}\n'
                'Do not move the sensor. Complete the toe-lift procedure for the other foot.')
        except Exception as exc:
            messagebox.showwarning('Toe calibration failed',
                f'Raw recording saved in {self.run_dir}.\n{exc}\n'
                'See toe log, and repeat 3 heel-planted toes-up lifts.')

    def write_trial_metadata(self, report):
        t0 = self.trial_t0
        meta = dict(schema='ACIT4035_GUIDED_WALK_V1',
            initial_stationary_s=[0., round(self.trial_move_start-t0, 5)],
            walking_window_s=[round(self.trial_move_start-t0, 5),
                              round(self.trial_move_end-t0, 5)],
            final_stationary_s=[round(self.trial_move_end-t0, 5), round(report['duration_s'], 5)],
            sides=list(self.sides), video_offset_s=None,
            synchronization_status='not synchronized; use external sync event',
            mount_consistency_self_reported=True,
            sensor_serial_numbers='not recorded; confirm port-to-shoe mapping',
            calibration={},
            straight_line_standing_zero_mode=bool(self.simple_straight_trial),
            zero_reference_assumption=('both shoes aligned to tape when CALIBRATE (5 s) was pressed'
                if self.simple_straight_trial else 'no straight-line zero assumption'))
        for side in self.sides:
            entry = self.calibration_manifest['foot'].get(side)
            path = verified_calibration_entry(entry, side, 'ACIT4035_FPA_CALIBRATION_V2')
            if path:
                meta['calibration'][side] = dict(path=str(path), sha256=sha256_file(path),
                                                valid_for_this_trial='mounting user-confirmed, not independently verified')
            else:
                meta['calibration'][side] = None
        (self.run_dir/'trial_session.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        report['guided_trial_session'] = meta
        (self.run_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def start_record(self, functional=False, toe_side=None, guided=False):
        if self.mode not in (('CALIBRATED', 'READY') if (functional or toe_side or guided) else ('CALIBRATED',)):
            return
        tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        raw_label = ('functional_calibration' if functional else
                     f'toe_lifts_{toe_side}' if toe_side else
                     f'guided_{self.args.label}' if guided else self.args.label)
        name = ''.join(c if c.isalnum() or c in '_-' else '_'
                       for c in raw_label)[:40]
        self.run_dir = OUTPUT_DIR / f'{tag}_{name or "walking"}'
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            start = time.perf_counter()
            writer = TrialWriter(self.run_dir, self.sides, raw_label,
                                 self.args.poll_hz, self.args.baud, self.ports,
                                 start, self.reference, self.readers).start()
            for reader in self.readers.values():
                reader.set_writer(writer)
            self.writer = writer
        except Exception as exc:
            messagebox.showerror('Recording error', str(exc))
            return
        self.mode = 'RECORDING'
        self.start_button.config(state='disabled')
        self.cal_button.config(state='disabled')
        self.stop_button.config(state=('disabled' if (functional or toe_side or guided) else 'normal'))
        for button in self.toe_buttons.values(): button.config(state='disabled')
        self.trial_button.config(state='disabled')
        self.straight_button.config(state='disabled')
        self.walk_done_button.config(state='disabled')
        self.functional_button.config(state='disabled')
        self.status.set('RECORDING | Raw data captured by dedicated writer thread')

    def start_functional(self):
        if self.mode not in ('READY', 'CALIBRATED'):
            return
        missing = [side for side in self.sides if not verified_calibration_entry(
            self.calibration_manifest['toe'].get(side), side, 'ACIT4035_TOE_AXIS_V1')]
        if missing:
            messagebox.showwarning('Missing toe calibration',
                'Complete TOE LIFTS first for: ' + ', '.join(missing))
            return
        if not messagebox.askyesno('Mounting consistency',
                'Sensors have NOT shifted or been removed since both toe-lift recordings?'):
            return
        # Calibration mode records the entire static/movement/static sequence.
        # The viewer's 5s yaw reference is NOT needed for gyro-PCA foot calibration.
        self.start_record(functional=True)
        if self.mode != 'RECORDING':
            return
        self.functional_stage = 'INITIAL_STILL'
        self.functional_t0 = self.writer.record_start
        self.functional_move_start = None
        self.functional_move_end = None
        self.steps_done_button.config(state='disabled')
        self.status.set('FPA CALIBRATION 1/3 | Stand completely still for 5 seconds')

    def finish_four_steps(self):
        if self.functional_stage != 'FOUR_STEPS':
            return
        self.functional_move_end = time.perf_counter()
        if self.functional_move_end - self.functional_move_start < 1.0:
            messagebox.showwarning('Too short', 'Aim for at least four clear straight steps for each foot before finishing.')
            self.functional_move_end = None
            return
        self.functional_stage = 'FINAL_STILL'
        self.steps_done_button.config(state='disabled')
        self.status.set('FPA CALIBRATION 3/3 | Stop walking, stand completely still for 5 seconds')

    def write_functional_metadata(self, report):
        t0 = self.functional_t0
        start = self.functional_move_start
        end = self.functional_move_end
        meta = {
            'schema': 'ACIT4035_FUNCTIONAL_SESSION_V1',
            'purpose': 'functional IMU-to-foot alignment; aim >=4 strides each foot',
            'initial_stationary_s': [0, round(start-t0, 5)],
            'walking_window_s': [round(start-t0, 5), round(end-t0, 5)],
            'final_stationary_s': [round(end-t0, 5), round(report['duration_s'], 5)],
            'mount_forward': self.args.mount_forward,
            'axes_must_be_physically_verified': True,
            'no_fpa_computed_during_recording': True,
            'sides': self.sides,
        }
        (self.run_dir/'functional_session.json').write_text(
            json.dumps(meta, indent=2), encoding='utf-8')
        report['functional_calibration_session'] = meta
        (self.run_dir/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return meta

    def run_offline_calibration(self, meta):
        script = Path(__file__).resolve().with_name('fpa_estimation.py')
        if not script.is_file():
            messagebox.showwarning('Calibration recording saved',
                'Missing fpa_estimation.py; raw calibration saved for later processing.')
            return
        messages = []
        for side in self.sides:
            toe = verified_calibration_entry(self.calibration_manifest['toe'].get(side),
                                             side, 'ACIT4035_TOE_AXIS_V1')
            if toe is None:
                messages.append(f'{side}: missing/modified toe-axis calibration')
                continue
            output = self.run_dir / f'foot_calibration_{side}.json'
            cmd = [sys.executable, str(script), 'calibrate', '--input',
                   str(self.run_dir/f'{side}_raw.csv'), '--side', side,
                   '--toe-axis', str(toe), '--session', str(self.run_dir/'functional_session.json'),
                   '--output', str(output)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
                (self.run_dir/f'calibration_{side}_log.txt').write_text(
                    (proc.stdout or '')+'\n'+(proc.stderr or ''), encoding='utf-8')
                if proc.returncode or not output.is_file():
                    raise RuntimeError((proc.stderr or proc.stdout or 'No output')[-550:])
                obj = json.loads(output.read_text(encoding='utf-8'))
                if obj.get('side') != side or obj.get('schema') != 'ACIT4035_FPA_CALIBRATION_V2':
                    raise ValueError('Unexpected functional calibration output')
                self.calibration_manifest['foot'][side] = dict(
                    path=str(output.resolve()), sha256=sha256_file(output),
                    toe_sha256=sha256_file(toe),
                    recorded_utc=datetime.now(timezone.utc).isoformat(),
                    warning=obj.get('warning'),
                    toe_pca_agreement_deg=obj.get('toe_pca_agreement_deg'),
                    toe_walking_gravity_difference_deg=obj.get('toe_walking_gravity_difference_deg'))
                messages.append(f'{side}: saved functional anatomical calibration')
            except Exception as exc:
                self.calibration_manifest['foot'].pop(side, None)
                messages.append(f'{side}: FAILED — {exc}')
        save_manifest(self.calibration_manifest)
        messagebox.showinfo('Functional calibration results', '\n'.join(messages)+
                            '\nReview quality metrics and logs; these are experimental axes, not validated FPA.')

    def stop_record(self):
        if self.mode != 'RECORDING':
            return
        if (self.functional_stage in ('INITIAL_STILL', 'FOUR_STEPS', 'FINAL_STILL')
                or self.trial_stage in ('INITIAL_STILL', 'WALKING', 'FINAL_STILL')
                or (self.toe_side is not None and time.perf_counter() - self.toe_t0 < TOE_SECONDS)):
            messagebox.showwarning('Recording active', 'The guided phase has not finished yet.')
            return
        # Disable producer submission before enqueueing writer sentinel.
        end = time.perf_counter()
        for reader in self.readers.values():
            reader.set_writer(None)
        writer = self.writer
        self.writer = None
        self.mode = ('CALIBRATED' if all(v is not None for v in self.reference.values()) else 'READY')
        self.stop_button.config(state='disabled')
        self.status.set('Finalizing CSV recordings...')
        self.update_idletasks()
        meta = None
        try:
            report = writer.close(end)
            meta = self.write_functional_metadata(report) if self.functional_stage == 'FINALIZING' else None
            if self.trial_stage == 'FINALIZING':
                self.write_trial_metadata(report)
            if self.toe_side is not None:
                (self.run_dir/'toe_session.json').write_text(json.dumps(dict(
                    schema='ACIT4035_TOE_SESSION_V1', side=self.toe_side,
                    instruction='three heel-planted toes-up lifts; beginning at least 2 s at rest',
                    duration_s=report['duration_s'], raw_csv=f'{self.toe_side}_raw.csv'),
                    indent=2), encoding='utf-8')
            counts = ', '.join(f'{side}: {report["sensors"][side]["rows"]} rows '
                               f'({report["sensors"][side]["saved_rows_per_s"]:.1f} Hz)'
                               for side in self.sides)
            self.status.set(f'SAVED | {counts} | {self.run_dir}')
        except Exception as exc:
            messagebox.showerror('Save failed', str(exc))
        self.cal_button.config(state='normal')
        self.start_button.config(state=('normal' if all(v is not None for v in self.reference.values()) else 'disabled'))
        self.functional_button.config(state='normal')
        for button in self.toe_buttons.values(): button.config(state='normal')
        self.trial_button.config(state='normal')
        self.straight_button.config(state='normal')
        self.walk_done_button.config(state='disabled')
        if self.toe_side is not None:
            self.run_toe_processing(self.toe_side)
        if meta is not None:
            self.run_offline_calibration(meta)
        self.functional_stage = None
        self.trial_stage = None
        self.simple_straight_trial = False
        self.toe_side = None
        try:
            plot_trial(self.run_dir, self.sides)
        except Exception as exc:
            messagebox.showwarning('Plot error', f'CSV saved; plotting failed: {exc}')

    def tick(self):
        if self.closing or not self.winfo_exists():
            return
        if self.toe_side is not None:
            remaining = TOE_SECONDS - (time.perf_counter() - self.toe_t0)
            if remaining <= 0:
                self.stop_record()
            elif remaining > TOE_SECONDS-2:
                self.status.set(f'TOE {self.toe_side.upper()} | Keep shoe flat and motionless: {remaining:.1f} s')
            else:
                self.status.set(f'TOE {self.toe_side.upper()} | Lift toes UP three times with heel fixed; '
                                f'hold each raised pose briefly, lower between lifts | {remaining:.1f} s')
        if self.trial_stage == 'INITIAL_STILL':
            remaining = 5.0 - (time.perf_counter() - self.trial_t0)
            if remaining <= 0:
                self.trial_move_start = time.perf_counter()
                self.trial_stage = 'WALKING'
                self.walk_done_button.config(state='normal')
                self.status.set('GUIDED WALK 2/3 | Walk straight; press I FINISHED WALKING when stopped')
            else:
                self.status.set(f'GUIDED WALK 1/3 | Remain still {remaining:.1f} s')
        elif self.trial_stage == 'FINAL_STILL':
            remaining = 5.0 - (time.perf_counter() - self.trial_move_end)
            if remaining <= 0:
                self.trial_stage = 'FINALIZING'
                self.stop_record()
            else:
                self.status.set(f'GUIDED WALK 3/3 | Stand still {remaining:.1f} s')
        if self.functional_stage == 'INITIAL_STILL':
            elapsed = time.perf_counter() - self.functional_t0
            if elapsed >= 5.0:
                self.functional_move_start = time.perf_counter()
                self.functional_stage = 'FOUR_STEPS'
                self.steps_done_button.config(state='normal')
                self.status.set('FPA CALIBRATION 2/3 | Walk at least FOUR straight steps PER FOOT, then stop. '
                                'Moving phase ends automatically after 7 s.')
            else:
                self.status.set(f'FPA CALIBRATION 1/3 | Stand still {5-elapsed:.1f} s')
        elif self.functional_stage == 'FOUR_STEPS':
            remaining = FUNCTIONAL_MOVE_SECONDS - (time.perf_counter() - self.functional_move_start)
            if remaining <= 0:
                self.finish_four_steps()
            else:
                self.status.set(f'FPA CALIBRATION 2/3 | Aim for four steps PER FOOT, then STOP. '
                                f'Movement window {remaining:.1f} s remaining')
        elif self.functional_stage == 'FINAL_STILL':
            remaining = 5.0 - (time.perf_counter() - self.functional_move_end)
            if remaining <= 0:
                self.functional_stage = 'FINALIZING'
                self.stop_record()
            else:
                self.status.set(f'FPA CALIBRATION 3/3 | Stand still {remaining:.1f} s')
        if self.mode == 'CALIBRATING':
            remaining = self.cal_until - time.perf_counter()
            if remaining <= 0:
                self.finish_cal()
            else:
                self.status.set(f'CALIBRATING | Stationary foot | {remaining:.1f}s left')
        self.draw()
        if not self.closing:
            self.tick_after_id = self.after(UI_REFRESH_MS, self.tick)

    def draw(self):
        self.canvas.delete('all')
        width = max(600, self.canvas.winfo_width())
        centers = {'left': (width * .28, 170), 'right': (width * .72, 170)}
        lines = []
        for side in self.sides:
            x, y = centers[side]
            color = '#2563eb' if side == 'left' else '#ea580c'
            self.canvas.create_text(x, 25, text=side.upper(),
                                    font=('Segoe UI', 15, 'bold'), fill=color)
            self.canvas.create_line(x, 55, x, 300, dash=(5, 5), fill='#aaa')
            reader = self.readers[side]
            with reader.lock:
                row = reader.latest
            angle = yaw_heading(row, self.reference[side])
            if np.isfinite(angle):
                rad = np.deg2rad(angle)
                dx, dy = -np.sin(rad), -np.cos(rad)
                self.canvas.create_line(x-40*dx, y-40*dy, x+105*dx, y+105*dy,
                                        fill=color, width=12, capstyle='round')
                self.canvas.create_oval(x-6, y-6, x+6, y+6,
                                        fill='white', outline=color)
                self.canvas.create_text(x, 333,
                                        text=f'{angle:+.1f}° fused yaw vs reference',
                                        fill=color, font=('Segoe UI', 11, 'bold'))
            else:
                self.canvas.create_text(x, 331, text='NO REFERENCE / DATA', fill=color)
            age = (time.perf_counter() - float(row['timestamp_midpoint_monotonic_s'])
                   if row is not None else float('nan'))
            lines.append(f'{side.upper():5s} {self.ports[side]} age={age:.2f}s '
                         f'OK={reader.ok_count} errors={reader.error_count} '
                         f'skipped={reader.skipped_slots} '
                         f'{reader.error or ""}')
        if self.writer:
            lines.append('Recording queue: '
                         f'{self.writer.queue.qsize()} pending, '
                         f'{sum(self.writer.drop_count.values())} dropped')
        self.info.set('\n'.join(lines))

    def close(self):
        if self.mode == 'RECORDING':
            if self.functional_stage or self.trial_stage or self.toe_side:
                if not messagebox.askyesno('Calibration active', 'Stop and save incomplete calibration recording, then exit?'):
                    return
                for reader in self.readers.values():
                    reader.set_writer(None)
                if self.writer:
                    self.writer.close(time.perf_counter())
                self.functional_stage = None
                self.trial_stage = None
                self.toe_side = None
            elif not messagebox.askyesno('Recording active', 'Stop, save and exit?'):
                return
            else:
                self.stop_record()
        self.closing = True
        if self.tick_after_id is not None:
            try: self.after_cancel(self.tick_after_id)
            except tk.TclError: pass
            self.tick_after_id = None
        for reader in self.readers.values():
            reader.stop()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description='High-rate foot IMU gait recorder / provisional yaw viewer')
    parser.add_argument('--sensor', choices=('left', 'right', 'both'), default='left')
    parser.add_argument('--left', default=LEFT_PORT)
    parser.add_argument('--right', default=RIGHT_PORT)
    parser.add_argument('--baud', type=int, default=DEFAULT_BAUD)
    parser.add_argument('--poll-hz', type=float, default=DEFAULT_HZ)
    parser.add_argument('--address', type=lambda s: int(s, 0), default=MODBUS_ADDRESS)
    parser.add_argument('--label', default='walking')
    parser.add_argument('--mount-forward', choices=('x','-x','y','-y','z','-z'),
        default=None, help='PHYSICALLY VERIFIED sensor axis pointing toward toes. '
                           'Legacy metadata only; guided calibration uses per-foot toe-axis JSON.')
    args = parser.parse_args()
    if not 0 < args.poll_hz <= 500:
        parser.error('--poll-hz must be between 0 and 500')
    if args.sensor == 'both' and args.left.lower() == args.right.lower():
        parser.error('Two sensors require separate serial ports')
    FPAReader(args).mainloop()


if __name__ == '__main__':
    main()
