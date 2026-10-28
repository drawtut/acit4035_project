"""Dual foot-mounted WT901C485-E trial recorder and reference-relative heading viewer.

Requires imu_reader.py (unchanged), numpy, scipy, matplotlib, pyserial.
Tkinter normally ships with Windows Python.

This is a *provisional straight-path foot heading* viewer, NOT validated FPA.
The 5-second reference assumes both shoes point down the marked walking line.
No sensor registers are changed or written by this program.

Run: python gait_recorder.py
"""
from __future__ import annotations

import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from scipy.spatial.transform import Rotation

from imu_reader import IMUReader, LEFT_PORT, RIGHT_PORT

# --- Experiment settings -----------------------------------------------------
CALIBRATE_SECONDS = 5.0
MAX_REFERENCE_GYRO_DPS = 8.0  # reject moving reference samples
MIN_CAL_SAMPLES = 15
# Sensor +Y is assumed to point towards the toes. Change to -y if necessary.
FOOT_FORWARD_AXIS = {"left": np.array([0., 1., 0.]),
                     "right": np.array([0., 1., 0.])}
# Default IMUReader rate is 10 Hz. Increasing this without testing throughput
# can cause missed transactions at 9600 baud.
POLL_HZ = 10.0
UI_REFRESH_MS = 80
# Exploratory stance thresholds. NOT validated gait event detection.
STANCE_GYRO_DPS = 18.0
STANCE_ACCEL_TOL_MS2 = 2.2
STANCE_MIN_SECONDS = 0.12
OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"
G = 9.80665
# ---------------------------------------------------------------------------


def sample_to_row(side, sample, start_time, ref_rotation, heading_offset):
    a, w, m, e = sample.accel, sample.gyro, sample.magnet, sample.euler
    heading = relative_heading(sample, side, ref_rotation, heading_offset)
    candidate = bool(np.linalg.norm(w) < STANCE_GYRO_DPS
                     and abs(np.linalg.norm(a) - G) < STANCE_ACCEL_TOL_MS2)
    return {
        "side": side, "time_s": sample.timestamp - start_time,
        "timestamp_monotonic_s": sample.timestamp,
        "ax_mps2": a[0], "ay_mps2": a[1], "az_mps2": a[2],
        "gx_dps": w[0], "gy_dps": w[1], "gz_dps": w[2],
        "mx_raw": m[0], "my_raw": m[1], "mz_raw": m[2],
        "roll_deg": e[0], "pitch_deg": e[1], "yaw_deg": e[2],
        "ref_heading_deg": heading if heading is not None else float("nan"),
        "stance_candidate": int(candidate),
    }


def rot_of(sample):
    # Same XYZ Euler convention as the existing alignment checker.
    return Rotation.from_euler('xyz', np.asarray(sample.euler, dtype=float), degrees=True)


def horizontal_heading(rotation, axis, up):
    vec = rotation.apply(axis)
    vec = vec - np.dot(vec, up) * up
    length = np.linalg.norm(vec)
    return None if length < 1e-3 else vec / length


def relative_heading(sample, side, reference, offset):
    if reference is None:
        return None
    # The neutral orientation establishes a local horizontal plane and heading.
    # Heading is relative to the marked forward direction, NOT true path FPA.
    up = np.array([0., 0., 1.])  # device Euler frame is assumed gravity-aligned
    f0 = horizontal_heading(reference, FOOT_FORWARD_AXIS[side], up)
    f1 = horizontal_heading(rot_of(sample), FOOT_FORWARD_AXIS[side], up)
    if f0 is None or f1 is None:
        return None
    cross = np.dot(up, np.cross(f0, f1))
    dot = np.clip(np.dot(f0, f1), -1.0, 1.0)
    return float(np.rad2deg(np.arctan2(cross, dot)) - offset)


class QueuedIMUReader(IMUReader):
    """Reuse the proven Modbus parser; retain EVERY successful poll in a queue."""
    def __init__(self, port, **kwargs):
        super().__init__(port, **kwargs)
        self.samples = Queue()  # unbounded for short experiments, drained by UI

    def _loop(self):
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            tic = time.monotonic()
            try:
                sample = self._read_once()
                with self._lock:
                    self._sample = sample
                self.samples.put(sample)
                self.ok_count += 1
                self.last_error = None
            except (ValueError, TimeoutError, OSError) as exc:
                self.error_count += 1
                self.last_error = str(exc)
            except Exception as exc:
                # Keep the display responsive and surface unexpected I/O errors.
                self.error_count += 1
                self.last_error = repr(exc)
            self._stop.wait(max(0., period - (time.monotonic() - tic)))


def stance_intervals(rows):
    """Simple contiguous stationary candidate runs; diagnostic only."""
    if not rows:
        return []
    times = np.array([r['time_s'] for r in rows])
    flags = np.array([r['stance_candidate'] for r in rows], dtype=bool)
    edges = np.diff(np.r_[False, flags, False].astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    results = []
    for start, end in zip(starts, ends):
        t0, t1 = times[start], times[end-1]
        if t1 - t0 >= STANCE_MIN_SECONDS:
            results.append((float(t0), float(t1)))
    return results


def save_csv(rows, file_path):
    if not rows:
        return
    with file_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_trial(recordings, tag):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"Foot-mounted IMU trial {tag} | provisional reference heading")
    labels = [(['ax_mps2', 'ay_mps2', 'az_mps2'], 'Acceleration (m/s²)'),
              (['gx_dps', 'gy_dps', 'gz_dps'], 'Angular velocity (°/s)'),
              (['mx_raw', 'my_raw', 'mz_raw'], 'Magnetometer (raw counts)'),
              (['roll_deg', 'pitch_deg', 'yaw_deg'], 'Euler angles (°)')]
    for side in ('left', 'right'):
        rows = recordings[side]
        if not rows:
            continue
        t = np.array([r['time_s'] for r in rows])
        for ax, (cols, ylabel) in zip(axes, labels):
            for column in cols:
                values = [r[column] for r in rows]
                ax.plot(t, values, label=f"{side[0].upper()} {column}", lw=1.1, alpha=.8)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
        for a, b in stance_intervals(rows):
            axes[3].axvspan(a, b, color=('tab:blue' if side == 'left' else 'tab:orange'), alpha=.09)
    for ax in axes:
        ax.legend(loc='upper right', ncol=3, fontsize=7)
    axes[-1].set_xlabel('Time since recording started (s)')
    fig.tight_layout()

    fig2, ax = plt.subplots(figsize=(11, 4))
    for side, color in [('left', 'tab:blue'), ('right', 'tab:orange')]:
        rows = recordings[side]
        if not rows:
            continue
        t = [r['time_s'] for r in rows]
        values = [r['ref_heading_deg'] for r in rows]
        ax.plot(t, values, label=f'{side.title()} reference-relative foot heading', color=color)
        for a, b in stance_intervals(rows):
            ax.axvspan(a, b, color=color, alpha=.08)
    ax.axhline(0, color='black', linewidth=.8, linestyle='--')
    ax.set(title='Provisional straight-path foot heading (NOT validated FPA)',
           xlabel='Time (s)', ylabel='Heading change from calibration (°)')
    ax.grid(alpha=.3)
    ax.legend()
    fig2.tight_layout()
    plt.show(block=False)


class GaitRecorder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('WT901C485-E | Foot gait recorder')
        self.geometry('940x660')
        self.resizable(True, True)
        self.readers = {}
        self.reference = {'left': None, 'right': None}
        self.offset = {'left': 0., 'right': 0.}
        self.cal_samples = {'left': [], 'right': []}
        self.cal_until = None
        self.mode = 'READY'
        self.rec_start = None
        self.recordings = {'left': [], 'right': []}
        self.last = {'left': None, 'right': None}
        self.tag = ''
        self._build_ui()
        self.protocol('WM_DELETE_WINDOW', self.close)
        try:
            self.readers['left'] = QueuedIMUReader(LEFT_PORT, poll_hz=POLL_HZ).start()
            self.readers['right'] = QueuedIMUReader(RIGHT_PORT, poll_hz=POLL_HZ).start()
        except Exception as exc:
            for reader in self.readers.values():
                reader.stop()
            messagebox.showerror('COM port error', str(exc))
            self.destroy()
            return
        self.after(UI_REFRESH_MS, self.tick)

    def _build_ui(self):
        self.status = tk.StringVar(value='READY | Mount sensors, toes straight along floor line')
        ttk.Label(self, textvariable=self.status, font=('Segoe UI', 12, 'bold'), wraplength=880).pack(pady=(16, 9))
        actions = ttk.Frame(self)
        actions.pack(pady=9)
        self.cal_button = ttk.Button(actions, text='CALIBRATE (5s)', command=self.calibrate)
        self.cal_button.grid(row=0, column=0, padx=8)
        self.start_button = ttk.Button(actions, text='START RECORDING', command=self.start_record, state='disabled')
        self.start_button.grid(row=0, column=1, padx=8)
        self.stop_button = ttk.Button(actions, text='STOP + SAVE + PLOT', command=self.stop_record, state='disabled')
        self.stop_button.grid(row=0, column=2, padx=8)
        ttk.Button(actions, text='RESET REFERENCE', command=self.reset).grid(row=0, column=3, padx=8)
        ttk.Label(self, text='Top view: blue LEFT / orange RIGHT. Upward dashed line = calibrated forward direction.').pack()
        self.canvas = tk.Canvas(self, width=860, height=365, background='white')
        self.canvas.pack(pady=10, fill='x', padx=20)
        self.info = tk.StringVar(value='Connecting sensors...')
        ttk.Label(self, textvariable=self.info, justify='left', font=('Consolas', 10)).pack(anchor='w', padx=25)
        ttk.Label(self, text=('The angles here are changes from the standing reference, not confirmed FPA. '
                              'For the first trials, walk a straight marked line; sensor +Y must point toward toes.'),
                  wraplength=850, foreground='#6b5540').pack(padx=20, pady=10)

    def calibrate(self):
        if self.mode == 'RECORDING':
            messagebox.showwarning('Recording active', 'Stop recording first.')
            return
        self.cal_samples = {'left': [], 'right': []}
        self.cal_until = time.monotonic() + CALIBRATE_SECONDS
        self.mode = 'CALIBRATING'
        self.cal_button.config(state='disabled')
        self.start_button.config(state='disabled')
        self.status.set('CALIBRATING | Stand still with BOTH toes pointing along the marked path')

    def reset(self):
        if self.mode in ('CALIBRATING', 'RECORDING'):
            messagebox.showwarning('Busy', 'Finish the current operation first.')
            return
        self.reference = {'left': None, 'right': None}
        self.offset = {'left': 0., 'right': 0.}
        self.mode = 'READY'
        self.start_button.config(state='disabled')
        self.status.set('REFERENCE RESET | Press CALIBRATE with toes straight ahead')

    def start_record(self):
        if self.mode != 'CALIBRATED' or any(self.reference[s] is None for s in ('left', 'right')):
            return
        self.recordings = {'left': [], 'right': []}
        self.rec_start = time.monotonic()
        self.tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.mode = 'RECORDING'
        self.cal_button.config(state='disabled')
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status.set('RECORDING | Walk along the marked straight path; press STOP when done')

    def stop_record(self):
        if self.mode != 'RECORDING':
            return
        self.mode = 'CALIBRATED'
        self.stop_button.config(state='disabled')
        self.start_button.config(state='normal')
        self.cal_button.config(state='normal')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for side in ('left', 'right'):
            path = OUTPUT_DIR / f'{self.tag}_{side}.csv'
            save_csv(self.recordings[side], path)
            if self.recordings[side]:
                paths.append(str(path))
        counts = ', '.join(f'{s}: {len(self.recordings[s])} samples' for s in ('left', 'right'))
        self.status.set(f'SAVED | {counts} | recordings/{self.tag}_*.csv')
        if paths:
            try:
                plot_trial(self.recordings, self.tag)
            except Exception as exc:
                messagebox.showwarning('Plot error', f'CSV files saved, but plotting failed:\n{exc}')
        else:
            messagebox.showwarning('No samples', 'No readings were collected during recording.')

    def _finish_cal(self):
        self.cal_until = None
        results = {}
        details = []
        for side in ('left', 'right'):
            samples = self.cal_samples[side]
            steady = [s for s in samples if np.linalg.norm(s.gyro) < MAX_REFERENCE_GYRO_DPS]
            details.append(f'{side}: {len(steady)} steady/{len(samples)} received')
            if len(steady) < MIN_CAL_SAMPLES:
                self.mode = 'READY' if any(v is None for v in self.reference.values()) else 'CALIBRATED'
                self.cal_button.config(state='normal')
                self.start_button.config(state='normal' if self.mode == 'CALIBRATED' else 'disabled')
                self.status.set('CALIBRATION FAILED | ' + ', '.join(details) + ' | Retry, holding still')
                return
            results[side] = Rotation.concatenate([rot_of(s) for s in steady]).mean()
        self.reference.update(results)
        self.offset = {'left': 0., 'right': 0.}
        self.mode = 'CALIBRATED'
        self.cal_button.config(state='normal')
        self.start_button.config(state='normal')
        self.status.set('CALIBRATED | ' + ', '.join(details) + ' | Ready to START RECORDING')

    def tick(self):
        now = time.monotonic()
        for side, reader in self.readers.items():
            while True:
                try:
                    sample = reader.samples.get_nowait()
                except Empty:
                    break
                self.last[side] = sample
                if self.mode == 'CALIBRATING' and sample.timestamp <= self.cal_until:
                    self.cal_samples[side].append(sample)
                elif self.mode == 'RECORDING' and sample.timestamp >= self.rec_start:
                    self.recordings[side].append(sample_to_row(
                        side, sample, self.rec_start, self.reference[side], self.offset[side]))
        if self.mode == 'CALIBRATING':
            if now >= self.cal_until:
                self._finish_cal()
            else:
                self.status.set(f'CALIBRATING | Hold still, toes forward | {self.cal_until-now:.1f}s remaining')
        self.draw()
        self.after(UI_REFRESH_MS, self.tick)

    def draw(self):
        c = self.canvas
        c.delete('all')
        c_width = max(600, c.winfo_width())
        centers = {'left': (c_width * .28, 182), 'right': (c_width * .72, 182)}
        lines = []
        for side, color in [('left', '#2563eb'), ('right', '#ea580c')]:
            x, y = centers[side]
            c.create_text(x, 28, text=side.upper(), font=('Segoe UI', 15, 'bold'), fill=color)
            c.create_line(x, 320, x, 66, dash=(6, 6), width=2, fill='#a0a0a0')
            sample, ref = self.last[side], self.reference[side]
            heading = (relative_heading(sample, side, ref, self.offset[side])
                       if sample is not None else None)
            if heading is None:
                c.create_text(x, 345, text='NO REFERENCE / DATA', fill=color)
            else:
                # Canvas screen y grows down; positive mathematical rotation = CCW.
                angle = np.deg2rad(heading)
                dx, dy = -np.sin(angle), -np.cos(angle)
                c.create_line(x-45*dx, y-45*dy, x+110*dx, y+110*dy,
                              fill=color, width=13, capstyle='round')
                c.create_oval(x-7, y-7, x+7, y+7, fill='white', outline=color, width=2)
                c.create_text(x, 345, text=f'{heading:+.1f}° vs reference', fill=color,
                              font=('Segoe UI', 13, 'bold'))
            if sample is not None:
                age = time.monotonic() - sample.timestamp
                lines.append(f'{side.upper():5s}: age={age:.2f}s  gyro={np.linalg.norm(sample.gyro):5.1f}°/s  '
                             f'accel={np.linalg.norm(sample.accel):5.2f}m/s²  '
                             f'poll ok/errors={self.readers[side].ok_count}/{self.readers[side].error_count} '
                             f' {self.readers[side].last_error or ""}')
            else:
                lines.append(f'{side.upper()}: waiting for data / {self.readers[side].last_error or ""}')
        self.info.set('\n'.join(lines))

    def close(self):
        if self.mode == 'RECORDING':
            if not messagebox.askyesno('Recording active', 'Stop, save and exit?'):
                return
            self.stop_record()
        for reader in self.readers.values():
            reader.stop()
        self.destroy()


if __name__ == '__main__':
    GaitRecorder().mainloop()
