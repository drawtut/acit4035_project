"""Loss-aware dual WT901C485-E acquisition for ACIT4035 foot-gait research.

READ ONLY: never changes device calibration, baud rate, or output rate.

Usage (from the folder containing the existing imu_reader.py):
    python imu_acquisition.py --probe
    python imu_acquisition.py --duration 30 --poll-hz 10
    python imu_acquisition.py --duration 30 --poll-hz 200 --baud auto
    python imu_acquisition.py --duration 30 --poll-hz 200 --left-baud 921600 --right-baud 921600

Requires: pip install pyserial

Records every successful Modbus response directly from the polling threads through
queues; does NOT sample the latest value in a slow UI loop. It does NOT filter,
calibrate, or claim to estimate FPA. The time-register read mode is a diagnostic;
Modbus responses may repeat the same internal IMU sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import json
import math
import re
from pathlib import Path
from queue import Empty, Full, Queue
import statistics
import struct
import sys
import threading
import time
from typing import Optional

import serial

# The existing project's port mapping is the source of truth.
# Run this file alongside imu_reader.py from your uploaded project.
from imu_reader import LEFT_PORT, RIGHT_PORT, MODBUS_ADDRESS

G = 9.80665
PROBE_BAUDS = (9600, 115200, 230400, 460800, 921600, 57600, 38400, 19200, 4800)
REQUEST_TIMEOUT_S = 0.22
# Windows 1 ms process timer + cooperative GIL yielding for two IMUs.
FIELDS = [
    'side', 'sequence', 'host_time_utc', 'host_elapsed_s',
    'request_start_monotonic_s', 'response_end_monotonic_s',
    'timestamp_midpoint_monotonic_s', 'transaction_ms',
    'device_yearmonth_raw', 'device_dayhour_raw', 'device_minsec_raw', 'device_millisecond_raw',
    'ax_counts', 'ay_counts', 'az_counts',
    'gx_counts', 'gy_counts', 'gz_counts',
    'mx_counts', 'my_counts', 'mz_counts',
    'roll_counts', 'pitch_counts', 'yaw_counts',
    'ax_mps2', 'ay_mps2', 'az_mps2',
    'gx_dps', 'gy_dps', 'gz_dps',
    'mx_raw', 'my_raw', 'mz_raw',
    'roll_deg', 'pitch_deg', 'yaw_deg',
]


def crc16(payload: bytes) -> int:
    result = 0xFFFF
    for b in payload:
        result ^= b
        for _ in range(8):
            result = (result >> 1) ^ (0xA001 if (result & 1) else 0)
    return result


# Windows timer resolution is normally too coarse for 5 ms polling intervals.
# The request is process-scoped and is reversed when Python exits normally.
# This does not modify the IMU hardware or CSV format.
_TIMER_LOCK = threading.Lock()
_TIMER_INITIALIZED = False
_TIMER_ACTIVE = False


def enable_high_resolution_timer():
    """Request 1 ms Windows timer resolution once per process when available."""
    global _TIMER_INITIALIZED, _TIMER_ACTIVE
    if _TIMER_INITIALIZED:
        return _TIMER_ACTIVE
    with _TIMER_LOCK:
        if _TIMER_INITIALIZED:
            return _TIMER_ACTIVE
        _TIMER_INITIALIZED = True
        if sys.platform != 'win32':
            return False
        try:
            import ctypes
            import atexit
            winmm = ctypes.WinDLL('winmm')
            if winmm.timeBeginPeriod(1) == 0:
                _TIMER_ACTIVE = True
                atexit.register(lambda: winmm.timeEndPeriod(1))
        except (AttributeError, OSError):
            pass
    return _TIMER_ACTIVE


def wait_until(deadline: float, stop_event: threading.Event, mode: str) -> bool:
    """Pace a serial polling thread without monopolizing the Python GIL.

    The original busy-spin used `pass` for the last ~3.5 ms. Two workers
    consequently starved each other (and the serial reader), explaining why
    `--timing max` outperformed `--timing precise` in the November 8 test.

    `precise` now yields the GIL via time.sleep(0) close to each deadline.
    Windows 1 ms timer resolution makes the earlier small sleeps more reliable.
    `legacy` retains its original wait behavior; `max` never calls this.
    """
    if mode == 'legacy':
        remaining = deadline - time.perf_counter()
        if remaining > 0 and stop_event.wait(remaining):
            return False
        return not stop_event.is_set()
    enable_high_resolution_timer()
    while not stop_event.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return True
        if remaining > 0.003:
            # Await briefly, leaving margin for Windows scheduling jitter.
            if stop_event.wait(min(remaining - 0.0015, 0.002)):
                return False
        else:
            # IMPORTANT: yield GIL so the other IMU and pyserial can run.
            # Never use a Python `pass` busy-spin on two polling threads.
            time.sleep(0)
    return False


def read_modbus_registers(ser, address: int, start: int, count: int):
    """Read signed raw words and unsigned raw words from the same transaction.

    Monotonic timestamps bracket the complete request/reply transaction;
    their midpoint is a host-side approximation, NOT the true sensor sample time.
    """
    if not 1 <= address <= 247 or not 1 <= count <= 40:
        raise ValueError('Invalid Modbus address/register count')
    request = struct.pack('>BBHH', address, 3, start, count)
    request += struct.pack('<H', crc16(request))
    # No other process should be using this serial port. Clearing input avoids
    # leftover bytes from a failed prior request. It does not synchronize a sensor.
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(request)
    ser.flush()
    # Match the faster, measured-good diagnostic: get the first byte, then
    # ask for the rest of the known-length frame in one read. A three-byte
    # read followed by another read can incur USB/Windows buffering delay.
    first = ser.read(1)
    if len(first) != 1:
        raise TimeoutError('No response byte received')
    rest = ser.read(2 * count + 4)
    t1 = time.perf_counter()
    if len(rest) != 2 * count + 4:
        raise TimeoutError(f'Incomplete response: expected {2 * count + 5} bytes, got {1 + len(rest)}')
    response = first + rest
    if response[0] != address:
        raise ValueError(f'Unexpected Modbus address 0x{response[0]:02x}')
    if response[1] == 0x83:
        raise ValueError(f'Modbus exception code {response[2]}')
    if response[1] != 3 or response[2] != 2 * count:
        raise ValueError(f'Unexpected Modbus response header: {response[:3].hex(" ")}')
    crc_received = struct.unpack('<H', response[-2:])[0]
    if crc16(response[:-2]) != crc_received:
        raise ValueError('Modbus CRC mismatch')
    tail = response[3:]
    words_unsigned = struct.unpack('>' + 'H' * count, tail[:-2])
    words_signed = struct.unpack('>' + 'h' * count, tail[:-2])
    return t0, t1, words_unsigned, words_signed


def decode_sample(side, sequence, t0, t1, words_u, words_s, run_start, utc_start, include_time):
    if include_time:
        if len(words_u) != 16:
            raise ValueError('Expected 16 registers with device time')
        device_words = list(words_u[:4])
        motion = words_s[4:]
    else:
        if len(words_u) != 12:
            raise ValueError('Expected 12 sensor registers')
        device_words = [None] * 4
        motion = words_s
    if len(motion) != 12:
        raise ValueError('Invalid motion register length')
    midpoint = (t0 + t1) / 2
    timestamp = (utc_start + timedelta(seconds=midpoint - run_start)).isoformat(timespec='microseconds')
    accel = [v * (16 * G / 32768) for v in motion[0:3]]
    gyro = [v * (2000 / 32768) for v in motion[3:6]]
    euler = [v * (180 / 32768) for v in motion[9:12]]
    row = {
        'side': side, 'sequence': sequence, 'host_time_utc': timestamp,
        'host_elapsed_s': midpoint - run_start,
        'request_start_monotonic_s': t0,
        'response_end_monotonic_s': t1,
        'timestamp_midpoint_monotonic_s': midpoint,
        'transaction_ms': (t1-t0)*1000,
    }
    for name, value in zip(FIELDS[8:12], device_words):
        row[name] = '' if value is None else value
    for name, value in zip(FIELDS[12:24], motion):
        row[name] = value
    for name, value in zip(FIELDS[24:], [*accel, *gyro, *motion[6:9], *euler]):
        row[name] = value
    return row


def try_baud(port: str, baud: int, address: int, read_time: bool, timeout=REQUEST_TIMEOUT_S) -> bool:
    """A valid CRC-checked FC03 response is required; no hardware writes."""
    try:
        with serial.Serial(port, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                           timeout=timeout, write_timeout=timeout) as ser:
            # Probe with the universally used sensor data register range.
            for _ in range(2):
                try:
                    read_modbus_registers(ser, address, 0x34, 1)
                    if read_time:
                        read_modbus_registers(ser, address, 0x30, 16)
                    return True
                except (OSError, serial.SerialException, ValueError, TimeoutError):
                    time.sleep(.02)
    except (OSError, serial.SerialException, ValueError):
        pass
    return False


def detect_baud(port: str, address: int, read_time: bool, preferred=None):
    options = ([preferred] if preferred else []) + list(PROBE_BAUDS)
    for baud in dict.fromkeys(options):
        if try_baud(port, baud, address, read_time):
            return baud
    raise RuntimeError(
        f'{port}: no valid Modbus reply at supported speeds. Check COM port, power, '
        'USB-RS485 adapter and ensure other IMU applications are closed. '
        'If you recently changed baud, power-cycle the IMU and retry.'
    )


class Worker:
    """One polling thread per serial port; main thread owns all file writing."""
    def __init__(self, side, ser, address, hz, duration, include_time, records, errors,
                 start_signal, run_clock, stop_event, queue_drop_limit, timing_mode):
        self.side = side
        self.ser = ser
        self.address = address
        self.hz = hz
        self.duration = duration
        self.include_time = include_time
        self.records = records
        self.errors = errors
        self.start_signal = start_signal
        self.run_clock = run_clock
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self.loop, name=f'{side}-modbus', daemon=True)
        self.sequence = 0
        self.attempts = 0
        self.error_counts = Counter()
        self.first_error = None
        self.queue_drops = 0
        self.skipped_slots = 0
        self.queue_drop_limit = queue_drop_limit
        self.timing_mode = timing_mode
        self.finished = threading.Event()

    def loop(self):
        try:
            self.start_signal.wait()
            start = self.run_clock['start']
            utc = self.run_clock['utc']
            period = 1.0 / self.hz
            next_poll = start
            end = start + self.duration
            start_reg, count = (0x30, 16) if self.include_time else (0x34, 12)
            while not self.stop_event.is_set():
                now = time.perf_counter()
                if now >= end:
                    break
                if self.timing_mode != 'max':
                    if not wait_until(next_poll, self.stop_event, self.timing_mode):
                        break
                    if time.perf_counter() >= end:
                        break
                self.attempts += 1
                try:
                    t0, t1, unsigned, signed = read_modbus_registers(
                        self.ser, self.address, start_reg, count
                    )
                    row = decode_sample(self.side, self.sequence, t0, t1, unsigned,
                                        signed, start, utc, self.include_time)
                    try:
                        self.records.put_nowait(row)
                    except Full:
                        self.queue_drops += 1
                        if self.queue_drops >= self.queue_drop_limit:
                            self.stop_event.set()
                    self.sequence += 1
                except (TimeoutError, ValueError, serial.SerialException, OSError) as exc:
                    reason = type(exc).__name__ + ': ' + str(exc)
                    self.error_counts[reason] += 1
                    if self.first_error is None:
                        self.first_error = reason
                    try:
                        self.errors.put_nowait((self.side, time.perf_counter() - start, reason))
                    except Full:
                        pass
                if self.timing_mode != 'max':
                    # Keep a stable requested cadence while recovering when a
                    # transaction took longer than the interval. Never add a
                    # full OS sleep to an already-overrun poll cycle.
                    next_poll += period
                    now = time.perf_counter()
                    if now > next_poll:
                        skip = int((now - next_poll) // period) + 1
                        self.skipped_slots += skip
                        next_poll += skip * period
        finally:
            self.finished.set()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    return ordered[lo] * (hi-index) + ordered[hi] * (index-lo) if hi != lo else ordered[lo]


def summarize(rows, worker, duration, baud, include_time):
    ends = [float(row['timestamp_midpoint_monotonic_s']) for row in rows]
    gaps = [(b-a)*1000 for a, b in zip(ends, ends[1:])]
    latencies = [float(row['transaction_ms']) for row in rows]
    identical_motion = sum(
        tuple(a[key] for key in FIELDS[12:24]) == tuple(b[key] for key in FIELDS[12:24])
        for a, b in zip(rows, rows[1:])
    )
    dtime = [tuple(row[key] for key in FIELDS[8:12]) for row in rows]
    identical_devtime = sum(a == b for a, b in zip(dtime, dtime[1:])) if include_time else None
    # Device time can be unavailable/unreliable in some firmware; don't infer
    # true sensor update rate from repeated motion values in stationary trials.
    valid_time = include_time and bool(dtime) and any(any(v != 0 for v in stamp) for stamp in dtime)
    return {
        'side': worker.side,
        'port': worker.ser.port,
        'baud': baud,
        'requested_poll_hz': worker.hz,
        'polling_schedule': worker.timing_mode,
        'duration_s': duration,
        'poll_attempts': worker.attempts,
        'successful_responses': worker.sequence,
        'saved_rows': len(rows),
        'read_errors': sum(worker.error_counts.values()),
        'errors_by_type': dict(worker.error_counts.most_common(10)),
        'first_error': worker.first_error,
        'queue_drops': worker.queue_drops,
        'scheduler_slots_skipped': worker.skipped_slots,
        'successful_responses_per_s': round(worker.sequence / duration, 3),
        'saved_samples_per_s': round(len(rows) / duration, 3),
        'median_interval_ms': percentile(gaps, .5),
        'p95_interval_ms': percentile(gaps, .95),
        'max_interval_ms': max(gaps, default=None),
        'median_transaction_ms': percentile(latencies, .5),
        'p95_transaction_ms': percentile(latencies, .95),
        'repeated_consecutive_motion_registers': identical_motion,
        'near_accel_range_limit_rows': sum(any(abs(row[f'a{axis}_counts']) >= 32000 for axis in 'xyz') for row in rows),
        'near_gyro_range_limit_rows': sum(any(abs(row[f'g{axis}_counts']) >= 32000 for axis in 'xyz') for row in rows),
        'repeated_consecutive_device_time_registers': identical_devtime,
        'device_time_seems_populated': bool(valid_time),
        'note': 'The Modbus read rate is NOT proof of distinct internal IMU updates or sensor synchronization.',
    }


def parse_baud(arg: str) -> Optional[int]:
    if str(arg).lower() == 'auto':
        return None
    try:
        baud = int(arg)
    except ValueError as exc:
        raise ValueError('Baud must be auto or a positive integer') from exc
    if baud <= 0:
        raise ValueError('Baud must be positive')
    return baud


def record(args):
    selected = getattr(args, 'sensor', 'both')
    if selected not in ('left', 'right', 'both'):
        raise ValueError('--sensor must be left, right, or both')
    ports = {'left': args.left, 'right': args.right}
    if selected != 'both':
        ports = {selected: ports[selected]}
    if len(ports) == 2 and ports['left'].lower() == ports['right'].lower():
        raise RuntimeError('Each sensor must use a different COM port for independent polling')
    if args.address < 1 or args.address > 247:
        raise RuntimeError('Modbus slave address must be 1–247')
    baud_args = {'left': args.left_baud or args.baud,
                 'right': args.right_baud or args.baud}
    resolved_bauds = {}
    for side, port in ports.items():
        requested = parse_baud(baud_args[side])
        print(f'Checking {side} sensor on {port}...', flush=True)
        if requested is None:
            resolved_bauds[side] = detect_baud(port, args.address, not args.data_only)
        else:
            if not try_baud(port, requested, args.address, not args.data_only):
                raise RuntimeError(f'No valid response from {side} ({port}) at {requested} baud. '
                                   'Use --probe to discover the current rate.')
            resolved_bauds[side] = requested
        print(f'  {side}: responding at {resolved_bauds[side]:,} baud')
    if args.probe:
        print('Probe complete. No sensor configuration changed.')
        return None
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label_raw = getattr(args, 'label', '') or ''
    label = re.sub(r'[^A-Za-z0-9_-]+', '_', label_raw.strip())[:32].strip('_')
    base_name = f'{tag}_{label + "_" if label else ""}raw_{int(args.poll_hz)}hz'
    run_dir = args.outdir / base_name
    if run_dir.exists():
        run_dir = args.outdir / f'{base_name}_{time.monotonic_ns() % 1000000}'
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f'Writing untouched IMU measurements to {run_dir}', flush=True)
    records = {side: Queue(maxsize=args.queue_capacity) for side in ports}
    errors = Queue(maxsize=10000)
    stop_event = threading.Event()
    start_signal = threading.Event()
    run_clock = {}
    serials = {}
    workers = {}
    writers = {}
    csv_files = {}
    stored_rows = {side: [] for side in ports} if args.keep_in_memory else None
    counts = Counter()
    # In memory per-foot time/quality diagnostics only (small compared to full values).
    stats_rows = {side: [] for side in ports}
    error_file = None
    error_writer = None
    def drain():
        for side, q in records.items():
            while True:
                try:
                    row = q.get_nowait()
                except Empty:
                    break
                writers[side].writerow(row)
                counts[side] += 1
                # Keep only metadata for summary computation; not full repeated sensor values.
                stats_rows[side].append({key: row[key] for key in [
                    'timestamp_midpoint_monotonic_s', 'transaction_ms', *FIELDS[8:24]]})
                if stored_rows is not None:
                    stored_rows[side].append(row)
        while True:
            try:
                side, elapsed, reason = errors.get_nowait()
            except Empty:
                break
            if error_writer is not None:
                error_writer.writerow([side, f'{elapsed:.6f}', reason])
    try:
        for side, port in ports.items():
            ser = serial.Serial(port, baudrate=resolved_bauds[side], bytesize=8,
                                parity='N', stopbits=1, timeout=REQUEST_TIMEOUT_S,
                                write_timeout=REQUEST_TIMEOUT_S)
            serials[side] = ser
            csv_handle = (run_dir / f'{side}_raw.csv').open('w', newline='', encoding='utf-8')
            csv_files[side] = csv_handle
            writer = csv.DictWriter(csv_handle, fieldnames=FIELDS)
            writer.writeheader()
            writers[side] = writer
        error_file = (run_dir / 'read_errors.csv').open('w', encoding='utf-8', newline='')
        error_writer = csv.writer(error_file)
        error_writer.writerow(['side', 'host_elapsed_s', 'reason'])
        for side, ser in serials.items():
            worker = Worker(side, ser, args.address, args.poll_hz, args.duration,
                            not args.data_only, records[side], errors,
                            start_signal, run_clock, stop_event, args.max_queue_drops,
                            getattr(args, 'timing', 'precise'))
            workers[side] = worker
            worker.thread.start()
        run_clock['start'] = time.perf_counter()
        run_clock['utc'] = datetime.now(timezone.utc)
        start_signal.set()
        last_print = time.perf_counter()
        while not all(w.finished.is_set() for w in workers.values()):
            drain()
            now = time.perf_counter()
            if now - last_print >= 2:
                print(' | '.join(f'{side.upper()} saved={counts[side]} errors={sum(w.error_counts.values())}'
                                 for side, w in workers.items()), flush=True)
                last_print = now
            # Wake frequently to keep queues below capacity without busy-spinning.
            stop_event.wait(.01)
    except KeyboardInterrupt:
        print('Interrupted: stopping and saving acquired data...', flush=True)
        stop_event.set()
    finally:
        stop_event.set()
        start_signal.set()  # Prevent deadlock if startup failed.
        for worker in workers.values():
            worker.thread.join(timeout=1.5)
        if writers:
            drain()
        for handle in csv_files.values():
            handle.flush()
            handle.close()
        if error_file is not None:
            error_file.flush()
            error_file.close()
        for ser in serials.values():
            ser.close()

    elapsed = max(0.001, time.perf_counter() - run_clock['start']) if 'start' in run_clock else 0.001
    recorded_duration = min(args.duration, elapsed)
    summary = {
        'experiment': 'ACIT4035 WT901C485-E raw acquisition',
        'trial_label': label,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'requested_duration_s': args.duration,
        'actual_run_elapsed_s': round(elapsed, 3),
        'read_mode': '0x30..0x3F (device time + all 12 channels)' if not args.data_only else '0x34..0x3F (all 12 channels)',
        'read_only': True,
        'timing_mode': getattr(args, 'timing', 'precise'),
        'scheduler_fix': 'cooperative_yield_v2',
        'windows_timer_1ms_requested': _TIMER_ACTIVE,
        'filtering_applied': False,
        'calibration_applied': False,
        'unit_scaling': 'Based on project baseline: acceleration ±16 g, gyroscope ±2000 deg/s, Euler ±180 deg; verify device ranges',
        'modbus_slave_address': args.address,
        'timebase': 'host monotonic midpoint of request/response; independent, not hardware synchronized',
        'sensors': {
            side: summarize(stats_rows[side], w, recorded_duration, resolved_bauds[side],
                            not args.data_only)
            for side, w in workers.items()
        },
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    for side, data in summary['sensors'].items():
        print(f'{side.upper():5}: {data["saved_rows"]} saved / {data["successful_responses"]} '
              f'successes, {data["read_errors"]} read errors, '
              f'{data["saved_samples_per_s"]} rows/s, p95 gap={data["p95_interval_ms"]} ms, '
              f'queue drops={data["queue_drops"]}', flush=True)
    print(f'Saved: {run_dir / "summary.json"}', flush=True)
    if any(w.queue_drops for w in workers.values()):
        print('WARNING: queue overflow occurred; some readings were lost!', flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Loss-aware dual-foot WT901C485-E raw acquisition (read-only)')
    parser.add_argument('--probe', action='store_true', help='Find selected sensor(s) and current baud rates; do not record')
    parser.add_argument('--sensor', choices=('left', 'right', 'both'), default='both',
                        help='Which connected IMU to test (default: both)')
    parser.add_argument('--duration', type=float, default=30.0, help='Recording duration in seconds (default: 30)')
    parser.add_argument('--poll-hz', type=float, default=10.0, help='Requested host Modbus polls/s per foot (default 10)')
    parser.add_argument('--timing', choices=('precise','legacy','max'), default='precise',
                        help='precise=cooperative high-resolution pacing (default); legacy=OS timer wait; max=no rate limit')
    parser.add_argument('--label', default='', help='Optional trial label, e.g. stationary or walking')
    parser.add_argument('--baud', default='auto', help='Current sensor baud (auto, 9600, 115200, 921600, ...)')
    parser.add_argument('--left-baud', help='Override baud only for the left sensor')
    parser.add_argument('--right-baud', help='Override baud only for the right sensor')
    parser.add_argument('--left', default=LEFT_PORT)
    parser.add_argument('--right', default=RIGHT_PORT)
    parser.add_argument('--address', type=lambda s: int(s, 0), default=MODBUS_ADDRESS)
    parser.add_argument('--data-only', action='store_true', help='Read 12 motion registers only, skip device-time registers')
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent / 'recordings' / 'raw_acquisition')
    parser.add_argument('--queue-capacity', type=int, default=20000)
    parser.add_argument('--max-queue-drops', type=int, default=20)
    parser.add_argument('--keep-in-memory', action='store_true', help='Retain all full sample rows in RAM (normally off)')
    args = parser.parse_args()
    if args.duration <= 0 or args.poll_hz <= 0 or args.poll_hz > 500:
        parser.error('Duration must be >0 and poll-hz in (0,500]')
    if args.queue_capacity < 1 or args.max_queue_drops < 1:
        parser.error('Queue limits must be positive')
    try:
        record(args)
    except (RuntimeError, OSError, serial.SerialException, ValueError) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc


if __name__ == '__main__':
    main()
