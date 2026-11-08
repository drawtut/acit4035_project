"""Foot + pelvis FPA recorder for two WT901C485-E sensors (Modbus RTU).

One sensor on the dorsum (top) of one foot, one on the sacrum (waist, middle
of the lower back). Settings for which foot and which sensor axes point
forward are in fpa_settings.py. Uses imu_reader.py unchanged.

Protocol (the program guides you):
  1. Put the foot along the marked walking line, face forward. Press START.
  2. Stand still for 5 s (countdown). This is the FPA = 0 reference.
  3. Walk straight along the line. Live FPA per step is shown.
  4. Press FINISH, stop in the same pose and stand still for 5 s.
     This is used to remove gyro drift.
  5. Data is saved in recordings/, then the FPA per stride is calculated
     and plotted (fpa_pelvis_analysis.py).

Sensor setup, once (sensors at the factory 9600 baud; see sensor_config.py):
    python sensor_config.py six-axis --baud 9600 --yes     no magnetometer
    python sensor_config.py moddelay 500 --baud 9600 --yes reply after 0.5 ms, not 3 ms
    python sensor_config.py bandwidth 98 --baud 9600 --yes fewer repeated samples
    python sensor_config.py baud 230400 --baud 9600 --yes  faster serial link
    python sensor_config.py read                           check (at 230400)
BAUDRATE in imu_reader.py must match the sensors (230400).

Sample rate: polls each sensor at POLL_HZ (200 Hz). The window shows the rate
actually reached and how many samples were new (not a repeat of the last).

Run:
    python fpa_pelvis_recorder.py          with the sensors
    python fpa_pelvis_recorder.py --demo   simulated sensors, to try the program

No sensor settings are changed by this program; it only reads the
algorithm register once at start-up to show 6-axis / 9-axis mode.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

import fpa_settings as S
import fpa_pelvis_analysis as A
from imu_reader import IMUReader, IMUSample, LEFT_PORT, RIGHT_PORT, BAUDRATE, MODBUS_ADDRESS

# --- Ports (COM numbers are set in imu_reader.py) ---------------------------
FOOT_PORT = LEFT_PORT
PELVIS_PORT = RIGHT_PORT
# ---------------------------------------------------------------------------
POLL_HZ = 200.0          # per sensor; needs 230400 baud and a short MODDELAY
UI_REFRESH_MS = 80
OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"
G = 9.80665
ROLES = ("foot", "pelvis")

# High-resolution clock for timestamps. time.monotonic() only ticks every
# ~15.6 ms on Windows before Python 3.13, which is too coarse at 200 Hz.
CLOCK = time.perf_counter
# Before Python 3.11, sleeps shorter than ~15 ms are rounded up on Windows.
COARSE_SLEEP = sys.platform == "win32" and sys.version_info < (3, 11)


def wait_until(deadline, stop_event):
    """Sleep until `deadline` (CLOCK time) with ~1 ms precision."""
    while not stop_event.is_set():
        remaining = deadline - CLOCK()
        if remaining <= 0:
            return
        if COARSE_SLEEP:
            time.sleep(remaining - 0.017 if remaining > 0.02 else 0)
        else:
            time.sleep(remaining - 0.0005 if remaining > 0.0015 else 0)


class RateMeter:
    """Samples per second over the last 2 s, total and new (non-repeated)."""

    def __init__(self):
        self.all, self.new = deque(maxlen=1000), deque(maxlen=1000)
        self._prev = None

    def add(self, sample):
        self.all.append(sample.timestamp)
        raw = (tuple(sample.accel), tuple(sample.gyro), tuple(sample.euler))
        if raw != self._prev:
            self.new.append(sample.timestamp)
        self._prev = raw

    @staticmethod
    def _rate(times, now):
        recent = [x for x in times if now - x <= 2.0]
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 2 else 0.0

    def rates(self):
        now = CLOCK()
        return self._rate(self.all, now), self._rate(self.new, now)


# ---------------------------------------------------------------------------
# Sensor readers
# ---------------------------------------------------------------------------

class QueuedIMUReader(IMUReader):
    """Reuse the proven Modbus parser; keep EVERY successful poll in a queue."""

    def __init__(self, port, **kwargs):
        super().__init__(port, **kwargs)
        self.samples = Queue()
        self.meter = RateMeter()

    def _read_once(self):
        # same parser; replace the coarse monotonic timestamp with CLOCK()
        return dataclasses.replace(super()._read_once(), timestamp=CLOCK())

    def _loop(self):
        period = 1.0 / self.poll_hz
        next_t = CLOCK()
        while not self._stop.is_set():
            try:
                sample = self._read_once()
                with self._lock:
                    self._sample = sample
                self.samples.put(sample)
                self.meter.add(sample)
                self.ok_count += 1
                self.last_error = None
            except Exception as exc:          # keep the UI alive on any I/O error
                self.error_count += 1
                self.last_error = str(exc)
            next_t += period
            now = CLOCK()
            if next_t < now - period:         # fell behind: do not burst to catch up
                next_t = now
            wait_until(next_t, self._stop)


class DemoReader:
    """Simulated sensor for --demo: walks with 8 deg toe-out when the app is
    in the WALKING phase, otherwise stands still. Heading drifts slowly."""

    def __init__(self, role, app, poll_hz=POLL_HZ):
        self.role, self.app, self.poll_hz = role, app, poll_hz
        self.port = f"DEMO-{role}"
        self.samples = Queue()
        self.meter = RateMeter()
        self.ok_count = self.error_count = 0
        self.last_error = None
        self._stop = threading.Event()
        self._walk_t0 = None

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        rng = np.random.default_rng(1 if self.role == "foot" else 2)
        stride, toe = 1.1, 8.0 if S.FOOT_SIDE == "left" else -8.0
        next_t = CLOCK()
        while not self._stop.is_set():
            ts = CLOCK()
            walking = self.app.mode == "WALKING"
            if walking and self._walk_t0 is None:
                self._walk_t0 = ts
            if not walking:
                self._walk_t0 = None
            ph = ((ts - self._walk_t0) / stride) % 1.0 if walking else 0.0
            if self.role == "foot":
                swing = walking and ph > 0.62
                s = math.sin(math.pi * (ph - 0.62) / 0.38) if swing else 0.0
                yaw = -90 + (toe if walking else 0) + 25 * s
                euler = [-15 + 35 * s, 4.0, yaw]
                gyro = [0.6 * (300 * s), 0.8 * (300 * s), 0.0]
                acc = G + 5 * s
            else:
                twist = 5 * math.sin(2 * math.pi * ph) if walking else 0.0
                euler = [90.0, 3.0, -90 + twist + 0.3 * (ts % 1000)]
                gyro = [15.0, 10.0, 0.0] if walking else [0.3, 0.2, 0.0]
                acc = G
            noise = rng.normal(0, 0.3, 3)
            euler = np.array(euler) + noise
            euler[2] = (euler[2] + 180) % 360 - 180
            sample = IMUSample(timestamp=ts, accel=np.array([0.1, -0.6, 0.79]) * acc,
                               gyro=np.abs(np.array(gyro) + noise), magnet=np.zeros(3),
                               euler=euler)
            self.samples.put(sample)
            self.meter.add(sample)
            self.ok_count += 1
            next_t = max(next_t + 1.0 / self.poll_hz, CLOCK() - 1.0 / self.poll_hz)
            wait_until(next_t, self._stop)


def check_algorithm(port):
    """Read register 0x24 once (read-only). Returns '6-axis', '9-axis' or None."""
    try:
        import serial
        from sensor_config import read_algorithm, SIX_AXIS, NINE_AXIS
        with serial.Serial(port, baudrate=BAUDRATE, bytesize=8, parity="N",
                           stopbits=1, timeout=1.0) as ser:
            value = read_algorithm(ser, MODBUS_ADDRESS)
        return {SIX_AXIS: "6-axis", NINE_AXIS: "9-axis"}.get(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live FPA (approximate; the final values come from fpa_pelvis_analysis.py)
# ---------------------------------------------------------------------------

class LiveFPA:
    def __init__(self, side=S.FOOT_SIDE):
        self.sign = A.side_sign(side)
        self.reset()

    def reset(self):
        self.ft, self.fh, self.fg = [], [], []      # foot time, heading, |gyro|
        self.pt, self.ph, self.pg = [], [], []      # pelvis time, heading, |gyro|
        self.f_horiz = self.p_horiz = None
        self._fprev = self._pprev = None
        self.offset = None
        self.walking = False
        self.stance = []                            # (t, heading) of current stance
        self.last_mid = None
        self.values = []
        n = A._window(0.05, POLL_HZ)                # ~50 ms median, as in the analysis
        self._g_hist, self._a_hist = deque(maxlen=n), deque(maxlen=n)

    @staticmethod
    def _unwrap(prev, h):
        """prev = (last wrapped heading, last unwrapped heading) or None.
        Returns (unwrapped heading, new prev)."""
        if prev is None:
            return h, (h, h)
        unwrapped = prev[1] + float(A.wrap180(h - prev[0]))
        return unwrapped, (h, unwrapped)

    def feed_pelvis(self, t, sample):
        h, horiz = A.heading_from_euler(sample.euler, S.PELVIS_FORWARD_AXIS)
        h_unw, self._pprev = self._unwrap(self._pprev, float(h[0]))
        self.pt.append(t); self.ph.append(h_unw)
        self.pg.append(float(np.linalg.norm(sample.gyro)))
        self.p_horiz = float(horiz[0])

    def feed_foot(self, t, sample):
        h, horiz = A.heading_from_euler(sample.euler, S.FOOT_FORWARD_AXIS)
        h_unw, self._fprev = self._unwrap(self._fprev, float(h[0]))
        g = float(np.linalg.norm(sample.gyro))
        self.ft.append(t); self.fh.append(h_unw); self.fg.append(g)
        self.f_horiz = float(horiz[0])
        self._g_hist.append(g)
        self._a_hist.append(float(np.linalg.norm(sample.accel)))
        on_ground = (float(np.median(self._g_hist)) < S.STANCE_GYRO_DPS
                     and abs(float(np.median(self._a_hist)) - G) < S.STANCE_ACCEL_TOL)
        if on_ground:
            self.stance.append((t, h_unw))
            return None
        result = self._end_stance(t)
        self.stance = []
        return result

    def _pelvis_mean(self, t0, t1):
        if len(self.pt) < 2:
            return None
        grid = np.linspace(t0, t1, 21)
        return float(np.interp(grid, self.pt, self.ph).mean())

    def _end_stance(self, t_now):
        if not self.stance or not self.walking or self.offset is None:
            return None
        ts = np.array([s[0] for s in self.stance])
        hs = np.array([s[1] for s in self.stance])
        dur = ts[-1] - ts[0] + 1.0 / POLL_HZ
        if not (S.MIN_STANCE_S <= dur <= S.MAX_STANCE_S):
            return None
        core = A.stance_core(0, len(hs) - 1)
        mid, foot_h = float(ts[core].mean()), float(np.median(hs[core]))
        stride = (mid - self.last_mid) if self.last_mid is not None else 1.1
        stride = min(max(stride, 0.6), 2.0)
        self.last_mid = mid
        pelvis = self._pelvis_mean(t_now - stride, t_now)   # one full stride
        if pelvis is None:
            return None
        fpa = float(self.sign * A.wrap180(foot_h - pelvis - self.offset))
        self.values.append(fpa)
        return fpa

    def set_reference(self, t0, t1):
        """Calibrate FPA = 0 from the still period [t0, t1]. Returns warnings."""
        warn = []
        ft, pt = np.array(self.ft), np.array(self.pt)
        fsel, psel = (ft >= t0) & (ft <= t1), (pt >= t0) & (pt <= t1)
        if fsel.sum() < 5 or psel.sum() < 5:
            warn.append("Too few samples during the still period; check the sensor connection.")
            if len(ft) == 0 or len(pt) == 0:
                return warn
            fsel, psel = np.ones(len(ft), bool), np.ones(len(pt), bool)
        rel = np.array(self.fh)[fsel] - np.interp(ft[fsel], pt, self.ph)
        self.offset = float(np.median(rel))
        if (np.array(self.fg)[fsel].max(initial=0) > 3 * S.STILL_GYRO_DPS
                or np.array(self.pg)[psel].max(initial=0) > 3 * S.STILL_GYRO_DPS):
            warn.append("You moved during the still period; the reference may be off. "
                        "Consider CANCEL and START again.")
        for name, val, setting in (("foot", self.f_horiz, "FOOT_FORWARD_AXIS"),
                                   ("pelvis", self.p_horiz, "PELVIS_FORWARD_AXIS")):
            if val is not None and val < S.MIN_HORIZONTAL:
                warn.append(f"{name} forward axis is nearly vertical: check {setting} "
                            "in fpa_settings.py.")
        self.stance = []
        self.last_mid = None
        return warn


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class FPARecorder(tk.Tk):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo
        self.title("WT901C485-E | Foot + pelvis FPA recorder" + (" (DEMO)" if demo else ""))
        self.geometry("900x700")
        self.mode = "READY"
        self.phase_end = None
        self.rec_start = None
        self.tag = ""
        self.rows = {r: [] for r in ROLES}
        self.last = {r: None for r in ROLES}
        self.live = LiveFPA()
        self.algorithms = {r: None for r in ROLES}
        self.readers = {}
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

        ports = {"foot": FOOT_PORT, "pelvis": PELVIS_PORT}
        try:
            if demo:
                self.readers = {r: DemoReader(r, self).start() for r in ROLES}
            else:
                for r in ROLES:                 # read-only check, before readers open the ports
                    self.algorithms[r] = check_algorithm(ports[r])
                for r in ROLES:
                    self.readers[r] = QueuedIMUReader(ports[r], baudrate=BAUDRATE, poll_hz=POLL_HZ).start()
        except Exception as exc:
            for reader in self.readers.values():
                reader.stop()
            messagebox.showerror("COM port error", str(exc))
            self.destroy()
            return
        if any(a == "9-axis" for a in self.algorithms.values()):
            self.warning.set("⚠ A sensor is in 9-axis mode (magnetometer in use). Heading can be "
                             "disturbed by nearby steel. Recommended: python sensor_config.py "
                             "six-axis --yes")
        self.after(UI_REFRESH_MS, self.tick)

    # ----- layout -----
    def _build_ui(self):
        side = S.FOOT_SIDE.upper()
        self.status = tk.StringVar(value=f"READY | {side} foot + pelvis. Stand with the foot along "
                                         "the line, facing forward, then press START.")
        ttk.Label(self, textvariable=self.status, font=("Segoe UI", 12, "bold"),
                  wraplength=860).pack(pady=(14, 6))
        bar = ttk.Frame(self)
        bar.pack(pady=6)
        self.start_btn = ttk.Button(bar, text="START", command=self.start)
        self.finish_btn = ttk.Button(bar, text="FINISH", command=self.finish, state="disabled")
        self.cancel_btn = ttk.Button(bar, text="CANCEL", command=self.cancel, state="disabled")
        for i, b in enumerate((self.start_btn, self.finish_btn, self.cancel_btn)):
            b.grid(row=0, column=i, padx=8)
        self.warning = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.warning, foreground="#b45309",
                  wraplength=860).pack()
        self.canvas = tk.Canvas(self, width=860, height=330, background="white")
        self.canvas.pack(pady=8, fill="x", padx=20)
        self.result = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.result, font=("Segoe UI", 13, "bold")).pack()
        self.info = tk.StringVar(value="Connecting sensors...")
        ttk.Label(self, textvariable=self.info, justify="left",
                  font=("Consolas", 10)).pack(anchor="w", padx=25, pady=6)
        ttk.Label(self, text=("Live values are approximate. Final FPA per stride, with drift "
                              "correction, is calculated after FINISH. Toe-out is positive."),
                  foreground="#6b5540", wraplength=850).pack(padx=20, pady=4)

    # ----- buttons -----
    def start(self):
        if self.mode != "READY":
            return
        self.rows = {r: [] for r in ROLES}
        self.live.reset()
        self.rec_start = CLOCK()
        self.tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mode = "START_STILL"
        self.phase_end = self.rec_start + S.STILL_SECONDS
        self.start_btn.config(state="disabled")
        self.finish_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.result.set("")

    def finish(self):
        if self.mode != "WALKING":
            return
        self.mode = "END_STILL"
        self.live.walking = False
        self.phase_end = CLOCK() + S.STILL_SECONDS
        self.finish_btn.config(state="disabled")

    def cancel(self):
        if self.mode == "READY":
            return
        self.mode = "READY"
        self.live.walking = False
        self._buttons_ready()
        self.status.set("CANCELLED | Nothing saved. Press START to try again.")

    def _buttons_ready(self):
        self.start_btn.config(state="normal")
        self.finish_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")

    # ----- main loop -----
    def tick(self):
        now = CLOCK()
        recording = self.mode != "READY"
        for role, reader in self.readers.items():
            while True:
                try:
                    sample = reader.samples.get_nowait()
                except Empty:
                    break
                self.last[role] = sample
                if not recording or sample.timestamp < self.rec_start:
                    continue
                t = sample.timestamp - self.rec_start
                self.rows[role].append(self._row(role, sample, t))
                if role == "pelvis":
                    self.live.feed_pelvis(t, sample)
                else:
                    fpa = self.live.feed_foot(t, sample)
                    if fpa is not None:
                        self._show_live()

        if self.mode == "START_STILL":
            left = self.phase_end - now
            if left > 0:
                self.status.set(f"STAND STILL | foot along the line, facing forward | {left:.1f} s")
            else:
                warn = self.live.set_reference(0.0, now - self.rec_start)
                self.live.walking = True
                self.mode = "WALKING"
                self.finish_btn.config(state="normal")
                self.status.set("WALK along the line now. Press FINISH when done, then stand still.")
                if warn:
                    self.warning.set("⚠ " + " ".join(warn))
        elif self.mode == "END_STILL":
            left = self.phase_end - now
            if left > 0:
                self.status.set(f"STAND STILL | same pose as at the start | {left:.1f} s")
            else:
                self._save_and_analyze()
        self.draw()
        self.after(UI_REFRESH_MS, self.tick)

    @staticmethod
    def _row(role, s, t):
        a, w, m, e = s.accel, s.gyro, s.magnet, s.euler
        return {"side": role, "time_s": t, "timestamp_monotonic_s": s.timestamp,
                "ax_mps2": a[0], "ay_mps2": a[1], "az_mps2": a[2],
                "gx_dps": w[0], "gy_dps": w[1], "gz_dps": w[2],
                "mx_raw": m[0], "my_raw": m[1], "mz_raw": m[2],
                "roll_deg": e[0], "pitch_deg": e[1], "yaw_deg": e[2]}

    def _show_live(self):
        v = self.live.values
        mean = np.mean(v[1:]) if len(v) > 1 else v[-1]
        self.result.set(f"Last step: {v[-1]:+.1f}°   |   mean of {len(v)} steps: {mean:+.1f}°  (live)")

    # ----- saving -----
    def _save_and_analyze(self):
        self.mode = "READY"
        self._buttons_ready()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if not self.rows["foot"] or not self.rows["pelvis"]:
            self.status.set("NOT SAVED | no data from one of the sensors.")
            messagebox.showwarning("No data", "One of the sensors delivered no samples.")
            return
        for role in ROLES:
            with (OUTPUT_DIR / f"{self.tag}_{role}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(self.rows[role][0].keys()))
                writer.writeheader()
                writer.writerows(self.rows[role])
        meta = {"foot_side": S.FOOT_SIDE, "foot_forward_axis": list(S.FOOT_FORWARD_AXIS),
                "pelvis_forward_axis": list(S.PELVIS_FORWARD_AXIS), "fpa_sign": S.FPA_SIGN,
                "foot_port": self.readers["foot"].port, "pelvis_port": self.readers["pelvis"].port,
                "algorithm": self.algorithms, "demo": self.demo, "poll_hz": POLL_HZ,
                "baudrate": BAUDRATE,
                "measured_hz": {r: round(len(self.rows[r]) / max(1e-6, self.rows[r][-1]["time_s"]
                                                                 - self.rows[r][0]["time_s"]), 1)
                                for r in ROLES}}
        (OUTPUT_DIR / f"{self.tag}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        counts = ", ".join(f"{r}: {len(self.rows[r])}" for r in ROLES)
        try:
            import matplotlib.pyplot as plt
            result = A.analyze(pd.DataFrame(self.rows["foot"]), pd.DataFrame(self.rows["pelvis"]),
                               S.FOOT_SIDE)
            result["strides"].to_csv(OUTPUT_DIR / f"{self.tag}_fpa_strides.csv", index=False)
            fig = A.plot(result, title=f"Recording {self.tag}")
            fig.savefig(OUTPUT_DIR / f"{self.tag}_fpa.png", dpi=130)
            plt.show(block=False)
            self.status.set(f"SAVED recordings/{self.tag}_* ({counts} samples) | "
                            + A.summary_text(result))
            self.result.set("")
            if result["warnings"]:
                self.warning.set("⚠ " + " ".join(result["warnings"]))
                messagebox.showwarning("Check the result", "\n\n".join(result["warnings"]))
        except Exception as exc:
            self.status.set(f"SAVED recordings/{self.tag}_* ({counts} samples), "
                            "but the analysis failed.")
            messagebox.showwarning("Analysis error", f"CSV files saved, but analysis failed:\n{exc}")

    # ----- drawing -----
    def draw(self):
        c = self.canvas
        c.delete("all")
        w = max(600, c.winfo_width())
        x, y = w / 2, 175
        c.create_text(x, 22, text=f"{S.FOOT_SIDE.upper()} FOOT, seen from above",
                      font=("Segoe UI", 13, "bold"), fill="#2563eb")
        c.create_line(x, 300, x, 45, dash=(6, 6), width=2, fill="#a0a0a0")
        c.create_text(x + 70, 50, text="walking direction (pelvis)", fill="#808080")
        if self.live.values:
            fpa = self.live.values[-1]
            a = math.radians(fpa)
            # toe-out turns the left foot's toes to the left, the right foot's to the right
            dx = -math.sin(a) if S.FOOT_SIDE == "left" else math.sin(a)
            dy = -math.cos(a)
            c.create_line(x - 50 * dx, y - 50 * dy, x + 115 * dx, y + 115 * dy,
                          fill="#2563eb", width=14, capstyle="round")
            c.create_oval(x - 7, y - 7, x + 7, y + 7, fill="white", outline="#2563eb", width=2)
            kind = "toe-out" if fpa >= 0 else "toe-in"
            c.create_text(x, 318, text=f"last step {fpa:+.1f}° ({kind})",
                          font=("Segoe UI", 13, "bold"), fill="#2563eb")
        else:
            c.create_text(x, 318, text="no steps yet", fill="#808080")

        lines = []
        for role in ROLES:
            reader, s = self.readers.get(role), self.last[role]
            if reader is None:
                continue
            algo = self.algorithms.get(role) or ("demo" if self.demo else "mode ?")
            horiz = self.live.f_horiz if role == "foot" else self.live.p_horiz
            hz = f" axis-horiz={horiz:.2f}" if horiz is not None else ""
            flag = "  ⚠ forward axis vertical?" if horiz is not None and horiz < S.MIN_HORIZONTAL else ""
            if s is not None:
                rate, new = reader.meter.rates()
                slow = ("  ⚠ below target" if reader.ok_count > 3 * POLL_HZ
                        and rate < 0.8 * POLL_HZ else "")
                lines.append(f"{role.upper():6s} {reader.port:>8s} {algo:7s} "
                             f"{rate:5.0f} Hz ({new:3.0f} new) "
                             f"gyro={np.linalg.norm(s.gyro):6.1f}°/s ok/err={reader.ok_count}/"
                             f"{reader.error_count}{hz}{flag}{slow} {reader.last_error or ''}")
            else:
                lines.append(f"{role.upper():6s} {reader.port:>8s} waiting for data "
                             f"{reader.last_error or ''}")
        self.info.set("\n".join(lines))

    def close(self):
        if self.mode != "READY":
            if not messagebox.askyesno("Recording active", "Discard this recording and exit?"):
                return
        for reader in self.readers.values():
            reader.stop()
        self.destroy()


def main():
    p = argparse.ArgumentParser(description="Foot + pelvis FPA recorder")
    p.add_argument("--demo", action="store_true", help="simulated sensors (no hardware)")
    args = p.parse_args()
    FPARecorder(demo=args.demo).mainloop()


if __name__ == "__main__":
    main()
