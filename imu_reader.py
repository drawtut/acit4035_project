"""Dual WT901C485-E acquisition using Modbus RTU with Live Logging & Plotting.

Set LEFT_PORT and RIGHT_PORT below. Both are independently polled USB/RS485 ports.
Requirements: pyserial, numpy, matplotlib
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
import threading
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import serial

# ------------ EDIT THESE ------------
LEFT_PORT = "COM5"
RIGHT_PORT = "COM4"
BAUDRATE = 9600
MODBUS_ADDRESS = 0x50
POLL_HZ = 10.0  # practical starting point for 9600 baud
# ------------------------------------

G = 9.80665


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if (crc & 1) else 0)
    return crc


@dataclass(frozen=True)
class IMUSample:
    timestamp: float  # monotonic clock, seconds
    accel: np.ndarray  # m/s^2, sensor frame, x/y/z
    gyro: np.ndarray   # deg/s, sensor frame, x/y/z
    magnet: np.ndarray # raw counts, sensor frame
    euler: np.ndarray  # degrees, roll/pitch/yaw from device

    @property
    def roll(self):
        return float(self.euler[0])

    @property
    def pitch(self):
        return float(self.euler[1])

    @property
    def yaw(self):
        return float(self.euler[2])


class IMUReader:
    """Thread-safe reader for one sensor. Read samples via get_latest()."""

    def __init__(self, port: str, *, baudrate=BAUDRATE,
                 address=MODBUS_ADDRESS, poll_hz=POLL_HZ):
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.poll_hz = poll_hz
        self._serial: Optional[serial.Serial] = None
        self._sample: Optional[IMUSample] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error = None
        self.ok_count = 0
        self.error_count = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def start(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError('Already started')
        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.25,
                                     bytesize=8, parity='N', stopbits=1)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._serial:
            self._serial.close()
            self._serial = None

    def get_latest(self) -> Optional[IMUSample]:
        with self._lock:
            return self._sample

    def _read_once(self) -> IMUSample:
        # 0x34..0x3F inclusive: 3 accel, 3 gyro, 3 mag, 3 Euler regs
        start, count = 0x34, 12
        cmd = struct.pack('>BBHH', self.address, 3, start, count)
        cmd += struct.pack('<H', crc16(cmd))
        s = self._serial
        s.reset_input_buffer()
        s.write(cmd)
        s.flush()
        h = s.read(3)
        if len(h) != 3:
            raise TimeoutError('No Modbus header')
        addr, func, byte_count = h
        if addr != self.address:
            raise ValueError('Unexpected device address')
        if func & 0x80:
            error = s.read(3)
            raise ValueError(f'Modbus exception {error.hex()}')
        if func != 3 or byte_count != 2 * count:
            raise ValueError('Unexpected function or register count')
        rest = s.read(byte_count + 2)
        if len(rest) != byte_count + 2:
            raise TimeoutError('Incomplete reply')
        packet = h + rest
        if crc16(packet[:-2]) != struct.unpack('<H', packet[-2:])[0]:
            raise ValueError('CRC mismatch')
        vals = np.array(struct.unpack('>' + 'h' * count, rest[:-2]), dtype=float)
        return IMUSample(
            timestamp=time.monotonic(),
            accel=vals[0:3] * (16 * G / 32768),
            gyro=vals[3:6] * (2000 / 32768),
            magnet=vals[6:9],
            euler=vals[9:12] * (180 / 32768),
        )

    def _loop(self):
        period = 1.0 / self.poll_hz
        while not self._stop.is_set():
            tic = time.monotonic()
            try:
                sample = self._read_once()
                with self._lock:
                    self._sample = sample
                self.ok_count += 1
                self.last_error = None
            except (ValueError, TimeoutError, serial.SerialException, OSError) as exc:
                self.error_count += 1
                self.last_error = str(exc)
            self._stop.wait(max(0.0, period - (time.monotonic() - tic)))


if __name__ == '__main__':
    print(f'Opening left={LEFT_PORT}, right={RIGHT_PORT}. Ctrl+C to stop and plot.')

    # Storage lists for logging
    left_t, left_euler, left_gyro = [], [], []
    right_t, right_euler, right_gyro = [], [], []

    start_time = time.monotonic()

    with IMUReader(LEFT_PORT) as left, IMUReader(RIGHT_PORT) as right:
        try:
            while True:
                l, r = left.get_latest(), right.get_latest()
                
                # Log left data
                if l:
                    left_t.append(l.timestamp - start_time)
                    left_euler.append(l.euler.copy())
                    left_gyro.append(l.gyro.copy())

                # Log right data
                if r:
                    right_t.append(r.timestamp - start_time)
                    right_euler.append(r.euler.copy())
                    right_gyro.append(r.gyro.copy())

                def fmt(s):
                    return ('roll/pitch/yaw=' + np.array2string(s.euler, precision=1)
                            + '  gyro=' + np.array2string(s.gyro, precision=1)) if s else 'waiting'
                print(f'L: {fmt(l)} | R: {fmt(r)}')

                if l is None and left.last_error:
                    print('LEFT ERROR:', left.last_error)
                if r is None and right.last_error:
                    print('RIGHT ERROR:', right.last_error)
                
                time.sleep(0.3)

        except KeyboardInterrupt:
            print('\nStopping acquisition... Generating graphs...')

    # --- Plotting Phase ---
    def plot_sensor_data(title_prefix, t_list, euler_list, gyro_list):
        if not t_list:
            print(f'No data recorded for {title_prefix}.')
            return

        t = np.array(t_list)
        euler = np.array(euler_list)  # columns: Roll, Pitch, Yaw
        gyro = np.array(gyro_list)    # columns: Gyro X, Y, Z

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f'{title_prefix} IMU Data', fontsize=16)

        # Plot Roll, Pitch, Yaw
        ax1.plot(t, euler[:, 0], label='Roll (°)', color='r')
        ax1.plot(t, euler[:, 1], label='Pitch (°)', color='g')
        ax1.plot(t, euler[:, 2], label='Yaw (°)', color='b')
        ax1.set_ylabel('Angle (degrees)')
        ax1.grid(True)
        ax1.legend(loc='upper right')
        ax1.set_title('Orientation (Euler Angles)')

        # Plot Gyroscope X, Y, Z
        ax2.plot(t, gyro[:, 0], label='Gyro X (°/s)', color='r')
        ax2.plot(t, gyro[:, 1], label='Gyro Y (°/s)', color='g')
        ax2.plot(t, gyro[:, 2], label='Gyro Z (°/s)', color='b')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Angular Velocity (°/s)')
        ax2.grid(True)
        ax2.legend(loc='upper right')
        ax2.set_title('Gyroscope')

        plt.tight_layout()

    # Generate graphs for both sensors
    plot_sensor_data('Left Leg', left_t, left_euler, left_gyro)
    plot_sensor_data('Right Leg', right_t, right_euler, right_gyro)

    plt.show()