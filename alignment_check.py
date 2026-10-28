"""WT901C485-E: dual thigh orientation / relative movement monitor.

Uses the EXISTING imu_reader.py in the same folder. No hardware config writes.

C or the visible CALIBRATE button: capture a steady neutral pose (4 seconds)
Y/X/Z: choose the physical sensor axis along the thigh (Y for photo)
R : reset the reference pose
Q : quit

Crucial distinction:
    A common neutral pose establishes relative MOVEMENT, not independently
    verified absolute parallelism or accurate sensor-to-femur alignment.

No experimental negative Euler signs. Per-device reference rotations cancel
constant sensor heading offsets in its *own* reference frame. The resulting
neutral frames are treated as comparable only if the sensors were mounted
with matching coordinate directions in the reference pose.
"""
from collections import deque
import time
import struct
import serial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from imu_reader import IMUReader, LEFT_PORT, RIGHT_PORT, BAUDRATE, MODBUS_ADDRESS, crc16

# ---------------- EDITABLE SETTINGS ----------------
THIGH_AXIS = 'y'             # photo: printed Y axis approximately along thigh
PARALLEL_TOLERANCE_DEG = 8.0
UPDATE_MS = 120
MAX_SAMPLE_AGE_S = 0.65
MAX_PAIR_SKEW_S = 0.20
REFERENCE_SECONDS = 4.0
MAX_REFERENCE_GYRO_DPS = 10.0
MIN_REFERENCE_PAIRS = 5
# If your two sensor axes do NOT have matched physical directions at the
# reference pose, remount them consistently before recording the reference.
# ---------------------------------------------------
AXES = {'x': np.array([1., 0., 0.]),
        'y': np.array([0., 1., 0.]),
        'z': np.array([0., 0., 1.])}


def sensor_rotation(sample):
    """World-from-sensor, assuming device roll/pitch/yaw uses ZYX composition.

    Euler convention must still be checked against known physical rotations.
    Do not negate Euler values to fix screen perspective.
    """
    return Rotation.from_euler('xyz', np.asarray(sample.euler, float), degrees=True)


def relative_rotation(current, neutral):
    """Current body axes in the neutral SENSOR coordinate system.

    R_neutral^-1 * R_current cancels a fixed world heading bias for this sensor.
    """
    return neutral.inv() * current


def angle_between_axes(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(a, b)), 0, 1))))


def axis_motion_angle(r, axis):
    v = r.apply(AXES[axis])
    return angle_between_axes(v, AXES[axis])


def get_pair(left, right, now):
    l, r = left.get_latest(), right.get_latest()
    valid = (l is not None and r is not None and
             0 <= now-l.timestamp < MAX_SAMPLE_AGE_S and
             0 <= now-r.timestamp < MAX_SAMPLE_AGE_S and
             abs(l.timestamp-r.timestamp) <= MAX_PAIR_SKEW_S)
    return (l, r) if valid else (None, None)


def draw_body(ax, rot, center, facecolor, axis):
    # Flat sensor, its printed label faces local +Z.
    points = np.array([[x, y, z] for x in (-.42, .42)
                       for y in (-.32, .32) for z in (-.075, .075)])
    world = rot.apply(points) + center
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    ax.add_collection3d(Poly3DCollection([world[f] for f in faces],
                                         facecolors=facecolor, edgecolor='#273342',
                                         linewidths=.8, alpha=.7))
    for name, color in [('x', '#de4545'), ('y', '#20a366'), ('z', '#3474db')]:
        v = rot.apply(AXES[name]); length = .80 if name == axis else .48
        xyz = np.stack((center, center+length*v))
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], color=color,
                linewidth=3.5 if name == axis else 2)
    v = rot.apply(AXES[axis])
    xyz = np.stack((center-.95*v, center+.95*v))
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], color='#087b91', alpha=.6, lw=3)


# WT901C485-E Modbus RTU magnetic calibration registers.
REG_UNLOCK = 0x69
REG_CALSW = 0x01
REG_SAVE = 0x00


def modbus_write(port, register, value):
    """Write and check the Modbus function-06 echo/CRC on a released COM port."""
    with serial.Serial(port, baudrate=BAUDRATE, bytesize=8,
                       parity='N', stopbits=1, timeout=1.0,
                       write_timeout=1.0) as ser:
        request = struct.pack('>BBHH', MODBUS_ADDRESS, 6, register, value)
        request += struct.pack('<H', crc16(request))
        ser.reset_input_buffer()
        ser.write(request)
        ser.flush()
        response = ser.read(8)
        if len(response) != 8:
            raise TimeoutError(f'{port}: no complete Modbus write response')
        if response[0] != MODBUS_ADDRESS:
            raise RuntimeError(f'{port}: unexpected slave address')
        if response[1] == 0x86:
            raise RuntimeError(f'{port}: Modbus exception {response[2]}')
        if response != request:
            raise RuntimeError(f'{port}: invalid Modbus write echo or CRC')


def mag_command(port, start):
    # Unlock, set calibration mode. End: switch back to normal and save.
    modbus_write(port, REG_UNLOCK, 0xB588)
    time.sleep(.10)
    modbus_write(port, REG_CALSW, 0x0007 if start else 0x0000)
    time.sleep(.10)
    if not start:
        modbus_write(port, REG_SAVE, 0x0000)
        time.sleep(.20)


class App:
    def __init__(self):
        self.left = IMUReader(LEFT_PORT).start()
        try:
            self.right = IMUReader(RIGHT_PORT).start()
        except Exception:
            self.left.stop()
            raise
        self.mag_active = False
        self.readers_running = True
        self.mag_ports_started = []
        self.axis = THIGH_AXIS
        self.ref_left = self.ref_right = None
        self.capture_until = None
        self.ref_pairs = []
        self.capture_rejected_motion = 0
        self.capture_rejected_sync = 0
        self.calibration_status = ""
        self.last_captured_ts = None
        self.history = deque(maxlen=230)
        self.t0 = time.monotonic()
        self.fig = plt.figure(figsize=(13.5, 8))
        self.fig.suptitle('WT901C485-E | Relative thigh movement', fontsize=17)
        self.status_ax = self.fig.add_axes((.06, .79, .88, .12)); self.status_ax.axis('off')
        self.ax3d = self.fig.add_axes((.045, .21, .52, .56), projection='3d')
        self.axplot = self.fig.add_axes((.62, .37, .34, .34))
        self.details = self.fig.text(.62, .31, '', family='monospace', fontsize=10, va='top')
        self.fig.text(.06, .08, 'C/CALIBRATE: neutral pose   R/RESET: clear   START/FINISH MAG: magnetic calibration   Q: quit', fontsize=10)
        self.fig.text(.06, .049,
             'Both IMUs should be mounted with matching local axes; Y along thigh in your photo.', fontsize=9)
        self.fig.text(.06, .027,
             'Reference-relative angles are NOT independently verified absolute anatomical alignment.', fontsize=9, color='#555f6d')
        # Visible controls work even when Matplotlib's plot window lacks keyboard focus.
        self.calibrate_button_ax = self.fig.add_axes((.06, .105, .16, .05))
        self.calibrate_button = Button(self.calibrate_button_ax, 'CALIBRATE (C)')
        self.calibrate_button.on_clicked(self.start_capture)
        self.reset_button_ax = self.fig.add_axes((.24, .105, .13, .05))
        self.reset_button = Button(self.reset_button_ax, 'RESET (R)')
        self.reset_button.on_clicked(self.reset_reference)
        self.mag_start_ax = self.fig.add_axes((.41, .105, .22, .05))
        self.mag_start_button = Button(self.mag_start_ax, 'START MAG CAL')
        self.mag_start_button.on_clicked(self.start_magnetometer_cal)
        self.mag_finish_ax = self.fig.add_axes((.65, .105, .24, .05))
        self.mag_finish_button = Button(self.mag_finish_ax, 'FINISH MAG CAL')
        self.mag_finish_button.on_clicked(self.finish_magnetometer_cal)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda _: self.close())
        self.anim = FuncAnimation(self.fig, self.update, interval=UPDATE_MS,
                                  cache_frame_data=False)

    def stop_readers(self):
        if self.readers_running:
            self.left.stop()
            self.right.stop()
            self.readers_running = False
            # Allow serial threads to release the ports before opening them.
            time.sleep(.20)

    def resume_readers(self):
        if self.readers_running:
            return
        self.left = IMUReader(LEFT_PORT).start()
        try:
            self.right = IMUReader(RIGHT_PORT).start()
        except Exception:
            self.left.stop()
            raise
        self.readers_running = True

    def start_magnetometer_cal(self, event=None):
        if self.mag_active:
            print('Magnetic calibration is already active.')
            return
        self.capture_until = None
        self.reset_reference()
        self.stop_readers()
        self.mag_ports_started = []
        try:
            for port in (LEFT_PORT, RIGHT_PORT):
                mag_command(port, start=True)
                self.mag_ports_started.append(port)
                print(f'{port}: magnetic calibration started', flush=True)
            self.mag_active = True
            self.calibration_status = ('MAG CAL ACTIVE: rotate EACH SENSOR slowly '
                                       '360 degrees around X, Y and Z; click FINISH')
        except Exception as exc:
            print(f'Magnetic calibration start error: {exc}', flush=True)
            # Return devices that started successfully to normal mode.
            for port in self.mag_ports_started:
                try:
                    mag_command(port, start=False)
                except Exception as cleanup_exc:
                    print(f'{port}: manual recovery required: {cleanup_exc}', flush=True)
            self.mag_ports_started = []
            self.mag_active = False
            self.calibration_status = f'MAG CAL START FAILED: {exc}'
            try:
                self.resume_readers()
            except Exception as restart_exc:
                print(f'Could not restart readers: {restart_exc}', flush=True)
        self.fig.canvas.draw_idle()

    def finish_magnetometer_cal(self, event=None):
        if not self.mag_active:
            print('No magnetic calibration is active.')
            return
        errors = []
        for port in self.mag_ports_started:
            try:
                mag_command(port, start=False)
                print(f'{port}: magnetic calibration ended and save requested', flush=True)
            except Exception as exc:
                errors.append(f'{port}: {exc}')
        self.mag_active = False
        self.mag_ports_started = []
        if errors:
            self.calibration_status = 'MAG CAL ERROR: ' + '; '.join(errors)
            print(self.calibration_status, flush=True)
            print('Check/calibrate these sensors with the vendor tool before use.', flush=True)
        else:
            self.calibration_status = ('MAG CAL FINISHED. Remount sensors and '
                                       'press CALIBRATE to capture a new neutral pose.')
        try:
            self.resume_readers()
        except Exception as exc:
            self.calibration_status += f' | reader restart failed: {exc}'
            print(self.calibration_status, flush=True)
        self.fig.canvas.draw_idle()

    def close(self):
        if self.mag_active:
            # Never intentionally leave a device in calibration mode.
            self.stop_readers()
            for port in self.mag_ports_started:
                try:
                    mag_command(port, start=False)
                except Exception as exc:
                    print(f'WARNING: {port} may still be in MAG CAL mode: {exc}')
            self.mag_active = False
        self.stop_readers()

    def start_capture(self, event=None):
        if self.mag_active or not self.readers_running:
            print('Finish magnetic calibration and restore readers first.')
            return
        self.capture_until = time.monotonic() + REFERENCE_SECONDS
        self.ref_pairs = []
        self.capture_rejected_motion = 0
        self.capture_rejected_sync = 0
        self.calibration_status = "CAPTURING: hold both sensors still for 4 seconds"
        self.last_captured_ts = None
        # Keep the previous valid calibration until a new one succeeds.
        self.history.clear()
        print("Calibration started — hold still for 4 seconds", flush=True)
        self.fig.canvas.draw_idle()

    def reset_reference(self, event=None):
        self.capture_until = None
        self.ref_pairs = []
        self.ref_left = self.ref_right = None
        self.calibration_status = "Reference reset. Press C or click CALIBRATE."
        self.history.clear()
        print(self.calibration_status, flush=True)
        self.fig.canvas.draw_idle()

    def on_key(self, e):
        key = (e.key or '').lower()
        if key in AXES:
            self.axis = key
            self.history.clear()
        elif key == 'c':
            self.start_capture()
        elif key == 'r':
            self.reset_reference()
        elif key == 'q':
            plt.close(self.fig)

    def capture(self, now, l, r):
        if self.capture_until is None:
            return ''
        if now < self.capture_until:
            if l is None or r is None:
                self.capture_rejected_sync += 1
            else:
                stamp = (l.timestamp, r.timestamp)
                if stamp != self.last_captured_ts:
                    self.last_captured_ts = stamp
                    if max(np.linalg.norm(l.gyro), np.linalg.norm(r.gyro)) <= MAX_REFERENCE_GYRO_DPS:
                        self.ref_pairs.append((sensor_rotation(l), sensor_rotation(r)))
                    else:
                        self.capture_rejected_motion += 1
            self.calibration_status = (
                f'CAPTURING: {max(0, self.capture_until-now):.1f}s left, '
                f'{len(self.ref_pairs)}/{MIN_REFERENCE_PAIRS} steady pairs '
                f'(moving: {self.capture_rejected_motion}, stale/skewed: {self.capture_rejected_sync})'
            )
            return self.calibration_status
        self.capture_until = None
        accepted = len(self.ref_pairs)
        if accepted < MIN_REFERENCE_PAIRS:
            self.ref_pairs = []
            self.calibration_status = (
                f'REFERENCE FAILED: only {accepted}/{MIN_REFERENCE_PAIRS} steady pairs. '
                f'Motion rejects: {self.capture_rejected_motion}; '
                f'stale/skewed: {self.capture_rejected_sync}. '
                'Keep still, check sensor connections, then press C to retry.'
            )
            print(self.calibration_status)
            return self.calibration_status
        self.ref_left = Rotation.concatenate([p[0] for p in self.ref_pairs]).mean()
        self.ref_right = Rotation.concatenate([p[1] for p in self.ref_pairs]).mean()
        self.ref_pairs = []
        self.calibration_status = f'REFERENCE SAVED ({accepted} pairs). Press R to reset.'
        print(self.calibration_status)
        self.history.clear()
        return self.calibration_status

    def update(self, _):
        now = time.monotonic()
        if self.mag_active or not self.readers_running:
            self.status_ax.clear()
            self.status_ax.axis('off')
            self.status_ax.text(.02, .5, self.calibration_status,
                                transform=self.status_ax.transAxes,
                                color='#9c6500', fontsize=11, va='center')
            return
        l, r = get_pair(self.left, self.right, now)
        capture_msg = self.capture(now, l, r)
        self.status_ax.clear(); self.status_ax.axis('off'); self.ax3d.cla(); self.axplot.cla()
        has_reference = self.ref_left is not None and self.ref_right is not None
        if l is None or r is None:
            headline = self.calibration_status or 'WAITING FOR FRESH DATA'
            color = '#886600'
            self.details.set_text(f'LEFT: {self.left.last_error or "waiting / stale"}\n'
                                  f'RIGHT: {self.right.last_error or "waiting / stale"}')
        else:
            raw_l, raw_r = sensor_rotation(l), sensor_rotation(r)
            # Raw orientations are displayed only as a diagnostic, not a pass/fail.
            raw_diff = angle_between_axes(raw_l.apply(AXES[self.axis]),
                                          raw_r.apply(AXES[self.axis]))
            if has_reference:
                rel_l = relative_rotation(raw_l, self.ref_left)
                rel_r = relative_rotation(raw_r, self.ref_right)
                vec_l = rel_l.apply(AXES[self.axis])
                vec_r = rel_r.apply(AXES[self.axis])
                difference = angle_between_axes(vec_l, vec_r)
                motion_l = axis_motion_angle(rel_l, self.axis)
                motion_r = axis_motion_angle(rel_r, self.axis)
                pass_check = difference <= PARALLEL_TOLERANCE_DEG
                headline = f'{"GREEN" if pass_check else "RED"} | REFERENCE-RELATIVE AXES {"MATCH" if pass_check else "DIFFER"}: {difference:.1f}°'
                color = '#168048' if pass_check else '#c0392b'
                self.history.append((now-self.t0, difference, motion_l, motion_r))
                use_l, use_r = rel_l, rel_r
                self.details.set_text(
                    f'Comparison axis: {self.axis.upper()}\n'
                    f'LEFT axis move:   {motion_l:5.1f}° from neutral\n'
                    f'RIGHT axis move:  {motion_r:5.1f}° from neutral\n'
                    f'RELATIVE axes:    {difference:5.1f}° (reference-relative)\n'
                    f'RAW Y/X/Z diff:   {raw_diff:5.1f}° ({self.axis.upper()}, unverified heading)\n'
                    f'Raw L RPY: {np.round(l.euler,1)}\n'
                    f'Raw R RPY: {np.round(r.euler,1)}\n'
                    f'Pair skew: {1000*abs(l.timestamp-r.timestamp):.0f} ms')
            else:
                use_l, use_r = Rotation.identity(), Rotation.identity()
                headline = self.calibration_status or 'NO REFERENCE | Stand or sit in a known common pose, press C'
                color = '#8d6800'
                self.details.set_text(
                    f'Axis: {self.axis.upper()}\n'
                    f'Uncalibrated raw axis difference: {raw_diff:.1f}°\n'
                    f'Raw L RPY: {np.round(l.euler,1)}\n'
                    f'Raw R RPY: {np.round(r.euler,1)}\n'
                    'Capture a known common pose to compare subsequent movement.')
            draw_body(self.ax3d, use_l, np.array([-1.2,0.,0.]), '#55a9e9', self.axis)
            draw_body(self.ax3d, use_r, np.array([1.2,0.,0.]), '#f5b94e', self.axis)
            self.ax3d.text(-1.2,-.85,-.7,'LEFT',ha='center')
            self.ax3d.text(1.2,-.85,-.7,'RIGHT',ha='center')
        if self.capture_until is not None:
            headline = self.calibration_status
            color = '#8d6800'
            self.details.set_text(self.calibration_status + '\n' + self.details.get_text())
        elif self.calibration_status and has_reference:
            self.details.set_text(self.calibration_status + '\n' + self.details.get_text())
        self.status_ax.add_patch(Rectangle((0,.06),1,.88,transform=self.status_ax.transAxes,
                                           color=color,alpha=.11))
        self.status_ax.text(.025,.56,headline,transform=self.status_ax.transAxes,
                            color=color,fontsize=17,weight='bold',va='center')
        self.ax3d.set(xlim=(-2.4,2.4),ylim=(-1.6,1.6),zlim=(-1.5,1.5))
        self.ax3d.set_box_aspect((4.8,3.2,3))
        self.ax3d.view_init(elev=25,azim=-57)
        self.ax3d.set_title('Relative-to-neutral sensor bodies (X red, Y green, Z blue)')
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        if self.history:
            data = np.array(self.history)
            self.axplot.plot(data[:,0],data[:,1],label='L vs R axis',lw=2)
            self.axplot.plot(data[:,0],data[:,2],label='Left vs neutral',alpha=.75)
            self.axplot.plot(data[:,0],data[:,3],label='Right vs neutral',alpha=.75)
            end=data[-1,0]
            self.axplot.set_xlim(max(0,end-20),max(20,end))
        self.axplot.axhline(PARALLEL_TOLERANCE_DEG,color='#c0392b',ls='--',lw=1)
        self.axplot.set_ylim(0,90); self.axplot.grid(alpha=.2)
        self.axplot.set_title('Relative movement from common neutral')
        self.axplot.set_ylabel('Axis angle (degrees)');self.axplot.set_xlabel('Elapsed seconds')
        if self.history:
            self.axplot.legend(loc='upper left',fontsize=8)


def main():
    print(f'Left: {LEFT_PORT} | Right: {RIGHT_PORT}')
    print('Mag calibration writes to BOTH devices; they must be in 9-axis mode.')
    app = App()
    try:
        plt.show()
    finally:
        app.close()


if __name__ == '__main__':
    main()
