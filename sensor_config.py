
"""Configure WT901C485-E installation direction and sensor fusion mode.

Requires:
    pip install pyserial

Uses Modbus RTU and the existing imu_reader.py configuration.

Registers:
    0x04 BAUD: 2 = 9600 (default) ... 6 = 115200, 7 = 230400
    0x1F BANDWIDTH: 0 = 256 Hz, 1 = 188, 2 = 98, 3 = 42, 4 = 20 (default), 5 = 10, 6 = 5
    0x23 ORIENT: 0 = horizontal, 1 = vertical
    0x24 AXIS6: 0 = 9-axis, 1 = 6-axis (no magnetometer fusion)
    0x74 MODDELAY: delay before each Modbus reply, microseconds (default 3000)

Close any applications using the COM ports before running.
--baud is the baud rate the sensor uses NOW (default: BAUDRATE in imu_reader.py).

Examples:
    python sensor_config.py read
    python sensor_config.py six-axis --yes
    python sensor_config.py nine-axis --yes
    python sensor_config.py vertical --yes
    python sensor_config.py horizontal --yes
    python sensor_config.py vertical --sensor left --yes
    python sensor_config.py bandwidth 98 --yes
    python sensor_config.py moddelay 500 --yes
    python sensor_config.py baud 230400 --baud 9600 --yes   (change 9600 -> 230400)
    python sensor_config.py scan          (find which baud rate a sensor uses)
    python sensor_config.py magcal --yes  (magnetic calibration, 9-axis mode only)

Run WITHOUT a command (e.g. the Run button in Spyder) to do the complete
setup automatically, using the SETUP_... values below. It finds each
sensor's current baud rate, applies mode, reply delay, bandwidth, optional
timed magnetic calibration and finally the baud rate. Settings that are
already correct are not written again, so it is safe to run every session.
    python sensor_config.py menu     numbered menu instead
"""

from __future__ import annotations

import argparse
import shlex
import struct
import sys
import time

import serial

from imu_reader import (
    LEFT_PORT,
    RIGHT_PORT,
    BAUDRATE,
    MODBUS_ADDRESS,
    crc16,
)


# ============================================================
# Register addresses
# ============================================================

REG_SAVE = 0x00
REG_CALSW = 0x01          # calibration mode (same values as alignment_check.py)
MAG_CAL_START = 0x0007
CAL_NORMAL = 0x0000
REG_BAUD = 0x04
REG_BANDWIDTH = 0x1F
REG_ORIENT = 0x23
REG_AXIS6 = 0x24
REG_UNLOCK = 0x69
REG_MODDELAY = 0x74

BAUD_CODES = {4800: 0x01, 9600: 0x02, 19200: 0x03, 38400: 0x04,
              57600: 0x05, 115200: 0x06, 230400: 0x07}
BANDWIDTH_CODES = {256: 0x00, 188: 0x01, 98: 0x02, 42: 0x03,
                   20: 0x04, 10: 0x05, 5: 0x06}
SCAN_BAUDS = [9600, 230400, 115200, 57600, 38400, 19200, 4800]


# ============================================================
# Automatic setup (Run button / no command). Edit these.
# ============================================================

SETUP_MODE = "9-axis"          # "9-axis" (magnetometer) or "6-axis" (no magnetometer)
SETUP_MAGCAL_SECONDS = 45      # 9-axis only: rotate the sensors during this countdown.
                               # 0 = skip (e.g. if already calibrated today)
SETUP_MODDELAY_US = 500        # sensor reply delay; needed for 200 Hz
SETUP_BANDWIDTH_HZ = 98        # 256/188/98/42/20/10/5
SETUP_BAUD = BAUDRATE          # = BAUDRATE in imu_reader.py (230400)

# Installation direction
HORIZONTAL = 0
VERTICAL = 1

# Sensor fusion algorithm
NINE_AXIS = 0
SIX_AXIS = 1


# ============================================================
# Modbus communication
# ============================================================

def transact(
    ser: serial.Serial,
    addr: int,
    function: int,
    register: int,
    value: int,
) -> bytes:

    request = struct.pack(
        ">BBHH",
        addr,
        function,
        register,
        value,
    )

    request += struct.pack("<H", crc16(request))

    ser.reset_input_buffer()
    ser.write(request)
    ser.flush()

    header = ser.read(3)

    if len(header) != 3:
        raise TimeoutError(
            "No Modbus reply "
            "(check power, COM port, baud, address)"
        )

    if header[0] != addr:
        raise ValueError("Unexpected slave address")

    if header[1] == (function | 0x80):
        rest = ser.read(2)

        raise ValueError(
            f"Modbus exception {header[2]} "
            f"(raw={(header + rest).hex(' ')})"
        )

    if header[1] != function:
        raise ValueError("Unexpected Modbus function")

    if function == 3:

        byte_count = header[2]

        if byte_count != value * 2:
            raise ValueError(
                "Unexpected register byte count"
            )

        response = header + ser.read(byte_count + 2)
        expected_length = 3 + byte_count + 2

    elif function == 6:

        response = header + ser.read(5)
        expected_length = 8

    else:
        raise ValueError("Unsupported function")

    if len(response) != expected_length:
        raise TimeoutError("Incomplete Modbus response")

    received_crc = struct.unpack(
        "<H", response[-2:]
    )[0]

    if received_crc != crc16(response[:-2]):
        raise ValueError("Modbus CRC mismatch")

    if function == 6 and response[:6] != request[:6]:
        raise ValueError(
            "Write echo does not match request"
        )

    return response


# ============================================================
# Register access
# ============================================================

def read_register(ser, addr, register):

    response = transact(
        ser,
        addr,
        3,
        register,
        1,
    )

    return struct.unpack(
        ">H",
        response[3:5],
    )[0]


def write_register(ser, addr, register, value):

    transact(
        ser,
        addr,
        6,
        register,
        value,
    )


def read_direction(ser, addr):
    return read_register(ser, addr, REG_ORIENT)


def read_algorithm(ser, addr):
    return read_register(ser, addr, REG_AXIS6)


# ============================================================
# Formatting
# ============================================================

def direction_name(value):

    return {
        HORIZONTAL: "HORIZONTAL",
        VERTICAL: "VERTICAL",
    }.get(value, f"UNKNOWN ({value})")


def algorithm_name(value):

    return {
        NINE_AXIS: "9-AXIS (magnetometer fusion enabled)",
        SIX_AXIS: "6-AXIS (magnetometer fusion disabled)",
    }.get(value, f"UNKNOWN ({value})")


def print_configuration(port, direction, algorithm, ser=None, addr=None):

    print(f"{port}: installation = {direction_name(direction)}")
    print(f"{port}: algorithm    = {algorithm_name(algorithm)}")

    if ser is None:
        return

    extras = (
        ("bandwidth", REG_BANDWIDTH,
         lambda v: next((f"{hz} Hz" for hz, c in BANDWIDTH_CODES.items() if c == v), f"code {v}")),
        ("moddelay ", REG_MODDELAY, lambda v: f"{v} us"),
        ("baud code", REG_BAUD,
         lambda v: next((f"{b} bps" for b, c in BAUD_CODES.items() if c == v), f"code {v}")),
    )
    for label, register, fmt in extras:
        try:
            print(f"{port}: {label}    = {fmt(read_register(ser, addr, register))}")
        except (TimeoutError, ValueError) as exc:
            print(f"{port}: {label}    = n/a ({exc})")


# ============================================================
# Configuration
# ============================================================

def open_port(port, baud, timeout=1.0):
    return serial.Serial(
        port,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=timeout,
        write_timeout=1.0,
    )


def scan_port(port, addr):
    """Try the common baud rates and report which one the sensor answers on."""

    for baud in SCAN_BAUDS:
        try:
            with open_port(port, baud, timeout=0.3) as ser:
                algorithm = read_algorithm(ser, addr)
            print(f"{port}: FOUND at {baud} baud "
                  f"({algorithm_name(algorithm)})")
            return baud
        except (serial.SerialException, TimeoutError, ValueError):
            print(f"{port}: no reply at {baud} baud")

    print(f"{port}: sensor not found at any baud rate "
          "(check power, COM port and Modbus address)")
    return None


def change_baud(port, from_baud, to_baud, addr):
    """Change the sensor's baud rate.

    The sensor may switch speed as soon as the BAUD register is written,
    so the save command is tried at the new speed first, then at the old."""

    code = BAUD_CODES[to_baud]

    with open_port(port, from_baud) as ser:
        print_configuration(port, read_direction(ser, addr),
                            read_algorithm(ser, addr))
        print(f"{port}: changing baud {from_baud} -> {to_baud}...")
        write_register(ser, addr, REG_UNLOCK, 0xB588)
        time.sleep(0.10)
        try:
            write_register(ser, addr, REG_BAUD, code)
        except (TimeoutError, ValueError):
            pass  # reply may already come at the new speed
        time.sleep(0.25)

    def save_at(baud):
        with open_port(port, baud) as ser:
            write_register(ser, addr, REG_UNLOCK, 0xB588)
            time.sleep(0.10)
            try:
                write_register(ser, addr, REG_SAVE, 0x0000)
            except (TimeoutError, ValueError):
                pass
        time.sleep(0.40)

    def verify_new():
        with open_port(port, to_baud) as ser:
            return read_register(ser, addr, REG_BAUD) == code

    for save_baud in (to_baud, from_baud):
        try:
            save_at(save_baud)
            if verify_new():
                print(f"{port}: SUCCESS, sensor now answers at {to_baud} baud "
                      "and the setting is saved")
                return True
        except (serial.SerialException, TimeoutError, ValueError):
            continue

    raise RuntimeError(
        f"Could not confirm the new baud rate. Power-cycle the sensor and run "
        f"'python sensor_config.py scan --sensor ...' to find its baud rate."
    )


def magnetic_calibration(ports, baud, addr, wait_for_user=input, bauds=None):
    """Calibrate the magnetometers of all given sensors at the same time.

    Saved only if the user finishes normally; on Ctrl+C the calibration mode
    is ended without saving. bauds: optional {port: baud} if they differ."""

    def baud_of(port):
        return bauds[port] if bauds else baud

    started = []
    finished_ok = False
    try:
        for label, port in ports:
            with open_port(port, baud_of(port)) as ser:
                if read_algorithm(ser, addr) == SIX_AXIS:
                    raise RuntimeError(
                        f"{label} is in 6-axis mode. Magnetic calibration needs "
                        "9-axis: python sensor_config.py nine-axis --yes")
                write_register(ser, addr, REG_UNLOCK, 0xB588)
                time.sleep(0.10)
                write_register(ser, addr, REG_CALSW, MAG_CAL_START)
                time.sleep(0.10)
            started.append((label, port))
            print(f"{label} ({port}): magnetic calibration started")

        print(
            "\nMAGNETIC CALIBRATION RUNNING\n"
            "  - Stand at least 1-2 m away from steel, laptops, tables, railings.\n"
            "  - Ideally keep each sensor mounted as for walking (on the shoe,\n"
            "    on the belt), so their own metal parts are calibrated too.\n"
            "  - Rotate EACH sensor slowly through every orientation: about\n"
            "    3 full turns around each of its X, Y and Z axes (30-60 s).\n"
        )
        wait_for_user("Press Enter when finished (Ctrl+C to abort without saving)... ")
        finished_ok = True

    finally:
        for label, port in started:
            try:
                with open_port(port, baud_of(port)) as ser:
                    write_register(ser, addr, REG_UNLOCK, 0xB588)
                    time.sleep(0.10)
                    write_register(ser, addr, REG_CALSW, CAL_NORMAL)
                    time.sleep(0.10)
                    if finished_ok:
                        write_register(ser, addr, REG_SAVE, 0x0000)
                        time.sleep(0.25)
                print(f"{label} ({port}): calibration "
                      + ("ended and SAVED" if finished_ok else
                         "ended WITHOUT saving (power-cycle to discard)"))
            except (serial.SerialException, TimeoutError, ValueError) as exc:
                print(f"{label} ({port}): could not end calibration: {exc}. "
                      "Power-cycle the sensor.")

    return finished_ok


def run_port(port, action, baud, addr, value=None):

    desired_register = None
    desired_value = None

    if action == "scan":
        return scan_port(port, addr) is not None

    if action == "baud":
        if value == baud:
            print(f"{port}: already at {baud} baud; no change needed")
            return True
        return change_baud(port, baud, value, addr)

    if action == "bandwidth":
        desired_register = REG_BANDWIDTH
        desired_value = BANDWIDTH_CODES[value]

    elif action == "moddelay":
        desired_register = REG_MODDELAY
        desired_value = value

    elif action == "vertical":
        desired_register = REG_ORIENT
        desired_value = VERTICAL

    elif action == "horizontal":
        desired_register = REG_ORIENT
        desired_value = HORIZONTAL

    elif action == "six-axis":
        desired_register = REG_AXIS6
        desired_value = SIX_AXIS

    elif action == "nine-axis":
        desired_register = REG_AXIS6
        desired_value = NINE_AXIS

    with serial.Serial(
        port,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1.0,
        write_timeout=1.0,
    ) as ser:

        before_direction = read_direction(ser, addr)
        before_algorithm = read_algorithm(ser, addr)

        print_configuration(
            port,
            before_direction,
            before_algorithm,
            ser if action == "read" else None,
            addr,
        )

        if action == "read":
            return True

        current_value = read_register(
            ser,
            addr,
            desired_register,
        )

        if current_value == desired_value:
            print(
                f"{port}: already configured; "
                "no write needed"
            )
            return True

        print(f"{port}: changing configuration...")

        # Unlock configuration registers
        write_register(
            ser,
            addr,
            REG_UNLOCK,
            0xB588,
        )

        time.sleep(0.10)

        # Apply requested configuration
        write_register(
            ser,
            addr,
            desired_register,
            desired_value,
        )

        time.sleep(0.10)

        # Verify before saving
        observed = read_register(
            ser,
            addr,
            desired_register,
        )

        if observed != desired_value:
            raise RuntimeError(
                "Pre-save verification failed: "
                f"expected {desired_value}, got {observed}"
            )

        # Save to sensor
        write_register(
            ser,
            addr,
            REG_SAVE,
            0x0000,
        )

        time.sleep(0.25)

        # Verify after saving
        saved_value = read_register(
            ser,
            addr,
            desired_register,
        )

        if saved_value != desired_value:
            raise RuntimeError(
                "Post-save verification failed: "
                f"expected {desired_value}, got {saved_value}"
            )

        print(f"{port}: save verification passed")

    # ========================================================
    # Reopen COM port and verify
    # ========================================================

    time.sleep(0.25)

    with serial.Serial(
        port,
        baudrate=baud,
        bytesize=8,
        parity="N",
        stopbits=1,
        timeout=1.0,
        write_timeout=1.0,
    ) as ser:

        final_direction = read_direction(ser, addr)
        final_algorithm = read_algorithm(ser, addr)
        final_value = read_register(ser, addr, desired_register)

    if final_value != desired_value:
        raise RuntimeError(
            "Reopen verification failed: "
            f"expected {desired_value}, got {final_value}"
        )

    print(f"{port}: SUCCESS")

    print_configuration(
        port,
        final_direction,
        final_algorithm,
    )

    return True


# ============================================================
# Main
# ============================================================

def countdown(seconds):
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            print("  done.", flush=True)
            return
        print(f"  rotate the sensors slowly in all directions... {left:3.0f} s left",
              flush=True)
        time.sleep(min(5.0, left))


def setup(addr=MODBUS_ADDRESS, magcal_seconds=None):
    """Complete sensor setup in one go, for both ports."""

    magcal_seconds = SETUP_MAGCAL_SECONDS if magcal_seconds is None else magcal_seconds
    if SETUP_MODE not in ("9-axis", "6-axis"):
        raise SystemExit("SETUP_MODE must be '9-axis' or '6-axis'")
    if SETUP_BANDWIDTH_HZ not in BANDWIDTH_CODES or SETUP_BAUD not in BAUD_CODES:
        raise SystemExit("Check SETUP_BANDWIDTH_HZ / SETUP_BAUD at the top of the file")

    ports = [("LEFT", LEFT_PORT), ("RIGHT", RIGHT_PORT)]
    do_magcal = SETUP_MODE == "9-axis" and magcal_seconds > 0
    print("AUTOMATIC SENSOR SETUP\n"
          f"  mode {SETUP_MODE}, reply delay {SETUP_MODDELAY_US} us, bandwidth "
          f"{SETUP_BANDWIDTH_HZ} Hz, baud {SETUP_BAUD}, magnetic calibration: "
          + (f"{magcal_seconds:.0f} s countdown" if do_magcal else "skipped"))

    def step(text):
        print(f"\n{'=' * 60}\n{text}\n{'=' * 60}")

    try:
        step("1. Finding the sensors")
        bauds = {}
        for label, port in ports:
            found = scan_port(port, addr)
            if found is None:
                raise RuntimeError(f"{label} sensor on {port} not found. Check power, "
                                   "cable and the COM port in imu_reader.py.")
            bauds[port] = found

        action = "nine-axis" if SETUP_MODE == "9-axis" else "six-axis"
        for label, port in ports:
            step(f"2. {label} ({port}): mode, reply delay, bandwidth")
            run_port(port, action, bauds[port], addr)
            run_port(port, "moddelay", bauds[port], addr, SETUP_MODDELAY_US)
            run_port(port, "bandwidth", bauds[port], addr, SETUP_BANDWIDTH_HZ)

        if do_magcal:
            step("3. Magnetic calibration (both sensors at the same time)")
            ok = magnetic_calibration(ports, None, addr, bauds=bauds,
                                      wait_for_user=lambda _: countdown(magcal_seconds))
            if not ok:
                raise RuntimeError("magnetic calibration was not completed")

        for label, port in ports:
            step(f"4. {label} ({port}): baud rate {SETUP_BAUD}")
            run_port(port, "baud", bauds[port], addr, SETUP_BAUD)

        step("5. Final check")
        for label, port in ports:
            run_port(port, "read", SETUP_BAUD, addr)

    except (serial.SerialException, TimeoutError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"\nSETUP STOPPED: {exc}")
    except KeyboardInterrupt:
        raise SystemExit("\nSETUP ABORTED by user.")

    if SETUP_BAUD != BAUDRATE:
        print(f"\nWARNING: set BAUDRATE = {SETUP_BAUD} in imu_reader.py "
              f"(it is {BAUDRATE}), otherwise the recorder cannot connect.")
    print("\nSETUP COMPLETE. Now open fpa_pelvis_recorder.py and press Run (F5).")


MENU = [
    ("read --baud 9600", "show settings (sensors still at factory 9600 baud)"),
    ("nine-axis --baud 9600 --yes", "9-axis mode (magnetometer on)"),
    ("six-axis --baud 9600 --yes", "6-axis mode (magnetometer off)"),
    ("magcal --baud 9600 --yes", "magnetic calibration (9-axis only)"),
    ("moddelay 500 --baud 9600 --yes", "reply delay 0.5 ms (needed for 200 Hz)"),
    ("bandwidth 98 --baud 9600 --yes", "bandwidth 98 Hz (fewer repeated samples)"),
    ("baud 230400 --baud 9600 --yes", "change 9600 -> 230400 baud (do this LAST)"),
    ("read", f"show settings at {BAUDRATE} baud (BAUDRATE in imu_reader.py)"),
    ("scan", "find which baud rate each sensor uses"),
]


def interactive():
    """Menu for running without command-line arguments (Spyder Run button)."""
    print("WT901C485-E sensor configuration")
    while True:
        print("\nChoose a number, or type a command (e.g. 'magcal --yes'):")
        for i, (cmd, text) in enumerate(MENU, 1):
            print(f"  {i}. {text:50s} [{cmd}]")
        print("  q. quit")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if choice.lower() in ("", "q", "quit", "exit"):
            return
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            choice = MENU[int(choice) - 1][0]
        print(f"\n>>> sensor_config.py {choice}")
        try:
            main(shlex.split(choice))
        except SystemExit:
            pass          # errors are already printed; show the menu again


def main(argv=None):

    parser = argparse.ArgumentParser(
        description=(
            "WT901C485-E installation direction "
            "and sensor fusion configuration"
        )
    )

    parser.add_argument(
        "action",
        choices=[
            "read",
            "vertical",
            "horizontal",
            "six-axis",
            "nine-axis",
            "bandwidth",
            "moddelay",
            "baud",
            "scan",
            "magcal",
        ],
        help="Read or change sensor configuration",
    )

    parser.add_argument(
        "value",
        nargs="?",
        type=int,
        help=("for bandwidth: Hz (256/188/98/42/20/10/5); "
              "for moddelay: microseconds; for baud: new baud rate"),
    )

    parser.add_argument(
        "--sensor",
        choices=[
            "left",
            "right",
            "both",
        ],
        default="both",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=BAUDRATE,
    )

    parser.add_argument(
        "--address",
        type=lambda s: int(s, 0),
        default=MODBUS_ADDRESS,
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm changing device configuration",
    )

    args = parser.parse_args(argv)

    if args.sensor == "left":

        ports = [
            ("LEFT", LEFT_PORT),
        ]

    elif args.sensor == "right":

        ports = [
            ("RIGHT", RIGHT_PORT),
        ]

    else:

        ports = [
            ("LEFT", LEFT_PORT),
            ("RIGHT", RIGHT_PORT),
        ]

    if args.action not in ("read", "scan") and not args.yes:

        parser.error(
            "Writing device settings requires --yes. "
            "Read current values first."
        )

    if args.action in ("bandwidth", "moddelay", "baud") and args.value is None:
        parser.error(f"'{args.action}' needs a value, e.g. "
                     "'bandwidth 98', 'moddelay 500' or 'baud 230400'.")

    if args.action == "bandwidth" and args.value not in BANDWIDTH_CODES:
        parser.error(f"bandwidth must be one of {sorted(BANDWIDTH_CODES)} Hz")

    if args.action == "moddelay" and not 0 <= args.value <= 65535:
        parser.error("moddelay must be 0-65535 microseconds")

    if args.action == "baud":
        for b in (args.value, args.baud):
            if b not in BAUD_CODES:
                parser.error(f"baud must be one of {sorted(BAUD_CODES)}")
        if args.value != BAUDRATE:
            print(f"NOTE: after this, set BAUDRATE = {args.value} in imu_reader.py "
                  f"(it is {BAUDRATE} now).")

    if args.action == "magcal":
        try:
            ok = magnetic_calibration(ports, args.baud, args.address)
        except (serial.SerialException, TimeoutError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            raise SystemExit(1)
        except KeyboardInterrupt:
            raise SystemExit("\nAborted; nothing was saved.")
        if not ok:
            raise SystemExit(1)
        print("\nMagnetic calibration completed.")
        return

    failed = False

    for label, port in ports:

        print(f"\n{'=' * 50}")
        print(f"{label} ({port})")
        print(f"{'=' * 50}")

        try:

            ok = run_port(
                port,
                args.action,
                args.baud,
                args.address,
                args.value,
            )
            if ok is False:
                failed = True

        except (
            serial.SerialException,
            TimeoutError,
            ValueError,
            RuntimeError,
        ) as exc:

            print(f"{label}: ERROR: {exc}")
            failed = True

    if failed:
        raise SystemExit(1)

    print("\nConfiguration completed successfully.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        setup()
    elif sys.argv[1:] == ["menu"]:
        interactive()
    else:
        main()
