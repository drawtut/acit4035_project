
"""Configure WT901C485-E installation direction and sensor fusion mode.

Requires:
    pip install pyserial

Uses Modbus RTU and the existing imu_reader.py configuration.

Registers:
    0x23 ORIENT: 0 = horizontal, 1 = vertical
    0x24 AXIS6: 0 = 9-axis, 1 = 6-axis (no magnetometer fusion)

Close any applications using the COM ports before running.

Examples:
    python sensor_config.py read
    python sensor_config.py six-axis --yes
    python sensor_config.py nine-axis --yes
    python sensor_config.py vertical --yes
    python sensor_config.py horizontal --yes
    python sensor_config.py vertical --sensor left --yes
"""

from __future__ import annotations

import argparse
import struct
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
REG_ORIENT = 0x23
REG_AXIS6 = 0x24
REG_UNLOCK = 0x69

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


def print_configuration(port, direction, algorithm):

    print(f"{port}: installation = {direction_name(direction)}")
    print(f"{port}: algorithm    = {algorithm_name(algorithm)}")


# ============================================================
# Configuration
# ============================================================

def run_port(port, action, baud, addr):

    desired_register = None
    desired_value = None

    if action == "vertical":
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

    final_value = (
        final_direction
        if desired_register == REG_ORIENT
        else final_algorithm
    )

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

def main():

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
        ],
        help="Read or change sensor configuration",
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

    args = parser.parse_args()

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

    if args.action != "read" and not args.yes:

        parser.error(
            "Writing device settings requires --yes. "
            "Read current values first."
        )

    failed = False

    for label, port in ports:

        print(f"\n{'=' * 50}")
        print(f"{label} ({port})")
        print(f"{'=' * 50}")

        try:

            run_port(
                port,
                args.action,
                args.baud,
                args.address,
            )

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
    main()
