
import time
import serial

# Current verified sensor configuration
PORT = "COM5"
BAUD = 230400
ADDRESS = 0x50

REGISTERS = {
    "Output rate": 0x03,
    "Baud setting": 0x04,
    "Filter bandwidth": 0x1F,
    "Gyroscope range": 0x20,
    "Accelerometer range": 0x21,
    "Installation direction": 0x23,
    "6-axis / 9-axis mode": 0x24,
    "Modbus response delay": 0x74,
}


def modbus_crc(data):
    crc = 0xFFFF

    for byte in data:
        crc ^= byte

        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1

    return crc.to_bytes(2, "little")


def read_register(ser, register):
    request = bytes([
        ADDRESS,
        0x03,
        (register >> 8) & 0xFF,
        register & 0xFF,
        0x00,
        0x01
    ])

    request += modbus_crc(request)

    ser.reset_input_buffer()
    ser.write(request)
    ser.flush()

    response = ser.read(7)

    if len(response) != 7:
        raise TimeoutError(
            f"Expected 7 bytes, received {len(response)}"
        )

    if modbus_crc(response[:-2]) != response[-2:]:
        raise ValueError("Invalid Modbus CRC")

    if response[0] != ADDRESS or response[1] != 0x03:
        raise ValueError(
            f"Unexpected Modbus response: {response.hex(' ')}"
        )

    if response[2] != 2:
        raise ValueError("Unexpected register byte count")

    return int.from_bytes(response[3:5], "big")


def main():
    print(f"Connecting to {PORT} at {BAUD:,} baud")

    with serial.Serial(
        port=PORT,
        baudrate=BAUD,
        timeout=0.5,
        write_timeout=0.5
    ) as ser:

        time.sleep(0.1)

        for name, register in REGISTERS.items():
            value = None

            for attempt in range(3):
                try:
                    value = read_register(ser, register)
                    break

                except Exception as error:
                    if attempt == 2:
                        print(f"{name:25s} ERROR: {error}")
                    time.sleep(0.05)

            if value is not None:
                print(
                    f"{name:25s} "
                    f"register=0x{register:02X} "
                    f"value={value} "
                    f"(0x{value:04X})"
                )


if __name__ == "__main__":
    main()
