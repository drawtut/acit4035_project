"""Read-only WT901C485-E Modbus RTU timing diagnostic (single USB-RS485 IMU).

Run: py imu_latency_diagnostic.py --port COM5 --baud 230400 --samples 200
Requires: pip install pyserial

Does NOT change the sensor. Reports host-side write/flush, first-byte wait,
remaining-response read, and complete transaction durations. The timings include
USB/OS/Python overhead; they do not by themselves isolate sensor processing.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import statistics
import struct
import time

import serial


def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def packet(address, start, count):
    req = struct.pack('>BBHH', address, 3, start, count)
    return req + struct.pack('<H', crc16(req))


def read_exact(ser, n):
    result = bytearray()
    while len(result) < n:
        chunk = ser.read(n - len(result))
        if not chunk:
            raise TimeoutError(f'Expected {n} bytes; received {len(result)}')
        result.extend(chunk)
    return bytes(result)


def timed_read(ser, address, count):
    req = packet(address, 0x34, count)
    expected = 3 + 2 * count + 2
    ser.reset_input_buffer()
    t0 = time.perf_counter_ns()
    ser.write(req)
    t_write = time.perf_counter_ns()
    ser.flush()
    t_flush = time.perf_counter_ns()
    first = read_exact(ser, 1)
    t_first = time.perf_counter_ns()
    rest = read_exact(ser, expected - 1)
    t_end = time.perf_counter_ns()
    response = first + rest
    if response[0] != address:
        raise ValueError(f'Unexpected address 0x{response[0]:02x}')
    if response[1] != 3 or response[2] != 2 * count:
        raise ValueError(f'Unexpected response header {response[:3].hex(" ")}')
    if crc16(response[:-2]) != int.from_bytes(response[-2:], 'little'):
        raise ValueError('CRC mismatch')
    raw = response[3:-2]
    return {
        'utc': datetime.now(timezone.utc).isoformat(),
        'count': count,
        'write_ms': (t_write-t0)/1e6,
        'flush_ms': (t_flush-t_write)/1e6,
        'first_byte_wait_ms': (t_first-t_flush)/1e6,
        'remainder_read_ms': (t_end-t_first)/1e6,
        'total_ms': (t_end-t0)/1e6,
        'data_hex': raw.hex(),
    }


def summarize(label, rows, errors, elapsed):
    print(f'\n{label}: {len(rows)} valid responses, {errors} errors, '
          f'{len(rows)/elapsed:.1f} responses/s ({elapsed:.2f}s elapsed)')
    if not rows:
        return
    for field in ('write_ms', 'flush_ms', 'first_byte_wait_ms',
                  'remainder_read_ms', 'total_ms'):
        vals = sorted(r[field] for r in rows)
        p95 = vals[min(len(vals)-1, int(.95 * len(vals)))]
        print(f'  {field:23s} median={statistics.median(vals):7.3f} ms '
              f'p95={p95:7.3f} ms')
    repeats = sum(a['data_hex'] == b['data_hex']
                  for a,b in zip(rows,rows[1:]))
    print(f'  identical neighboring payloads: {repeats}/{max(0,len(rows)-1)} '
          '(not a definitive sample-freshness measure)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='COM5', help='Windows COM port')
    ap.add_argument('--baud', type=int, default=230400)
    ap.add_argument('--samples', type=int, default=200, help='reads per mode')
    ap.add_argument('--timeout', type=float, default=.25)
    ap.add_argument('--address', type=lambda s:int(s,0), default=0x50)
    ap.add_argument('--output', default='imu_latency_results.csv')
    args = ap.parse_args()
    if args.samples < 10 or args.timeout <= 0 or not 1 <= args.address <= 247:
        ap.error('samples >= 10, timeout > 0, address 1..247 are required')
    out = Path(args.output)
    if not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    print(f'Opening {args.port} at {args.baud:,} baud; READ ONLY')
    try:
        with serial.Serial(args.port, args.baud, timeout=args.timeout,
                           write_timeout=args.timeout, bytesize=8, parity='N',
                           stopbits=1) as ser:
            time.sleep(.15)
            for count in (1, 12):
                print(f'\nReading {count} register(s) from 0x34, '
                      f'{args.samples} attempts. Move IMU gently for 12-register test.')
                successes, errors = [], 0
                start = time.perf_counter()
                for i in range(args.samples):
                    try:
                        row = timed_read(ser, args.address, count)
                        row['attempt'] = i+1
                        successes.append(row)
                        all_rows.append(row)
                    except (TimeoutError, ValueError, serial.SerialException) as exc:
                        errors += 1
                        if errors <= 4:
                            print(f'  Attempt {i+1}: {type(exc).__name__}: {exc}')
                elapsed = time.perf_counter() - start
                summarize(f'{count}-register reads', successes, errors, elapsed)
    except (serial.SerialException, OSError) as exc:
        print(f'Unable to use serial port: {exc}')
        return 1
    if all_rows:
        fields = ['utc','attempt','count','write_ms','flush_ms',
                  'first_byte_wait_ms','remainder_read_ms','total_ms','data_hex']
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'\nSaved full timing data: {out.resolve()}')
    print('No sensor settings were changed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
