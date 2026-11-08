"""WT901C485-E Modbus RTU configuration with readback and recovery.

Usage:
  py sensor_config.py read --sensor left
  py sensor_config.py raw-max --sensor left --yes
  py sensor_config.py set-baud --sensor left --target-baud 230400 --yes
  py sensor_config.py set-baud --sensor left --target-baud 921600 --yes
  py sensor_config.py set-delay --sensor left --delay-us 2000 --current-baud 230400 --yes
  py sensor_config.py set-delay --sensor left --delay-us 3000 --current-baud 230400 --yes
  py sensor_config.py vertical --sensor left --yes
  py sensor_config.py apply-profile --sensor left --yes
  py sensor_config.py check-profile --sensor left

IMPORTANT: raw-max selects the device's WIDEST documented bandwidth (256 Hz),
NOT a proven bypass of its firmware filters. Register outputs are original
sensor register counts, NOT guaranteed raw ADC samples. Auto-output remains
disabled for reliable request/response Modbus polling. NO hardware writes are
performed unless --yes is given. Close other programs using the COM port.
"""
from __future__ import annotations
import argparse
import struct
import time

import serial
from imu_reader import LEFT_PORT, RIGHT_PORT, MODBUS_ADDRESS, crc16

REG_SAVE = 0x00
REG_RRATE = 0x03
REG_BAUD = 0x04
REG_BANDWIDTH = 0x1F
REG_GYRO_RANGE = 0x20
REG_ACC_RANGE = 0x21
REG_ORIENT = 0x23
REG_AXIS6 = 0x24
REG_UNLOCK = 0x69
REG_MODDELAY = 0x74  # RS485 Modbus reply delay in microseconds

# ACIT4035 validated single-IMU acquisition profile, 2026-11-04.
# Reapply with: py sensor_config.py apply-profile --sensor left --yes
# NOTE: 200 Hz HOST POLLING is configured in imu_acquisition.py, NOT here.
# RRATE=13 disables unsolicited streaming while retaining on-demand Modbus reads.
# Values of sensor counts remain firmware-processed; 256 Hz is the widest
# documented filter bandwidth, not a bypass of internal filtering.
PROFILE_BAUD = 230400
PROFILE_REGISTERS = {
    REG_RRATE: 13,       # Automatic output disabled (Modbus polling mode)
    REG_BANDWIDTH: 0,    # Widest documented internal bandwidth, 256 Hz
    REG_GYRO_RANGE: 3,   # +/- 2000 deg/s
    REG_ACC_RANGE: 3,    # +/- 16 g
    REG_ORIENT: 1,       # Vertical sensor installation
    REG_AXIS6: 0,        # Keep existing 9-axis fusion, pending FPA algorithm
    REG_MODDELAY: 1000,  # 1 ms; verified working with USB-RS485 adapter
}

BAUD_CODES = {4800: 1, 9600: 2, 19200: 3, 38400: 4, 57600: 5,
              115200: 6, 230400: 7, 460800: 8, 921600: 9}
PROBE_BAUDS = (230400, 115200, 460800, 921600, 9600, 57600, 38400, 19200, 4800)
REGISTER_NAMES = {REG_RRATE: 'Auto output (RRATE)', REG_BAUD: 'Baud code',
                  REG_BANDWIDTH: 'Internal bandwidth', REG_GYRO_RANGE: 'Gyro range',
                  REG_ACC_RANGE: 'Accel range', REG_ORIENT: 'Installation',
                  REG_AXIS6: 'Fusion mode', REG_MODDELAY: 'Modbus reply delay'}
BANDWIDTHS = {0: '256 Hz (widest available, NOT filter bypass)', 1: '184 Hz',
              2: '94 Hz', 3: '44 Hz', 4: '21 Hz', 5: '10 Hz', 6: '5 Hz'}
RRATES = {6:'10 Hz auto',7:'20 Hz auto',8:'50 Hz auto',9:'100 Hz auto',
          10:'125/200 Hz (firmware dependent)',11:'200 Hz auto (SDK)',
          12:'single output',13:'automatic output disabled'}


def exchange(ser: serial.Serial, addr: int, fn: int, reg: int, val: int,
             *, allow_missing_echo=False) -> bytes | None:
    request = struct.pack('>BBHH', addr, fn, reg, val)
    request += struct.pack('<H', crc16(request))
    ser.reset_input_buffer()
    ser.write(request)
    ser.flush()
    header = ser.read(3)
    if len(header) != 3:
        if allow_missing_echo:
            return None  # baud changes may take effect before a write reply
        raise TimeoutError(f'No Modbus reply at {ser.baudrate} baud')
    if header[0] != addr:
        raise ValueError(f'Unexpected device address {header[0]:02x}')
    if header[1] == (fn | 0x80):
        rest = ser.read(2)
        raise ValueError(f'Modbus exception code={header[2]} packet={(header+rest).hex()}')
    if header[1] != fn:
        raise ValueError(f'Unexpected function {header[1]}')
    if fn == 3:
        if header[2] != 2*val:
            raise ValueError('Unexpected read length')
        tail = ser.read(2*val + 2)
    elif fn == 6:
        tail = ser.read(5)
    else:
        raise ValueError('Unsupported Modbus function')
    response = header + tail
    if len(tail) != (2*val+2 if fn == 3 else 5):
        if allow_missing_echo:
            return None
        raise TimeoutError('Truncated Modbus reply')
    if struct.unpack('<H', response[-2:])[0] != crc16(response[:-2]):
        raise ValueError('Modbus reply CRC mismatch')
    if fn == 6 and response[:6] != request[:6]:
        raise ValueError('Write echo mismatch')
    return response


def read_reg(ser, addr, register):
    r = exchange(ser, addr, 3, register, 1)
    return struct.unpack('>H', r[3:5])[0]


def write_reg(ser, addr, register, value, *, allow_missing_echo=False):
    return exchange(ser, addr, 6, register, value,
                    allow_missing_echo=allow_missing_echo)


def open_serial(port, baud):
    return serial.Serial(port, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                         timeout=0.30, write_timeout=0.5)


def probe(port, addr, known_baud=None):
    """Probe measurement register first, like imu_acquisition.py.

    Some units intermittently reject configuration-register reads during startup.
    Retry a CRC-validated sensor measurement before accessing configuration.
    No register writes are performed.
    """
    rates = [known_baud] if known_baud else list(PROBE_BAUDS)
    last_errors = []
    for baud in rates:
        try:
            with open_serial(port, baud) as ser:
                for attempt in range(3):
                    try:
                        exchange(ser, addr, 3, 0x34, 1)
                        return baud
                    except (serial.SerialException, OSError, ValueError, TimeoutError) as exc:
                        last_errors.append(f'{baud:,} baud attempt {attempt+1}: {type(exc).__name__}: {exc}')
                        time.sleep(0.06)
        except (serial.SerialException, OSError, ValueError, TimeoutError) as exc:
            last_errors.append(f'{baud:,} baud opening port: {type(exc).__name__}: {exc}')
    last = '; '.join(last_errors[-3:])
    raise RuntimeError(f'No valid Modbus reply on {port}. Last errors: {last}')


def read_reg_with_retry(ser, addr, reg, attempts=3):
    last = None
    for _ in range(attempts):
        try:
            return read_reg(ser, addr, reg)
        except (serial.SerialException, OSError, ValueError, TimeoutError) as exc:
            last = exc
            time.sleep(0.05)
    raise RuntimeError(f'Register 0x{reg:02X} read failed after {attempts} tries: {type(last).__name__}: {last}')


def print_state(port, addr, baud):
    with open_serial(port, baud) as ser:
        readings = {reg:read_reg_with_retry(ser, addr, reg) for reg in REGISTER_NAMES}
    print(f'\n{port} at {baud:,} baud:')
    for reg, label in REGISTER_NAMES.items():
        value = readings[reg]
        detail = ''
        if reg == REG_BANDWIDTH: detail = BANDWIDTHS.get(value, 'unknown')
        if reg == REG_RRATE: detail = RRATES.get(value, 'unknown')
        if reg == REG_MODDELAY: detail = f'{value} us ({value/1000:g} ms)'
        if reg == REG_BAUD: detail = str(next((b for b,c in BAUD_CODES.items() if c==value), 'unknown')) + ' baud'
        if reg == REG_ORIENT: detail = {0:'horizontal',1:'vertical'}.get(value,'unknown')
        if reg == REG_AXIS6: detail = {0:'9-axis',1:'6-axis'}.get(value,'unknown')
        if reg == REG_GYRO_RANGE: detail = {0:'250',1:'500',2:'1000',3:'2000'}.get(value,'unknown')+' deg/s'
        if reg == REG_ACC_RANGE: detail = {0:'2',1:'4',2:'8',3:'16'}.get(value,'unknown')+' g'
        print(f'  0x{reg:02X} {label:25s}: 0x{value:04X}  {detail}')
    return readings


def modify_regular(port, addr, baud, updates):
    """Write non-baud registers, save, and independently read back after reopen."""
    with open_serial(port, baud) as ser:
        previous = {reg:read_reg_with_retry(ser,addr,reg) for reg in updates}
        changed = {reg:value for reg,value in updates.items() if previous[reg]!=value}
        if not changed:
            print('  Already configured; no writes necessary')
            return
        write_reg(ser,addr,REG_UNLOCK,0xB588)
        time.sleep(.10)
        for reg,value in changed.items():
            write_reg(ser,addr,reg,value)
            time.sleep(.10)
            verified=read_reg_with_retry(ser,addr,reg)
            if verified!=value:
                raise RuntimeError(f'Register 0x{reg:02X} expected {value}, got {verified}')
        write_reg(ser,addr,REG_SAVE,0)
        time.sleep(.25)
    time.sleep(.15)
    with open_serial(port, baud) as ser:
        for reg,value in changed.items():
            observed=read_reg_with_retry(ser,addr,reg)
            if observed!=value:
                raise RuntimeError(f'Post-reopen mismatch 0x{reg:02X}: {observed} != {value}')
    print('  Saved, verified after reopening port. Power-cycle later to confirm persistence.')


def change_baud(port,addr,old_baud,new_baud):
    if new_baud == old_baud:
        print(f'  Already communicating at {new_baud:,} baud')
        return
    print(f'  Switching {old_baud:,} -> {new_baud:,} baud')
    with open_serial(port,old_baud) as ser:
        write_reg(ser,addr,REG_UNLOCK,0xB588)
        time.sleep(.10)
        # Some firmware switches serial bit rate before returning the write echo.
        try:
            write_reg(ser,addr,REG_BAUD,BAUD_CODES[new_baud], allow_missing_echo=True)
        except (TimeoutError, ValueError) as exc:
            # Some firmware changes baud before completing the write echo;
            # treat a garbled acknowledgement as inconclusive, not success.
            # Reprobe and read back the register below before saving.
            print(f'  Baud write acknowledgement inconclusive ({exc}); verifying by reprobe')
    time.sleep(.30)
    # The change may take effect immediately OR only after a save/reboot.
    try:
        live=probe(port,addr)
    except RuntimeError as exc:
        raise RuntimeError('Baud change sent, but sensor could not be found. '
                           'Power-cycle, then run "read" to probe again.') from exc
    print(f'  Now responding at {live:,} baud')
    if live not in (new_baud,old_baud):
        raise RuntimeError('Unexpected baud detected; stopping before save')
    with open_serial(port,live) as ser:
        value=read_reg_with_retry(ser,addr,REG_BAUD)
        if value != BAUD_CODES[new_baud]:
            raise RuntimeError(f'Baud register did not change (code={value}); stopping')
        write_reg(ser,addr,REG_UNLOCK,0xB588)
        time.sleep(.10)
        write_reg(ser,addr,REG_SAVE,0)
    time.sleep(.25)
    final=probe(port,addr)
    if final != new_baud:
        print(f'  NOTE: communicating at {final:,}; power-cycle then probe again')
    else:
        print(f'  SUCCESS: {new_baud:,} baud verified, save requested')


def main():
    p=argparse.ArgumentParser(description='WT901C485-E register config and maximum-bandwidth setup')
    p.add_argument('action',choices=['read','check-profile','apply-profile','raw-max','set-baud','set-bandwidth','set-delay','vertical','horizontal','six-axis','nine-axis'])
    p.add_argument('--sensor',choices=['left','right','both'],default='left')
    p.add_argument('--left',default=LEFT_PORT)
    p.add_argument('--right',default=RIGHT_PORT)
    p.add_argument('--address',type=lambda x:int(x,0),default=MODBUS_ADDRESS)
    p.add_argument('--current-baud',type=int,choices=list(BAUD_CODES),help='Optional known current baud; otherwise auto-probe')
    p.add_argument('--target-baud',type=int,choices=list(BAUD_CODES),help='New baud, only for set-baud')
    p.add_argument('--bandwidth-code',type=int,choices=range(7),help='For set-bandwidth; 0=256 Hz, 4=21 Hz (previous setting)')
    p.add_argument('--delay-us',type=int,help='Modbus response delay for set-delay (1000 to 10000 microseconds); default 3000')
    p.add_argument('--yes',action='store_true',help='Required for configuration writes')
    a=p.parse_args()
    if a.action not in ('read', 'check-profile') and not a.yes:
        p.error('Writes require --yes')
    if a.action=='set-baud' and a.target_baud is None:
        p.error('set-baud requires --target-baud')
    if a.action=='set-bandwidth' and a.bandwidth_code is None:
        p.error('set-bandwidth requires --bandwidth-code 0..6')
    if a.action=='set-delay' and (a.delay_us is None or not 1000 <= a.delay_us <= 10000):
        p.error('set-delay requires --delay-us 1000..10000 (test 2000, then 1000; restore 3000)')
    if not 1<=a.address<=247:
        p.error('Modbus address must be 1..247')
    ports=[('left',a.left),('right',a.right)] if a.sensor=='both' else [(a.sensor,a.left if a.sensor=='left' else a.right)]
    failed=False
    for name,port in ports:
        print(f'\n{name.upper()} ({port})')
        try:
            current=probe(port,a.address,a.current_baud)
            before=print_state(port,a.address,current)
            if a.action=='read': continue
            if a.action == 'check-profile':
                mismatches = [(reg, before[reg], desired) for reg, desired in PROFILE_REGISTERS.items()
                              if before[reg] != desired]
                if current != PROFILE_BAUD:
                    print(f'  MISMATCH: baud {current:,}, expected {PROFILE_BAUD:,}')
                    failed = True
                for reg, actual, desired in mismatches:
                    print(f'  MISMATCH: 0x{reg:02X} {REGISTER_NAMES[reg]} actual={actual}, expected={desired}')
                    failed = True
                if current == PROFILE_BAUD and not mismatches:
                    print('  ACIT4035 PROFILE MATCHES (no settings changed)')
                continue
            if a.action == 'apply-profile':
                print('  Applying hard-coded ACIT4035 200-Hz acquisition SENSOR profile:')
                print('  230400 baud, 256-Hz bandwidth, RRATE=13, 1-ms Modbus delay,')
                print('  +/-16 g, +/-2000 deg/s, vertical installation, 9-axis fusion.')
                modify_regular(port, a.address, current, PROFILE_REGISTERS)
                if current != PROFILE_BAUD:
                    change_baud(port, a.address, current, PROFILE_BAUD)
                verify_baud = probe(port, a.address, PROFILE_BAUD)
                actual = print_state(port, a.address, verify_baud)
                mismatch = [(reg, actual[reg], desired) for reg, desired in PROFILE_REGISTERS.items()
                            if actual[reg] != desired]
                if mismatch:
                    raise RuntimeError(f'ACIT4035 profile verification failed: {mismatch}')
                print('  ACIT4035 PROFILE VERIFIED. Record at 200 Hz using imu_acquisition.py.')
                continue
            if a.action=='raw-max':
                # RRATE_NONE prevents unsolicited output from colliding with Modbus replies.
                # Does NOT prove that the internal sensor ADC/filtering is bypassed.
                modify_regular(port,a.address,current,{REG_BANDWIDTH:0, REG_RRATE:13})
            elif a.action=='set-bandwidth':
                modify_regular(port,a.address,current,{REG_BANDWIDTH:a.bandwidth_code})
            elif a.action=='set-delay':
                previous=before[REG_MODDELAY]
                print(f'  Modbus reply delay: {previous} -> {a.delay_us} us')
                modify_regular(port,a.address,current,{REG_MODDELAY:a.delay_us})
            elif a.action=='set-baud':
                change_baud(port,a.address,current,a.target_baud)
            else:
                reg,val=(REG_ORIENT,1) if a.action=='vertical' else (
                        (REG_ORIENT,0) if a.action=='horizontal' else (
                        (REG_AXIS6,1) if a.action=='six-axis' else (REG_AXIS6,0)))
                modify_regular(port,a.address,current,{reg:val})
            print_state(port,a.address,probe(port,a.address))
        except (serial.SerialException,OSError,TimeoutError,ValueError,RuntimeError) as exc:
            failed=True
            print(f'ERROR: {exc}')
    if failed: raise SystemExit(1)

if __name__=='__main__': main()
