"""Compare Modbus polling rates for one or two WT901C485-E sensors.

This script ONLY reads sensor registers; it never changes hardware settings.
It uses imu_acquisition.py, saves raw data and reports the actual rates.

Examples:
  py imu_benchmark.py --rates 10,20,50,100,200 --duration 10
  py imu_benchmark.py --sensor left --rates 50,100,200 --duration 15 --data-only
  py imu_benchmark.py --sensor right --rates 10,20,50,100 --duration 15
  py imu_benchmark.py --sensor both --rates 10,20,50,100 --duration 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace

from imu_acquisition import LEFT_PORT, RIGHT_PORT, MODBUS_ADDRESS, record


def parse_rates(text: str) -> list[float]:
    try:
        rates = [float(s.strip()) for s in text.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Expected comma-separated numeric rates') from exc
    if not rates or any(rate <= 0 or rate > 500 for rate in rates):
        raise argparse.ArgumentTypeError('Each rate must be > 0 and <= 500 Hz')
    return rates


def main():
    parser = argparse.ArgumentParser(description='Measure achieved Modbus response rates per IMU')
    parser.add_argument('--sensor', choices=('left', 'right', 'both'), default='both',
                        help='Test only left, only right, or both IMUs (default: both)')
    parser.add_argument('--rates', type=parse_rates, default=[10, 20, 50, 100, 200])
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds to record at each rate')
    parser.add_argument('--left', default=LEFT_PORT)
    parser.add_argument('--right', default=RIGHT_PORT)
    parser.add_argument('--address', type=lambda text: int(text, 0), default=MODBUS_ADDRESS)
    parser.add_argument('--baud', default='auto')
    parser.add_argument('--left-baud')
    parser.add_argument('--right-baud')
    parser.add_argument('--timing', choices=('precise','legacy','max'), default='precise',
                        help='precise=high-resolution polling; legacy=old Windows wait; max=unthrottled')
    parser.add_argument('--data-only', action='store_true', help='Use 12 sensor data registers rather than 16')
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent / 'recordings' / 'benchmarks')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('duration must be positive')
    run_dir = args.outdir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for rate in args.rates:
        print(f'\n--- Target {rate:g} polls/sec per sensor ---', flush=True)
        record_args = SimpleNamespace(
            probe=False, sensor=args.sensor, duration=args.duration, poll_hz=rate,
            baud=args.baud, left_baud=args.left_baud, right_baud=args.right_baud,
            left=args.left, right=args.right, address=args.address,
            data_only=args.data_only, outdir=run_dir,
            queue_capacity=20000, max_queue_drops=20, keep_in_memory=False,
            timing=args.timing,
        )
        path = record(record_args)
        summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        reports.append({'target_poll_hz': rate, 'result_folder': str(path), 'sensors': summary['sensors']})
    result_path = run_dir / 'benchmark_summary.json'
    result_path.write_text(json.dumps({
        'sensor_selection': args.sensor,
        'timing_mode': args.timing,
        'note': 'Achieved Modbus responses/s are not proof of distinct internal IMU updates/s.',
        'trials': reports,
    }, indent=2), encoding='utf-8')
    print('\nPolling-rate comparison (saved responses/s; errors; queue drops):')
    for trial in reports:
        cells = []
        for side in (('left', 'right') if args.sensor == 'both' else (args.sensor,)):
            item = trial['sensors'][side]
            cells.append(f'{side}: {item["saved_samples_per_s"]:.1f} Hz, errors={item["read_errors"]}, drops={item["queue_drops"]}')
        print(f'  target={trial["target_poll_hz"]:g} Hz | ' + ' | '.join(cells))
    print(f'Full report: {result_path}')


if __name__ == '__main__':
    main()
