#!/usr/bin/env bash
# Optional. One-time install: run ./enable_autostart.sh /dev/serial/by-id/LEFT /dev/serial/by-id/RIGHT
set -euo pipefail
cd "$(dirname "$0")"
if (( $# != 2 )); then echo 'Usage: ./enable_autostart.sh LEFT_SERIAL_PATH RIGHT_SERIAL_PATH'; exit 2; fi
for dev in "$1" "$2"; do
  if [[ "$dev" != /dev/* || "$dev" == *$'
'* || "$dev" == *' '* ]]; then echo 'Serial ports must be simple absolute /dev/... paths.'; exit 2; fi
  if [[ ! -e "$dev" ]]; then echo "Not connected: $dev"; exit 2; fi
done
if [[ "$(readlink -f "$1")" == "$(readlink -f "$2")" ]]; then echo 'Choose separate serial ports'; exit 2; fi
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup_pi.sh first.'; exit 1; fi
port=${FPA_PORT:-8000}
if ! [[ "$port" =~ ^[0-9]+$ ]]; then echo 'FPA_PORT must be numeric'; exit 2; fi
service=/etc/systemd/system/fpa-imu.service
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
cat > "$tmp" <<EOF
[Unit]
Description=ACIT4035 shoe-IMU bilateral FPA research dashboard
Wants=network-online.target
After=network-online.target
[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/.venv/bin/python $(pwd)/online_unified/fpa_online.py --left $1 --right $2 --baud 230400 --hz 150 --timing precise --host 0.0.0.0 --port $port
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
sudo install -m 0644 "$tmp" "$service"
sudo systemctl daemon-reload
sudo systemctl enable --now fpa-imu.service
echo 'Autostart enabled. View status: systemctl status fpa-imu.service'
echo "Phone dashboard: http://fpa-pi.local:$port (same private Wi-Fi)"
echo 'Disable later: sudo systemctl disable --now fpa-imu.service'
