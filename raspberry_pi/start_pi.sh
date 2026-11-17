#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup_pi.sh first.'; exit 1; fi
shopt -s nullglob
ports=(/dev/serial/by-id/* /dev/ttyUSB* /dev/ttyACM*)
if (( ${#ports[@]} == 0 )); then
 echo 'No serial ports found. Connect both USB-RS485 adapters.'
 exit 2
fi
printf 'Available serial device paths:\n'
for p in "${ports[@]}"; do printf '  %s -> %s\n' "$p" "$(readlink -f "$p")"; done
left=${1:-${LEFT_PORT:-}}
right=${2:-${RIGHT_PORT:-}}
if [[ -z "$left" || -z "$right" ]]; then
 echo 'Pass the LEFT and RIGHT sensor device paths in that order.'
 echo 'Example: ./start_pi.sh /dev/ttyUSB0 /dev/ttyUSB1'
 echo 'Prefer /dev/serial/by-id/... if your adapters provide distinct stable IDs.'
 exit 2
fi
if [[ ! -e "$left" || ! -e "$right" ]]; then echo 'One of the specified ports is not present.'; exit 2; fi
if [[ $(readlink -f "$left") == $(readlink -f "$right") ]]; then echo 'Left and right cannot share one serial port.'; exit 2; fi
if [[ ! -r "$left" || ! -w "$left" || ! -r "$right" || ! -w "$right" ]]; then
  echo 'Serial device permission denied. Run sudo usermod -aG dialout "$USER"; logout/login.'
  exit 3
fi
port=${FPA_PORT:-8000}
local_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Pi dashboard: http://${local_ip:-fpa-pi.local}:${port} (phone on same Wi-Fi)"
exec .venv/bin/python online_unified/fpa_online.py --left "$left" --right "$right" --baud 230400 --hz 150 --timing precise --host 0.0.0.0 --port "$port"
