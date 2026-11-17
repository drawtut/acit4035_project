#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup_pi.sh first.'; exit 1; fi
exec .venv/bin/python online_unified/fpa_online.py --demo --host 0.0.0.0 --port "${FPA_PORT:-8000}"
