#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null; then echo 'Python3 is required; install Raspberry Pi OS 64-bit.'; exit 1; fi
if ! python3 -m venv --help >/dev/null 2>&1; then echo 'Install python3-venv: sudo apt install -y python3-venv'; exit 1; fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r online_unified/requirements-online.txt
.venv/bin/python -m unittest test_pi_quickstart -v
if ! id -nG | tr ' ' '\n' | grep -qx dialout; then
    echo 'Serial permission: run sudo usermod -aG dialout "$USER", log out and SSH in again.'
fi
printf '\nInstall finished. Attach sensors; run: ./start_pi.sh LEFT_PORT RIGHT_PORT\n'
