#!/usr/bin/env python3
"""Check that this repository is source-only and its paired platform code agrees.

Safe/read-only: no network or sensor access. Run `python scripts/verify_repo.py`.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WIN=ROOT/'windows'
PI=ROOT/'raspberry_pi'
CORE=[
 'fpa_calibration_engine.py', 'fpa_combined_detector.py',
 'fpa_continuous_relative_phase.py', 'fpa_differential.py',
 'fpa_frozen_profile.py', 'fpa_phase13_detector.py',
 'fpa_raw6_filters.py', 'fpa_raw6_fpa.py',
 'fpa_raw6_orientation.py', 'fpa_raw6_stride.py',
 'fpa_reference_line.py', 'fpa_synchronous.py',
 'fpa_timebase.py', 'fpa_unified_pipeline.py',
 'online_unified/fpa_online.py', 'online_unified/processing_phase2.py',
 'online_unified/acquisition_phase2.py', 'online_unified/dashboard.html',
]
REQUIRED=[
 'README.md','docs/IMPLEMENTATION.md','docs/DATA_AND_PRIVACY.md',
 'windows/fpa_filter_studio.py','windows/requirements.txt',
 'raspberry_pi/start_pi.sh','raspberry_pi/setup_pi.sh',
 '.github/workflows/tests.yml',
]
PRIVATE_SUFFIXES={'.mp4','.mov','.avi','.zip','.7z','.tar','.gz','.csv','.tsv','.jpg','.jpeg','.png'}
PRIVATE_DIRS={'recordings','sessions','raw_acquisition','private_data','__pycache__'}

def error(message):
 print('ERROR:',message,file=sys.stderr)

def main():
 failures=0
 for rel in REQUIRED:
  if not (ROOT/rel).is_file():error(f'missing required file: {rel}');failures+=1
 for rel in CORE:
  a=WIN/rel;b=PI/rel
  if not a.is_file() or not b.is_file():
   error(f'core absent from platform folder: {rel}');failures+=1
  elif a.read_bytes()!=b.read_bytes():
   error(f'platform drift: {rel}');failures+=1
 # Do not inspect ignored local data: only validate versioned source when available.
 git=subprocess.run(['git','ls-files','-z'],cwd=ROOT,capture_output=True)
 if git.returncode==0:
  for b in git.stdout.split(b'\0'):
   if not b:continue
   path=Path(b.decode('utf-8','replace'))
   if path.suffix.lower() in PRIVATE_SUFFIXES or any(p in PRIVATE_DIRS for p in path.parts):
    error(f'private/generated file is Git-tracked: {path}');failures+=1
 if failures:
  print(f'REPO CHECK FAILED ({failures} issues)')
  return 1
 print('Repository structure OK; 18 core files match on Windows/Pi; no forbidden tracked data found.')
 return 0

if __name__=='__main__':raise SystemExit(main())
