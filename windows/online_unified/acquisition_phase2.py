"""ACIT4035 Phase 2: isolated two-port Modbus acquisition (Windows + Pi 5).

Uses the PROVEN imu_acquisition.py read/CRC/pacing routines, but reads ONLY
registers 0x34..0x39 (acceleration + gyro). One process per physical port.
Backpressure / queue drops, timing gaps and serial errors are explicit.
No filtering, no fusion, no sensor configuration writes, no estimated samples.
"""
from __future__ import annotations
from collections import deque
import multiprocessing as mp
from queue import Empty, Full
import statistics
import threading
import time
from typing import Callable

G = 9.80665

def _safe_put(q, value) -> bool:
    try:
        q.put_nowait(value)
        return True
    except Full:
        return False


def serial_process(side: str, port: str, baud: int, hz: float, timing: str,
                   out_q, stop_event):
    """Spawn-safe process entry; all serial reads stay outside the web process."""
    # Import here so web/API imports and replay require no serial hardware.
    try:
        import serial
        from imu_acquisition import (read_modbus_registers, wait_until,
                                     enable_high_resolution_timer)
    except Exception as ex:
        why=f'Acquisition dependency failure: {type(ex).__name__}: {ex}'
        _safe_put(out_q, ('fatal',side,why))
        _safe_put(out_q, ('closed',side,{'attempts':0,'errors':1,
                  'last_error':why,'queue_drops':0,'scheduler_skipped':0}))
        return
    interval = 1.0 / hz
    stats = {'attempts': 0, 'good': 0, 'errors': 0, 'scheduler_skipped': 0,
             'queue_drops': 0, 'last_error': None, 'port': port,
             'requested_hz': hz, 'timing': timing}
    batch = []
    last_report = time.perf_counter()
    last_flush = last_report
    next_poll = time.perf_counter()

    def flush():
        nonlocal batch,last_flush
        if not batch:
            return
        if not _safe_put(out_q, ('samples', side, batch)):
            stats['queue_drops'] += len(batch)
        batch = []
        last_flush=time.perf_counter()

    try:
        if timing == 'precise':
            enable_high_resolution_timer()
        with serial.Serial(port, baudrate=baud, bytesize=8, parity='N',
                           stopbits=1, timeout=0.12, write_timeout=0.12) as ser:
            while not stop_event.is_set():
                if timing != 'max' and not wait_until(next_poll, stop_event, 'precise'):
                    break
                stats['attempts'] += 1
                try:
                    t0,t1,_,v = read_modbus_registers(ser,0x50,0x34,6)
                    sample = ((t0+t1)/2,
                              tuple(float(x)*16*G/32768 for x in v[:3]) +
                              tuple(float(x)*2000/32768 for x in v[3:]))
                    batch.append(sample)
                    stats['good'] += 1
                    if len(batch) >= 12:
                        flush()
                except (TimeoutError, ValueError, serial.SerialException, OSError) as ex:
                    stats['errors'] += 1
                    stats['last_error'] = f'{type(ex).__name__}: {ex}'
                    if stats['errors'] <= 3:
                        _safe_put(out_q, ('error', side, stats['last_error']))
                if timing != 'max':
                    next_poll += interval
                    now = time.perf_counter()
                    if now > next_poll:
                        skipped = int((now-next_poll)//interval) + 1
                        stats['scheduler_skipped'] += skipped
                        next_poll += skipped*interval
                now = time.perf_counter()
                if batch and (now-last_flush >= .12):
                    flush()
                if now-last_report >= 1.0:
                    flush()
                    _safe_put(out_q, ('health', side, stats.copy()))
                    last_report = now
    except Exception as ex:
        stats['errors'] += 1
        stats['last_error'] = f'Port failure: {type(ex).__name__}: {ex}'
        _safe_put(out_q, ('fatal', side, stats['last_error']))
    finally:
        flush()
        _safe_put(out_q, ('health', side, stats.copy()))
        _safe_put(out_q, ('closed', side, stats.copy()))


class TimingStats:
    """Measured host midpoint times, NOT sensor-native sampling frequency."""
    def __init__(self, side, port, hz):
        self.side = side
        self.port = port
        self.requested_hz = hz
        self.good = 0
        self.errors = 0
        self.last_error = None
        self.queue_drops = 0
        self.scheduler_skipped = 0
        self.attempts = 0
        self.last_tm = None
        self.first_tm = None
        self.max_interval_ms = 0.0
        self.gaps_over_30ms = 0
        self.gaps_over_60ms = 0
        self.nonmonotonic = 0
        self._recent = deque(maxlen=10000)
        self._session = None

    def reset_session(self):
        self._session = {'first': None, 'last': None, 'samples':0,
                         'max_interval_ms':0., 'gaps_over_30ms':0,
                         'gaps_over_60ms':0,'nonmonotonic':0,
                         'intervals': deque(maxlen=50000),
                         'last_tm':None}

    def add(self, tm):
        self.good += 1
        if self.first_tm is None:
            self.first_tm = tm
        if self.last_tm is not None:
            dt = (tm-self.last_tm)*1000
            if dt <= 0:
                self.nonmonotonic += 1
            else:
                self.max_interval_ms = max(self.max_interval_ms, dt)
                self.gaps_over_30ms += dt > 30.
                self.gaps_over_60ms += dt > 60.
                self._recent.append(dt)
        self.last_tm = tm
        s=self._session
        if s is not None:
            if s['first'] is None:s['first']=tm
            if s['last_tm'] is not None:
                dt=(tm-s['last_tm'])*1000
                if dt <= 0:s['nonmonotonic']+=1
                else:
                    s['max_interval_ms']=max(s['max_interval_ms'],dt)
                    s['gaps_over_30ms']+=dt>30
                    s['gaps_over_60ms']+=dt>60
                    s['intervals'].append(dt)
            s['last_tm']=tm
            s['last']=tm
            s['samples']+=1

    @staticmethod
    def _metrics(gaps, first, last, samples):
        vals=list(gaps)
        return dict(saved_samples=samples,
            effective_hz=(samples-1)/(last-first) if samples>1 and last>first else None,
            median_interval_ms=statistics.median(vals) if vals else None,
            p95_interval_ms=sorted(vals)[int((len(vals)-1)*.95)] if vals else None)

    def snapshot(self, session=False):
        if session and self._session is not None:
            s=self._session
            return {**self._metrics(s['intervals'],s['first'],s['last'],s['samples']),
                    'max_interval_ms':round(s['max_interval_ms'],3),
                    'gaps_over_30ms':s['gaps_over_30ms'],
                    'gaps_over_60ms':s['gaps_over_60ms'],
                    'nonmonotonic':s['nonmonotonic'],
                    'timing_quality':'HOST_MIDPOINT_NOT_HARDWARE_TIMESTAMP'}
        return {**self._metrics(self._recent,self.first_tm,self.last_tm,self.good),
                'good':self.good,'errors':self.errors,'last_error':self.last_error,
                'queue_drops':self.queue_drops,'scheduler_skipped':self.scheduler_skipped,
                'poll_attempts':self.attempts,'port':self.port,
                'requested_hz':self.requested_hz,
                'max_interval_ms':round(self.max_interval_ms,3),
                'gaps_over_30ms':self.gaps_over_30ms,
                'gaps_over_60ms':self.gaps_over_60ms,
                'nonmonotonic':self.nonmonotonic,
                'timing_quality':'HOST_MIDPOINT_NOT_HARDWARE_TIMESTAMP'}


class ProcessAcquisition:
    """Collector in server process; isolated serial workers with bounded IPC."""
    def __init__(self, port_map, baud, hz, add_callback: Callable, timing='precise'):
        self.port_map=dict(port_map)
        self.baud,self.hz,self.add_callback,self.timing=baud,hz,add_callback,timing
        self.stats={s:TimingStats(s,p,hz) for s,p in self.port_map.items() if p}
        self._lock=threading.RLock()
        self._ctx=mp.get_context('spawn')
        self._q=None
        self._stop=None
        self._processes=[]
        self._collector=None
        self._done=threading.Event()
        self._closed_sides=set()

    def start(self):
        if not self.stats:return
        self._q=self._ctx.Queue(maxsize=512)
        self._stop=self._ctx.Event()
        self._done.clear()
        self._closed_sides.clear()
        self._processes=[self._ctx.Process(
            target=serial_process,args=(s,p,self.baud,self.hz,self.timing,self._q,self._stop),
            name='IMU-'+s,daemon=True) for s,p in self.port_map.items() if p]
        self._collector=threading.Thread(target=self._collect,name='imu-ipc-collector',daemon=True)
        self._collector.start()
        for p in self._processes:p.start()

    def _collect(self):
        while not self._done.is_set():
            try: kind,side,payload=self._q.get(timeout=.1)
            except Empty:continue
            if kind=='samples':
                for tm,values in payload:
                    with self._lock:
                        self.stats[side].add(tm)
                    # Do not invoke a controller callback while holding our lock:
                    # status/start hold controller lock and may request our stats.
                    self.add_callback(side,tm,values)
            else:
                with self._lock:
                    diag=self.stats[side]
                    if kind in ('health','closed'):
                        if kind=='closed':self._closed_sides.add(side)
                        diag.attempts=payload['attempts']
                        diag.errors=payload['errors']
                        diag.last_error=payload['last_error']
                        diag.queue_drops=payload['queue_drops']
                        diag.scheduler_skipped=payload['scheduler_skipped']
                    elif kind in ('error','fatal'):
                        diag.last_error=payload

    def begin_session(self):
        with self._lock:
            for stat in self.stats.values():stat.reset_session()

    def snapshot(self, session=False):
        with self._lock:
            return {s:stats.snapshot(session=session) for s,stats in self.stats.items()}

    def stop(self):
        if self._stop:self._stop.set()
        for p in self._processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate();p.join(timeout=.5)
        # Each worker sends a 'closed' marker AFTER its final sample batch.
        # Queue.empty() is not reliable across processes, so await markers.
        deadline=time.perf_counter()+1.5
        while time.perf_counter()<deadline:
            with self._lock:
                finished=len(self._closed_sides)>=len(self._processes)
            if finished:break
            time.sleep(.015)
        self._done.set()
        if self._collector:self._collector.join(timeout=1.0)
        if self._q:
            self._q.close();self._q.join_thread()
