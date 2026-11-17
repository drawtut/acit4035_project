"""ACIT4035 experimental online acquisition + browser dashboard.

Hardware: two WT901C485-E on independent RS485 ports; vertical register 0x23=1.
Only raw acceleration/gyro registers are read; fused orientation and magnetometer
are never read. Stage 4 uses only the initial-aligned walkway heading at each landing.
"""
from __future__ import annotations
import argparse, csv, json, threading, time, sys, multiprocessing as mp
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from contextlib import asynccontextmanager

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))  # SINGLE canonical Filter Studio engine; no online/engine copy
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from online_unified.acquisition_phase2 import ProcessAcquisition
from online_unified.processing_phase2 import compute_preview

COLUMNS=['timestamp_midpoint_monotonic_s','host_elapsed_s','ax_mps2','ay_mps2','az_mps2','gx_dps','gy_dps','gz_dps']

class Controller:
    def __init__(self,args):
        self.args=args;self.lock=threading.RLock()
        self.acquisition=ProcessAcquisition({'left':args.left,'right':args.right},
            args.baud,args.hz,self.add,timing=getattr(args,'timing','precise'))
        self._pool=None
        self.started=None;self.recording=False;self.session=None
        self.reference_line_mode=True
        self.buffers={s:deque(maxlen=2000) for s in ('left','right')}
        self.files={};self.writers={};self.counts={'left':0,'right':0}
        self.last_results={};self.last_error=None;self.processing=False
        self.live_rows={'left':[],'right':[]};self.sync_result=None;self.crp_result=None;self.live_error=None
        self.live_points={'left':[],'right':[]};self.live_version=0
        self.live_stop=threading.Event();self.live_thread=None
    def boot(self):
        self.acquisition.start()
    def shutdown(self):
        self.live_stop.set()
        if self.live_thread:self.live_thread.join(timeout=3)
        self.acquisition.stop()
        self.stop()
        if self._pool:
            self._pool.shutdown(wait=False,cancel_futures=True)
            self._pool=None
    def add(self,side,tm,xyz):
        with self.lock:
            row={'time_s':tm, **dict(zip(COLUMNS[2:],xyz))}
            self.buffers[side].append(row)
            if self.recording and side in self.writers:
                self.live_points[side].append((tm,*xyz))
                self.writers[side].writerow([tm,tm-self.started,*xyz]);self.counts[side]+=1
    def start(self,label='walking'):
        with self.lock:
            if self.recording or self.processing:raise ValueError('Session already recording or processing')
            clean=''.join(c for c in label if c.isalnum() or c in '-_')[:45] or 'walking'
            folder=HERE/'sessions'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{clean}'
            folder.mkdir(parents=True,exist_ok=False)
            self.started=time.monotonic();self.session=folder
            self.reference_line_mode=True  # Fixed reference-line protocol; no UI opt-in
            (folder/'protocol.json').write_text(json.dumps({
                'reference_line_mode':self.reference_line_mode,
                'initial_left_right_shoe_heading_deg':0 if self.reference_line_mode else None,
                'requires_external_check':'Both longitudinal shoe axes aligned to straight reference; participant progression parallel to reference',
                'interpretation':'walkway-relative heading; not validated anatomical FPA without real progression reference'
            },indent=2),encoding='utf-8')
            self.counts={'left':0,'right':0};self.files={};self.writers={}
            self.live_rows={'left':[],'right':[]};self.sync_result=None;self.crp_result=None;self.live_points={'left':[],'right':[]}
            self.live_version+=1;self.live_error=None;self.last_results={}
            self.acquisition.begin_session()

            for side in ('left','right'):
                f=(folder/f'{side}_raw.csv').open('w',newline='',encoding='utf-8')
                w=csv.writer(f);w.writerow(COLUMNS);self.files[side]=f;self.writers[side]=w
            self.recording=True;self.last_error=None
            if self._pool is None:
                self._pool=ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))
            if self.live_thread is None or not self.live_thread.is_alive():
                self.live_stop.clear()
                self.live_thread=threading.Thread(target=self.live_loop,daemon=True,name='live-stage4')
                self.live_thread.start()
            return str(folder)
    def stop(self):
        with self.lock:
            self.recording=False
            for f in self.files.values():f.flush();f.close()
            self.files={};self.writers={}
            if self.session:
                (self.session/'summary.json').write_text(json.dumps({
                    'counts':self.counts,
                    'acquisition':self.acquisition.snapshot(session=True),
                    'sensors':self.acquisition.snapshot(),
                    'requested_poll_hz':self.args.hz,
                    'timing_mode':getattr(self.args,'timing','precise'),
                    'filtering_applied_to_csv':False,
                    'reference_line_mode':self.reference_line_mode,
                    'magnetometer_or_fused_euler_used':False,
                    'note':'raw six-channel samples; host request-response midpoint timestamps, no invented measurements'},indent=2))
            return str(self.session) if self.session else None
    def status(self):
        with self.lock:
            return dict(recording=self.recording,processing=self.processing,session=str(self.session) if self.session else None,
                samples=self.counts.copy(),sensors=self.acquisition.snapshot(),
                latest={s:list(self.buffers[s])[-1] if self.buffers[s] else None for s in ('left','right')},
                traces={s:list(self.buffers[s])[::max(1,len(self.buffers[s])//300)] for s in ('left','right')},
                last_results=self.last_results,last_error=self.last_error,
                engine='PHASE16_BASE_PLUS_EXPLORATORY_BILATERAL_CRP',
                reference_line_mode=self.reference_line_mode,
                frozen_profile='impact az 11-24 Hz; stance gyro 10 Hz; FPA gyro 10 Hz / accel 5 Hz',
                step_events={s:list(self.live_rows[s]) for s in ('left','right')},
                synchronous_relative=(dict(
                    summary=self.sync_result['summary'],
                    rows=self.sync_result['rows'][-350:],
                    double_support_periods=self.sync_result['double_support_periods'][-30:]
                ) if self.sync_result else None),
                continuous_relative_phase=(dict(
                    summary=self.crp_result['summary'],
                    rows=self.crp_result['rows'][-500:]
                ) if self.crp_result else None),
                live_error=self.live_error)
    def live_loop(self):
        """Non-blocking preview: orientation/step processing runs in a *different process*.

        Snapshot generation copies measurements under a brief lock and submits
        at most one outstanding preview.  The acquisition processes are
        unaffected by zero-phase filtering or API/browser rendering.
        """
        current=None
        preview_version=None
        while not self.live_stop.wait(4.0):
            if current is not None and current.done():
                try:
                    update=current.result()
                    err=None
                except Exception as exc:
                    update=None
                    err=f'{type(exc).__name__}: {exc}'
                with self.lock:
                    if self.recording and preview_version==self.live_version:
                        if update is not None:
                            self.live_rows=update['events']
                            self.sync_result=update['synchronous']
                            self.crp_result=update.get('crp')
                        self.live_error=err
                current=None
            with self.lock:
                if not self.recording:break
                version=self.live_version
                if current is not None:continue
                if min(map(len,self.live_points.values())) < 900:continue
                points={side:list(self.live_points[side]) for side in ('left','right')}
            try:
                current=self._pool.submit(compute_preview,points,str(HERE/'calibration_frozen_v1'),
                                           self.reference_line_mode)
                preview_version=version
            except Exception as exc:
                with self.lock:self.live_error=f'Preview submission: {type(exc).__name__}: {exc}'
                break

    def analyze(self):
        with self.lock:
            if self.recording:raise ValueError('Stop recording before running Stage 4')
            if self.processing:raise ValueError('Already analyzing')
            if not self.session:raise ValueError('No recorded session')
            self.processing=True;self.last_error=None;folder=self.session
        try:
            # Make broken/missing IMU cables actionable instead of reaching
            # a NumPy IndexError inside the offline processing pipeline.
            with self.lock:
                counts=self.counts.copy()
            low={side:n for side,n in counts.items() if n < 300}
            if low:
                raise ValueError('Not enough sensor data for Stage 4: '+
                    ', '.join(f'{side}={count} samples' for side,count in low.items())+
                    '. Check both USB-RS485 adapters and record 5 seconds standing still before walking.')
            from fpa_unified_pipeline import (process_csv_pair, export_results,
                                              report_summary, events_for_dashboard)
            results,stage=process_csv_pair(folder/'left_raw.csv',folder/'right_raw.csv',
                                           HERE/'calibration_frozen_v1',
                                           reference_line_mode=self.reference_line_mode)
            out=folder/'stage4'
            manifest=export_results(results,stage,out)
            report=report_summary(stage,manifest)
            with self.lock:
                self.last_results={'folder':str(out),'feet':report,
                    'validation':'PROVISIONAL walkway-relative FPA; A impact+stance, B full stance without impact, C soft-stance review. Requires aligned start and straight progression',
                    'pipeline':'Phase16 frozen FPA and synchronized relative heading + exploratory offline Hilbert CRP from both shoe-pitch streams; no displacement integration',
                    'reference_line_mode':self.reference_line_mode,
                    'timebase':{side:results[side].get('timebase') for side in ('left','right')},
                    'bilateral_window':results['left'].get('bilateral_window'),
                    'bilateral_differential':manifest.get('bilateral_differential',{}),
                    'synchronous_relative_heading':manifest.get('synchronous_relative_heading',{}),
                    'continuous_relative_phase':manifest.get('continuous_relative_phase',{})}
                self.sync_result=results['left']['synchronous_relative']
                self.crp_result=results['left'].get('bilateral_crp')
                self.live_rows=events_for_dashboard(stage)
        except Exception as e:
            with self.lock:self.last_error=f'{type(e).__name__}: {e}'
        finally:
            with self.lock:self.processing=False

parser=argparse.ArgumentParser()
parser.add_argument('--left',default='/dev/ttyUSB0')
parser.add_argument('--right',default='/dev/ttyUSB1')
parser.add_argument('--baud',type=int,default=230400)
parser.add_argument('--hz',type=float,default=150.)
parser.add_argument('--timing',choices=('precise','max'),default='precise',
    help='precise uses the proven cooperative scheduler; max polls as quickly as possible')
parser.add_argument('--host',default='0.0.0.0')
parser.add_argument('--port',type=int,default=8000)
parser.add_argument('--demo',action='store_true',help='Dashboard without serial readers')
# Parse at entry, not import: supports API TestClient and unit tests.
controller=None
@asynccontextmanager
async def lifespan(app):
    if controller is not None:controller.boot()
    yield
    if controller is not None:controller.shutdown()
app=FastAPI(title='ACIT4035 Bilateral FPA (research)',lifespan=lifespan)
class StartRequest(BaseModel):
    label:str='walking'
@app.get('/',response_class=HTMLResponse)
def root():return (HERE/'dashboard.html').read_text(encoding='utf-8')
@app.get('/api/status')
def status():return controller.status() if controller else {'error':'Controller not initialized'}
@app.post('/api/start')
def start(req:StartRequest):
    try:return {'folder':controller.start(req.label)}
    except ValueError as e:raise HTTPException(409,str(e))
@app.post('/api/stop')
def stop():return {'folder':controller.stop()}
@app.get('/api/download_latest')
def download_latest():
    """Export the current participant's raw data, reports, and calibration metadata.

    No path argument is accepted: arbitrary server file downloads are impossible.
    Do not expose this unauthenticated research endpoint to untrusted Wi-Fi.
    """
    if controller is None:
        raise HTTPException(503,'Controller not initialized')
    with controller.lock:
        if controller.recording or controller.processing:
            raise HTTPException(409,'Stop recording and finish analysis before downloading')
        folder=controller.session
        if folder is None or not folder.is_dir():
            raise HTTPException(404,'No recorded session')
        dest=folder.parent/(folder.name+'.zip')
        with ZipFile(dest,'w',compression=ZIP_DEFLATED,compresslevel=4) as z:
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,arcname=folder.name+'/'+str(p.relative_to(folder)))
            for name in ('left_foot_calibration.json','right_foot_calibration.json'):
                cal=HERE/'calibration_frozen_v1'/name
                if cal.is_file():z.write(cal,arcname=folder.name+'/calibration/'+name)
    return FileResponse(str(dest),media_type='application/zip',filename=dest.name)

@app.post('/api/analyze')
def analyze():
    with controller.lock:
        if controller.recording or controller.processing or not controller.session:raise HTTPException(409,'Stop recording first')
    threading.Thread(target=controller.analyze,daemon=True).start()
    return {'started':True,'note':'Frozen raw6 filtering; finalized reference-line results are available after processing'}

if __name__=='__main__':
    import uvicorn
    a=parser.parse_args()
    if not 1 <= a.hz <= 500:parser.error('--hz must be between 1 and 500')
    if a.demo:a.left=a.right=None
    controller=Controller(a)
    uvicorn.run(app,host=a.host,port=a.port)
