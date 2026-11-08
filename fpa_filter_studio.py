"""ACIT4035 interactive bilateral IMU filter studio + synchronized video viewer.

Research/diagnostic tool. Does not calculate validated FPA or identify heel strikes.
Raw CSVs are only read.  A user-specified video offset is required for alignment.

Install: py -m pip install numpy pandas scipy matplotlib opencv-python pillow
Example:
  py fpa_filter_studio.py --left recordings\\...\\left_raw.csv --right recordings\\...\\right_raw.csv --video CrossreferenceTest.mp4
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d, binary_closing, binary_opening
from scipy.signal import butter, sosfiltfilt, savgol_filter, iirnotch, filtfilt

G = 9.80665
GYRO = ('gx_dps', 'gy_dps', 'gz_dps')
ACC = ('ax_mps2', 'ay_mps2', 'az_mps2')
CHANNELS = ['Angular speed |gyro|', 'Acceleration magnitude |a|', *GYRO, *ACC]
FILTERS = ['Raw', 'Butterworth low-pass', 'Butterworth high-pass', 'Butterworth band-pass', 'Savitzky-Golay', 'Gaussian', 'Median', 'EMA (causal)', 'Median + Butterworth', 'Notch']


def load_imu(path: str):
    df = pd.read_csv(path)
    for k in (*GYRO, *ACC):
        if k not in df.columns:
            raise ValueError(f'{path}: missing column {k}')
    if 'timestamp_midpoint_monotonic_s' in df.columns:
        t = df['timestamp_midpoint_monotonic_s'].to_numpy(dtype=float, copy=True)
    elif 'time_s' in df.columns:
        t = df['time_s'].to_numpy(dtype=float, copy=True)
    elif 'host_elapsed_s' in df.columns:
        t = df['host_elapsed_s'].to_numpy(dtype=float, copy=True)
    else:
        raise ValueError(f'{path}: no supported time column')
    t -= t[0]
    if len(t) < 40 or not np.all(np.diff(t) > 0):
        raise ValueError('Need >=40 samples and strictly increasing timestamps')
    accel = df[list(ACC)].to_numpy(dtype=float)
    gyro = df[list(GYRO)].to_numpy(dtype=float)
    if not np.isfinite(accel).all() or not np.isfinite(gyro).all():
        raise ValueError('Non-finite sensor values in input')
    fs = float(1 / np.median(np.diff(t)))
    if not 10 < fs < 2000:
        raise ValueError(f'Unexpected sampling frequency {fs:.1f} Hz')
    return dict(t=t, accel=accel, gyro=gyro, fs=fs, path=str(path), rows=len(t))


def odd_window(ms, fs, n, minimum=3):
    w = max(minimum, round(ms * fs / 1000))
    w += (w % 2 == 0)
    return min(w, n - 1 if (n % 2 == 0) else n) | 1


def filter_signal(x, fs, method='Butterworth low-pass', cutoff=15., window_ms=75., order=4, high_cut=35., notch_hz=50.):
    """Filter sensor axes separately. N x 3 arrays supported."""
    x = np.asarray(x, float)
    if method == 'Raw': return x.copy()
    nyquist = fs / 2
    if method in ('Butterworth low-pass', 'Butterworth high-pass', 'Median + Butterworth'):
        sos = butter(int(order), np.clip(cutoff, 0.2, nyquist * .90),
                     btype='highpass' if method == 'Butterworth high-pass' else 'lowpass', fs=fs, output='sos')
        if method == 'Median + Butterworth':
            w = odd_window(window_ms, fs, len(x))
            xx = median_filter(x, size=(w, 1) if x.ndim == 2 else w, mode='nearest')
        else: xx = x
        return sosfiltfilt(sos, xx, axis=0)
    if method == 'Butterworth band-pass':
        lo = min(cutoff, high_cut)
        hi = max(cutoff, high_cut)
        lo = max(0.2, lo)
        hi = min(nyquist * .92, hi)
        if hi <= lo * 1.05: raise ValueError('Bandpass needs separated cutoff frequencies')
        return sosfiltfilt(butter(int(order), [lo, hi], fs=fs, btype='bandpass', output='sos'), x, axis=0)
    if method == 'Savitzky-Golay':
        w = odd_window(window_ms, fs, len(x), 5)
        return savgol_filter(x, w, 3 if w > 3 else 2, axis=0, mode='interp')
    if method == 'Gaussian':
        sigma = max(0.2, window_ms * fs / 1000 / 2.355)  # window is approximate FWHM
        return gaussian_filter1d(x, sigma, axis=0, mode='nearest')
    if method == 'Median':
        w = odd_window(window_ms, fs, len(x))
        return median_filter(x, size=(w, 1) if x.ndim == 2 else w, mode='nearest')
    if method == 'EMA (causal)':
        alpha = 1 - np.exp(-2 * np.pi * max(0.2, cutoff) / fs)
        output = np.empty_like(x)
        output[0] = x[0]
        for i in range(1,len(x)):
            output[i] = alpha*x[i] + (1-alpha)*output[i-1]
        return output
    if method == 'Notch':
        target = min(notch_hz, .9 * nyquist)
        b,a=iirnotch(target, 15.0, fs)
        return filtfilt(b,a,x,axis=0)
    raise ValueError('Unknown filter')


def detect_stance(t, gyro, accel, fs, gyro_threshold=35, accel_tol=1.0,
                  variance_threshold=0.5, min_stance_ms=100, variance_window_ms=110):
    # Inputs are prefiltered 3-vector axes. Not validated ground contacts.
    gnorm=np.linalg.norm(gyro,axis=1)
    anorm=np.linalg.norm(accel,axis=1)
    win=odd_window(variance_window_ms, fs, len(t))
    avar=np.maximum(0., uniform_filter1d(anorm**2,size=win,mode='nearest')
                    -uniform_filter1d(anorm,size=win,mode='nearest')**2)
    mask=(gnorm<gyro_threshold)&(abs(anorm-G)<accel_tol)&(avar<variance_threshold)
    # Join tiny gaps and eliminate tiny candidate intervals.
    gap=max(1,int(round(.025*fs)))
    minimum=max(1,int(round(min_stance_ms/1000*fs)))
    mask=binary_closing(mask,structure=np.ones(gap),border_value=0)
    mask=binary_opening(mask,structure=np.ones(minimum),border_value=0)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    intervals=[(float(t[s]),float(t[e-1])) for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    return mask,intervals,avar


def video_info(path):
    import cv2
    cap=cv2.VideoCapture(path)
    try:
        if not cap.isOpened(): raise ValueError('Cannot open video with OpenCV')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <=0 or n <=0: raise ValueError('Invalid video FPS or frame count')
        return dict(fps=fps,frames=n,duration=n/fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally: cap.release()


class Studio:
    def __init__(self, root, left=None, right=None, video=None):
        self.root=root
        root.title('ACIT4035 | Bilateral FPA Filter Studio — experimental')
        root.geometry('1510x900')
        root.minsize(1090,720)
        self.data={}
        self.video_cap=None
        self.video_path=None
        self.video_duration=0.
        self.video_fps=30.
        self.video_photo=None
        self.video_frame=-1
        self.video_time=0.
        self.playing=False
        self.play_started=0.
        self.play_anchor=0.
        self.plot_job=None
        self.video_job=None
        self.current_t=0.
        self.offset=tk.DoubleVar(value=0.)
        self.current_channel=tk.StringVar(value=CHANNELS[0])
        self.method=tk.StringVar(value='Butterworth low-pass')
        self.cutoff=tk.DoubleVar(value=15.)
        self.high_cut=tk.DoubleVar(value=35.)
        self.window=tk.DoubleVar(value=75.)
        self.order=tk.IntVar(value=4)
        self.notch=tk.DoubleVar(value=50.)
        self.bias=tk.BooleanVar(value=True)
        self.stance=tk.BooleanVar(value=True)
        self.gyro_threshold=tk.DoubleVar(value=35.)
        self.accel_tol=tk.DoubleVar(value=1.)
        self.variance=tk.DoubleVar(value=.5)
        self.min_stance=tk.DoubleVar(value=100.)
        self.follow=tk.BooleanVar(value=True)
        self.labels=[]
        self.latest={}
        self._construct()
        if left and right: self.open_csvs(left,right)
        if video: self.open_video(video)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.video_job=self.root.after(65,self.video_tick)

    def _construct(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        outer=ttk.Frame(self.root,padding=7);outer.pack(fill='both',expand=True)
        top=ttk.Frame(outer);top.pack(fill='x')
        ttk.Button(top,text='Open LEFT + RIGHT CSV',command=self.choose_csvs).pack(side='left',padx=2)
        ttk.Button(top,text='Open VIDEO',command=self.choose_video).pack(side='left',padx=2)
        ttk.Button(top,text='Save settings',command=self.save_settings).pack(side='left',padx=2)
        ttk.Button(top,text='Export processed CSVs',command=self.export_processed).pack(side='left',padx=2)
        self.status=ttk.Label(top,text='Choose two raw CSVs and optionally an MP4 video',foreground='#3e647e')
        self.status.pack(side='right',padx=8)
        middle=ttk.PanedWindow(outer,orient='horizontal');middle.pack(fill='both',expand=True,pady=6)
        left=ttk.Frame(middle);right=ttk.Frame(middle,width=450)
        middle.add(left,weight=4);middle.add(right,weight=2)
        self.figure=Figure(figsize=(9,6),dpi=100)
        self.axes=self.figure.subplots(2,1,sharex=True)
        self.figure.subplots_adjust(left=.10,right=.98,top=.94,bottom=.10,hspace=.26)
        self.plotcanvas=FigureCanvasTkAgg(self.figure,master=left)
        self.plotcanvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.plotcanvas,left,pack_toolbar=False).pack(fill='x')
        self.plotcanvas.mpl_connect('button_press_event',self.plot_click)
        self.video_view=tk.Label(right,text='Load MP4 to preview video',bg='#171a21',fg='white',height=22)
        self.video_view.pack(fill='both',expand=True,padx=5,pady=5)
        vt=ttk.Frame(right);vt.pack(fill='x',padx=6)
        ttk.Button(vt,text='◀ frame',command=lambda:self.seek_video(self.video_time-1/max(1,self.video_fps))).pack(side='left')
        self.play_button=ttk.Button(vt,text='Play',command=self.toggle_play);self.play_button.pack(side='left',padx=6)
        ttk.Button(vt,text='frame ▶',command=lambda:self.seek_video(self.video_time+1/max(1,self.video_fps))).pack(side='left')
        self.time_label=ttk.Label(vt,text='video 0.00 s');self.time_label.pack(side='right')
        self.video_slider=tk.Scale(right,from_=0,to=1,resolution=.01,orient='horizontal',showvalue=False,
                                 command=self.scrub_video)
        self.video_slider.pack(fill='x',padx=8)
        os=ttk.LabelFrame(right,text='Video synchronization (video time = IMU time + offset)')
        os.pack(fill='x',padx=5,pady=6)
        ttk.Scale(os,from_=-8,to=8,variable=self.offset,command=lambda _=None:self.on_offset()).pack(fill='x',padx=5)
        self.offset_label=ttk.Label(os,text='Offset +0.00 s — manual alignment required')
        self.offset_label.pack()
        ttk.Checkbutton(os,text='Cursor follows video',variable=self.follow).pack(anchor='w')
        mark=ttk.LabelFrame(right,text='Optional video foot-plant reference labels')
        mark.pack(fill='x',padx=5,pady=5)
        ttk.Button(mark,text='Mark LEFT planted',command=lambda:self.add_label('left')).pack(side='left',padx=4)
        ttk.Button(mark,text='Mark RIGHT planted',command=lambda:self.add_label('right')).pack(side='left',padx=4)
        ttk.Button(mark,text='Export labels',command=self.export_labels).pack(side='left',padx=4)
        self.mark_info=ttk.Label(right,text='Reference marks are manually annotated, not automatic detections.',wraplength=420)
        self.mark_info.pack(fill='x',padx=8)
        controls=ttk.LabelFrame(outer,text='Live signal processing · original CSVs are never modified')
        controls.pack(fill='x',pady=2)
        row1=ttk.Frame(controls);row1.pack(fill='x',pady=3)
        ttk.Label(row1,text='Signal:').pack(side='left',padx=5)
        cbox=ttk.Combobox(row1,textvariable=self.current_channel,values=CHANNELS,state='readonly',width=28)
        cbox.pack(side='left');cbox.bind('<<ComboboxSelected>>',lambda _:self.schedule_plot())
        ttk.Label(row1,text='Filter:').pack(side='left',padx=(15,4))
        mbox=ttk.Combobox(row1,textvariable=self.method,values=FILTERS,state='readonly',width=26)
        mbox.pack(side='left');mbox.bind('<<ComboboxSelected>>',lambda _:self.schedule_plot())
        ttk.Checkbutton(row1,text='Subtract stationary gyro bias',variable=self.bias,command=self.schedule_plot).pack(side='left',padx=14)
        ttk.Checkbutton(row1,text='Show candidate foot-plant intervals',variable=self.stance,command=self.schedule_plot).pack(side='left')
        r2=ttk.Frame(controls);r2.pack(fill='x',pady=3)
        self.scale_field(r2,'Low cutoff (Hz)',self.cutoff,1,45)
        self.scale_field(r2,'High cutoff (Hz)',self.high_cut,5,90)
        self.scale_field(r2,'Window (ms)',self.window,15,250)
        self.scale_field(r2,'Order',self.order,2,6,step=1)
        self.scale_field(r2,'Notch (Hz)',self.notch,10,90)
        r3=ttk.Frame(controls);r3.pack(fill='x',pady=3)
        self.scale_field(r3,'Stationary gyro (°/s)',self.gyro_threshold,5,80)
        self.scale_field(r3,'Gravity tolerance (m/s²)',self.accel_tol,.25,3)
        self.scale_field(r3,'Accel variance',self.variance,.1,2)
        self.scale_field(r3,'Min stationary (ms)',self.min_stance,30,300)
        self.metrics=ttk.Label(outer,text='Load CSV files to compare raw and processed signals.',wraplength=1350)
        self.metrics.pack(anchor='w',fill='x',pady=3)

    def scale_field(self,parent,title,var,lo,hi,step=None):
        box=ttk.Frame(parent);box.pack(side='left',fill='x',expand=True,padx=5)
        value=ttk.Label(box,text=f'{title}: {float(var.get()):.1f}',width=29,anchor='w')
        value.pack(anchor='w')
        def moved(_):
            v=var.get()
            value.config(text=f'{title}: {int(v) if step else v:.1f}' if not step else f'{title}: {int(v)}')
            self.schedule_plot()
        tk.Scale(box,from_=lo,to=hi,resolution=step or (0.05 if hi<=5 else 1),variable=var,
                 orient='horizontal',showvalue=False,command=moved,length=150).pack(fill='x')

    def choose_csvs(self):
        l=filedialog.askopenfilename(title='Select LEFT raw CSV',filetypes=[('CSV','*.csv')]);
        if not l: return
        r=filedialog.askopenfilename(title='Select RIGHT raw CSV',filetypes=[('CSV','*.csv')]);
        if r: self.open_csvs(l,r)

    def open_csvs(self,l,r):
        try:
            self.data={'left':load_imu(l),'right':load_imu(r)}
            self.current_t=0.
            self.status.config(text=f'L {self.data["left"]["rows"]} / R {self.data["right"]["rows"]} rows')
            self.schedule_plot()
        except Exception as e: messagebox.showerror('CSV load failed',str(e))

    def choose_video(self):
        path=filedialog.askopenfilename(title='Open reference walking video',filetypes=[('Videos','*.mp4 *.mov *.avi *.mkv'),('All files','*.*')]);
        if path:self.open_video(path)

    def open_video(self,path):
        try:
            import cv2
            info=video_info(path)
            if self.video_cap:self.video_cap.release()
            self.video_cap=cv2.VideoCapture(path)
            self.video_path=str(path);self.video_duration=info['duration'];self.video_fps=info['fps'];self.video_frame=-1
            self.video_slider.config(to=self.video_duration)
            self.seek_video(0)
            self.status.config(text=f'Video {info["duration"]:.2f}s, {info["fps"]:.2f}fps | CSVs '+ ('loaded' if self.data else 'missing'))
        except Exception as e: messagebox.showerror('Video load failed',str(e))

    def scrub_video(self,value):
        if self.video_cap:
            requested=float(value)
            if abs(requested-self.video_time)>.045: self.seek_video(requested,update_slider=False)

    def seek_video(self,seconds,update_slider=True):
        if not self.video_cap: return
        import cv2
        from PIL import Image, ImageTk
        t=float(np.clip(seconds,0,max(0,self.video_duration-1/self.video_fps)))
        frame=int(round(t*self.video_fps))
        if frame != self.video_frame:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,image=self.video_cap.read()
            if ok:
                self.video_frame=frame
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                img=Image.fromarray(rgb)
                w=max(300,self.video_view.winfo_width()-16);h=max(200,self.video_view.winfo_height()-16)
                img.thumbnail((w,h),Image.Resampling.LANCZOS)
                self.video_photo=ImageTk.PhotoImage(img)
                self.video_view.config(image=self.video_photo,text='')
        self.video_time=t
        self.time_label.config(text=f'Video {t:.2f}/{self.video_duration:.2f}s | IMU {t-self.offset.get():.2f}s')
        if update_slider: self.video_slider.set(t)
        if self.follow.get():
            self.current_t=t-self.offset.get()
            self.draw_cursor()

    def toggle_play(self):
        if not self.video_cap: return
        self.playing=not self.playing
        self.play_button.config(text='Pause' if self.playing else 'Play')
        self.play_started=time.perf_counter();self.play_anchor=self.video_time

    def video_tick(self):
        if self.playing and self.video_cap:
            requested=self.play_anchor+(time.perf_counter()-self.play_started)
            if requested >= self.video_duration-1/self.video_fps:
                self.playing=False;self.play_button.config(text='Play')
            else: self.seek_video(requested)
        self.video_job=self.root.after(60,self.video_tick)

    def on_offset(self):
        self.offset_label.config(text=f'Offset {self.offset.get():+.2f} s — verify using visible event')
        self.current_t=self.video_time-self.offset.get()
        self.draw_cursor()

    def plot_click(self,event):
        if event.inaxes not in self.axes or event.xdata is None:return
        self.current_t=float(event.xdata)
        self.seek_video(self.current_t+self.offset.get())
        self.draw_cursor()

    def schedule_plot(self):
        if self.plot_job is not None:
            try:self.root.after_cancel(self.plot_job)
            except Exception:pass
        self.plot_job=self.root.after(160,self.refresh_plot)

    def processed(self,side):
        d=self.data[side]
        w=d['gyro'].copy()
        if self.bias.get():
            # Only low-angular-velocity points from first 2s; do not silently presume first 2s are stationary.
            first=d['t']<min(2.0,d['t'][-1]/4)
            quiet=first & (np.linalg.norm(w,axis=1)<8)
            if quiet.sum()>=max(15,int(.08*d['fs'])): bias=np.median(w[quiet],axis=0)
            else:bias=np.zeros(3)  # warn instead of guessing bias from movement
            w-=bias
        else:bias=np.zeros(3)
        opts=dict(method=self.method.get(),cutoff=self.cutoff.get(),high_cut=self.high_cut.get(),
                  window_ms=self.window.get(),order=self.order.get(),notch_hz=self.notch.get())
        filtered_g=filter_signal(w,d['fs'],**opts)
        filtered_a=filter_signal(d['accel'],d['fs'],**opts)
        # Stance requires gravity preserved in acceleration. High-pass/band-pass
        # transform away the DC gravity term, so use a fixed 15 Hz low-pass
        # detector path rather than producing misleading empty intervals.
        if self.method.get() in ('Butterworth high-pass', 'Butterworth band-pass'):
            detector_g = filter_signal(w,d['fs'],method='Butterworth low-pass',cutoff=15.)
            detector_a = filter_signal(d['accel'],d['fs'],method='Butterworth low-pass',cutoff=15.)
        else:
            detector_g,detector_a=filtered_g,filtered_a
        mask,iv,avar=detect_stance(d['t'],detector_g,detector_a,d['fs'],
                self.gyro_threshold.get(),self.accel_tol.get(),self.variance.get(),self.min_stance.get())
        return dict(gyro_raw=w,gyro=filtered_g,accel_raw=d['accel'],accel=filtered_a,
                    mask=mask,intervals=iv,bias=bias,avar=avar)

    def get_channel(self,d,p,processed):
        name=self.current_channel.get()
        if name == CHANNELS[0]:return np.linalg.norm(p['gyro'] if processed else p['gyro_raw'],axis=1)
        if name == CHANNELS[1]:return np.linalg.norm(p['accel'] if processed else p['accel_raw'],axis=1)
        if name in GYRO:return (p['gyro'] if processed else p['gyro_raw'])[:,GYRO.index(name)]
        if name in ACC:return (p['accel'] if processed else p['accel_raw'])[:,ACC.index(name)]
        raise ValueError('Unknown signal')

    def refresh_plot(self):
        self.plot_job=None
        if not self.data:return
        try:
            stats=[]
            for ax,side in zip(self.axes,('left','right')):
                d=self.data[side];p=self.processed(side);self.latest[side]=p
                raw=self.get_channel(d,p,False);flt=self.get_channel(d,p,True)
                ax.clear()
                ax.plot(d['t'],raw,color='0.65',lw=.75,label='Raw',alpha=.72)
                ax.plot(d['t'],flt,color='#157da5',lw=1.15,label='Filtered')
                if self.stance.get():
                    for i,(a,b) in enumerate(p['intervals']):ax.axvspan(a,b,color='#3d9a77',alpha=.20,
                                                         label='Stationary candidate' if not i else None)
                ax.grid(alpha=.18)
                ax.set_ylabel(f'{side.upper()}\n'+('°/s' if self.current_channel.get() in GYRO or self.current_channel.get()==CHANNELS[0] else 'm/s²'))
                for mark in self.labels:
                    if mark['side']==side:ax.axvline(mark['imu_time_s'],color='#ce602f',alpha=.65,lw=1,ls='--')
                ax.legend(loc='upper right',fontsize=8,ncol=3)
                stats.append(f'{side}: {len(p["intervals"])} stationary candidates, bias [{", ".join(f"{x:.2f}" for x in p["bias"])}] °/s')
            self.axes[-1].set_xlabel('IMU time since first recorded sample (s) — click to seek video')
            self.axes[-1].set_xlim(0,max(self.data['left']['t'][-1],self.data['right']['t'][-1]))
            self.figure.suptitle(self.current_channel.get()+' | '+self.method.get()+' | candidate stance is NOT validated',fontsize=11)
            suffix=' | Stance uses fixed 15 Hz low-pass for high/band-pass display' if self.method.get() in ('Butterworth high-pass','Butterworth band-pass') else ''
            self.metrics.config(text=' | '.join(stats)+' | video offset '+f'{self.offset.get():+.2f}s'+suffix)
            self.draw_cursor(redraw=False)
            self.plotcanvas.draw_idle()
        except Exception as e:
            self.metrics.config(text='Filter error: '+str(e))

    def draw_cursor(self,redraw=True):
        if not hasattr(self,'axes'):return
        for ax in self.axes:
            if getattr(ax,'_studio_cursor',None) is not None:
                try:ax._studio_cursor.remove()
                except Exception:pass
            ax._studio_cursor=ax.axvline(self.current_t,color='#e27b28',lw=1.6,alpha=.85)
        if redraw:self.plotcanvas.draw_idle()

    def add_label(self,side):
        if not self.video_cap or not self.data:return
        t=self.video_time-self.offset.get()
        if not 0<=t<=max(d['t'][-1] for d in self.data.values()):
            messagebox.showwarning('Not aligned','Current video time is outside the IMU recording. Adjust offset first.')
            return
        self.labels.append(dict(side=side,imu_time_s=round(t,4),video_time_s=round(self.video_time,4),
                                offset_s=round(self.offset.get(),4),event='visually_stationary_foot_mark'))
        self.mark_info.config(text=f'{len(self.labels)} manual video marks (latest: {side} at video {self.video_time:.2f}s / IMU {t:.2f}s).')
        self.schedule_plot()

    def export_labels(self):
        if not self.labels:
            messagebox.showinfo('No annotations','Mark a left/right planted foot from the video first.');return
        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='foot_plant_video_marks.csv',filetypes=[('CSV','*.csv')]);
        if not path:return
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(self.labels[0]));w.writeheader();w.writerows(self.labels)
        messagebox.showinfo('Saved',path)

    def save_settings(self):
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='fpa_filter_settings.json',filetypes=[('JSON','*.json')]);
        if not path:return
        payload={'channel':self.current_channel.get(),'filter':self.method.get(),'cutoff_hz':self.cutoff.get(),
                 'high_cut_hz':self.high_cut.get(),'window_ms':self.window.get(),'butterworth_order':self.order.get(),
                 'notch_hz':self.notch.get(),'gyro_bias_correction':self.bias.get(),'video_offset_s':self.offset.get(),
                 'stance_gyro_threshold_dps':self.gyro_threshold.get(),'stance_accel_tolerance_mps2':self.accel_tol.get(),
                 'stance_accel_variance_limit':self.variance.get(),'min_stance_ms':self.min_stance.get(),
                 'note':'Experimental parameters. Not validated foot plants or FPA.'}
        Path(path).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        messagebox.showinfo('Saved',path)

    def export_processed(self):
        if not self.data:return
        folder=filedialog.askdirectory(title='Select output folder for processed CSVs')
        if not folder:return
        try:
            for side,d in self.data.items():
                p=self.processed(side)
                out=pd.DataFrame({'imu_time_s':d['t'],'raw_gyro_magnitude_dps':np.linalg.norm(p['gyro_raw'],axis=1),
                     'processed_gyro_magnitude_dps':np.linalg.norm(p['gyro'],axis=1),
                     'raw_acc_magnitude_mps2':np.linalg.norm(p['accel_raw'],axis=1),
                     'processed_acc_magnitude_mps2':np.linalg.norm(p['accel'],axis=1),
                     'stationary_candidate':p['mask'].astype(int)})
                for j,key in enumerate(GYRO):out[key+'_processed']=p['gyro'][:,j]
                for j,key in enumerate(ACC):out[key+'_processed']=p['accel'][:,j]
                out.to_csv(Path(folder)/f'{side}_processed_studio.csv',index=False)
            messagebox.showinfo('Saved',f'Processed CSVs saved in {folder}\nOriginal raw files unchanged.')
        except Exception as e:messagebox.showerror('Export error',str(e))

    def close(self):
        self.playing=False
        for job in (self.video_job,self.plot_job):
            if job:
                try:self.root.after_cancel(job)
                except Exception:pass
        if self.video_cap:self.video_cap.release()
        self.root.destroy()


def run_cli():
    ap=argparse.ArgumentParser(description='Interactive bilateral IMU signal filtering and video comparison')
    ap.add_argument('--left',help='left_raw.csv')
    ap.add_argument('--right',help='right_raw.csv')
    ap.add_argument('--video',help='reference MP4 (optional)')
    ap.add_argument('--self-test',action='store_true',help='Headless filter and data-loading test')
    args=ap.parse_args()
    if args.self_test:
        if not args.left or not args.right:ap.error('--self-test needs --left and --right')
        for side,p in [('left',args.left),('right',args.right)]:
            d=load_imu(p)
            print(side,len(d['t']),round(d['fs'],2),'Hz')
            for method in FILTERS:
                x=filter_signal(d['gyro'],d['fs'],method=method)
                mask,iv,_=detect_stance(d['t'],x,filter_signal(d['accel'],d['fs'],method=method),d['fs'])
                assert x.shape==d['gyro'].shape and np.isfinite(x).all()
                print(' ',method, 'candidates',len(iv))
        if args.video:print('video',video_info(args.video))
        print('SELF-TEST PASS')
        return
    if bool(args.left)!=bool(args.right):ap.error('Provide both --left and --right, or neither to select through dialogs')
    root=tk.Tk()
    Studio(root,args.left,args.right,args.video)
    root.mainloop()

if __name__=='__main__':run_cli()
