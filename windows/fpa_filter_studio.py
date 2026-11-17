"""ACIT4035 interactive bilateral IMU filter studio + synchronized video viewer.



Research/diagnostic tool. Estimates IMU-only relative sensor orientation, NOT validated FPA.

Raw CSVs are only read.  A user-specified video offset is required for alignment.



Install: py -m pip install numpy pandas scipy matplotlib opencv-python pillow

Example:

  py fpa_filter_studio.py --left PATH_TO_LEFT.csv --right PATH_TO_RIGHT.csv --video CrossreferenceTest.mp4

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
import fpa_processing as shared
from fpa_raw6_filters import filter_raw_six_axis, ACC_COLUMNS, GYRO_COLUMNS
from fpa_raw6_orientation import estimate_six_axis_orientation
from fpa_raw6_stride import process_side as process_offline_stage123, export_stage123, plot_stage123
from fpa_raw6_fpa import stage4_fpa, plot_stage4, export_stage4
from fpa_calibration_engine import build_calibration,save_pair,load_pair
from fpa_segmentation_v21 import detect as detect_swings
from fpa_phase13_detector import detect as detect_multievidence
from fpa_differential import pair_landings, summarize_pairs
from fpa_combined_detector import fpa_for_event, export_rows as export_combined_rows

from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d, binary_closing, binary_opening

from scipy.signal import butter, sosfiltfilt, savgol_filter, iirnotch, filtfilt, find_peaks



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

    shared_host_t=df['timestamp_midpoint_monotonic_s'].to_numpy(dtype=float, copy=True) if 'timestamp_midpoint_monotonic_s' in df.columns else None
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

    return dict(t=t, shared_host_t=shared_host_t, accel=accel, gyro=gyro, fs=fs, path=str(path), rows=len(t), dataframe=df)





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






def impact_envelope_and_peaks(t, filtered_acceleration, fs, rms_window_ms=50.,
                              min_prominence=1.5, min_gap_s=0.35):
    """Offline impact *candidates*, not verified heel-strike timestamps.

    Compute centered RMS energy of the selected band-passed acceleration axis.
    A minimum prominence and distance suppress oscillations within an impact burst.
    """
    values = np.asarray(filtered_acceleration, dtype=float).reshape(-1)
    if len(values) != len(t):
        raise ValueError('Impact signal and timestamps must be the same length')
    win = odd_window(rms_window_ms, fs, len(values))
    envelope = np.sqrt(np.maximum(0., uniform_filter1d(values * values,
                                                        size=win, mode='nearest')))
    min_distance = max(1, int(round(min_gap_s * fs)))
    indices, properties = find_peaks(envelope,
                                     prominence=max(0.01, float(min_prominence)),
                                     distance=min_distance)
    return envelope, indices, properties


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

        self.current_channel=tk.StringVar(value='az_mps2')

        self.method=tk.StringVar(value='Butterworth band-pass')

        self.cutoff=tk.DoubleVar(value=5.)

        self.high_cut=tk.DoubleVar(value=25.)

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
        self.fpa_rows={}
        self.combined_rows={}
        self.selected_step=0
        self.aligned_start=tk.BooleanVar(value=False)
        self.fpa_side=tk.StringVar(value="left")
        self.show_rms=tk.BooleanVar(value=True)
        self.show_impact_peaks=tk.BooleanVar(value=True)
        self.show_combined=tk.BooleanVar(value=True)
        self.use_gyro_rms=tk.BooleanVar(value=True)
        self.gyro_rms_threshold=tk.DoubleVar(value=40.0)
        self.gyro_rms_min_duration=tk.DoubleVar(value=0.45)
        self.active_channel='az_mps2'
        self.impact_source='az_mps2'
        self.channel_profiles={}
        self._switching_channel=False
        self._slider_fields=[]
        self.rms_window=tk.DoubleVar(value=50.)
        self.peak_prominence=tk.DoubleVar(value=1.5)
        self.peak_spacing=tk.DoubleVar(value=0.35)

        # An independent raw-six-axis FPA filter laboratory.  These variables
        # do not feed the previously validated footstep detector.
        self.fpa_gyro_hz=tk.DoubleVar(value=10.)
        self.fpa_accel_hz=tk.DoubleVar(value=5.)
        self.fpa_filter_order=tk.IntVar(value=4)
        self.fpa_bias_start=tk.DoubleVar(value=1.)
        self.fpa_bias_end=tk.DoubleVar(value=4.)
        self.fpa_correct_bias=tk.BooleanVar(value=True)
        self.fpa_filter_gyro=tk.BooleanVar(value=True)
        self.fpa_filter_accel=tk.BooleanVar(value=True)
        self.fpa_lab_window=None
        self.fpa_lab_figure=None
        self.fpa_lab_canvas=None
        self.fpa_lab_axes=None
        self.fpa_zoom_start=tk.DoubleVar(value=0.)
        self.fpa_zoom_end=tk.DoubleVar(value=28.)
        # Independent raw-six-axis orientation VIEW; never fed to step detector.
        self.fpa_forward_axis=tk.StringVar(value='+X')
        self.fpa_tilt_tau=tk.DoubleVar(value=0.75)
        self.fpa_static_gyro=tk.DoubleVar(value=12.)
        self.fpa_gravity_tol=tk.DoubleVar(value=0.9)
        self.fpa_orientation_window=None
        self.fpa_orientation_figure=None
        self.fpa_orientation_axes=None
        self.fpa_orientation_canvas=None
        # Offline stride steps 1-3 are a separate analysis branch.
        # Hypothesized mount axes are NOT physically verified shoe-toe directions.
        self.fpa_mount_left=tk.StringVar(value='+Y')
        self.fpa_mount_right=tk.StringVar(value='+Y')
        self.fpa_stage123_window=None
        self.fpa_stage123_figure=None
        self.fpa_stage123_canvas=None
        # Offline Stage 4 never filters derived orientation/heading/FPA.
        # Explicit physical mount-axis confirmation; default MUST remain false.
        self.fpa_verify_mount_left=tk.BooleanVar(value=False)
        self.fpa_verify_mount_right=tk.BooleanVar(value=False)
        self.fpa_calibration_folder=None
        self.fpa_confirm_straight=tk.BooleanVar(value=False)
        self.fpa_confirm_mount_both=tk.BooleanVar(value=False)
        self.fpa_calibration_info=tk.StringVar(value='No independent calibration loaded; Stage 4 requires separate JSON files.')
        self.fpa_stage4_window=None
        self.fpa_stage4_canvas=None
        self.fpa_stage4_figure=None
        self.fpa_stage4_differential_text=None
        self.fpa_stage4_crp_text=None

        self.shared_config=shared.load_config()
        self._initialize_signal_profiles()
        self._apply_pipeline_config_to_controls()
        self._construct()
        self._restore_profile(self.active_channel)

        if left and right: self.open_csvs(left,right)

        if video: self.open_video(video)

        self.root.protocol('WM_DELETE_WINDOW',self.close)

        self.video_job=self.root.after(65,self.video_tick)



    def _construct(self):

        from matplotlib.figure import Figure

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # Scroll the entire application vertically.  Keeping the graphs and controls
        # in one scrolling page prevents the bottom sliders from disappearing on
        # smaller screens or when Windows display scaling is above 100%.
        page=ttk.Frame(self.root)
        page.pack(fill='both', expand=True)
        self.page_canvas=tk.Canvas(page,highlightthickness=0,borderwidth=0)
        self.page_scroll=ttk.Scrollbar(page,orient='vertical',command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=self.page_scroll.set)
        self.page_scroll.pack(side='right',fill='y')
        self.page_canvas.pack(side='left',fill='both',expand=True)
        outer=ttk.Frame(self.page_canvas,padding=7)
        self.page_content_id=self.page_canvas.create_window((0,0),window=outer,anchor='nw')
        outer.bind('<Configure>',lambda event:self.page_canvas.configure(
            scrollregion=self.page_canvas.bbox('all')))
        self.page_canvas.bind('<Configure>',lambda event:self.page_canvas.itemconfigure(
            self.page_content_id,width=event.width))
        # Mousewheel scrolling works when hovering over application controls.
        # Leave the Matplotlib plot wheel alone so its own navigation remains usable.
        self.root.bind_all('<MouseWheel>',self._page_mousewheel,add='+')
        self.root.bind_all('<Button-4>',lambda event:self._page_scroll_linux(event,-1),add='+')
        self.root.bind_all('<Button-5>',lambda event:self._page_scroll_linux(event,1),add='+')

        top=ttk.Frame(outer);top.pack(fill='x')

        ttk.Button(top,text='Open LEFT + RIGHT CSV',command=self.choose_csvs).pack(side='left',padx=2)

        ttk.Button(top,text='Open VIDEO',command=self.choose_video).pack(side='left',padx=2)

        ttk.Button(top,text='Save settings',command=self.save_settings).pack(side='left',padx=2)
        ttk.Button(top,text='Load settings',command=self.load_settings).pack(side='left',padx=2)
        ttk.Button(top,text='73 tested profile',command=lambda:self.apply_recovered_profile('73')).pack(side='left',padx=2)
        ttk.Button(top,text='69 recovery profile',command=lambda:self.apply_recovered_profile('69')).pack(side='left',padx=2)


        ttk.Button(top,text='Export processed CSVs',command=self.export_processed).pack(side='left',padx=2)
        ttk.Button(top,text='Export impact events',command=self.export_impact_events).pack(side='left',padx=2)
        ttk.Button(top,text='Export combined footsteps',command=self.export_combined).pack(side='left',padx=2)
        ttk.Button(top,text='Save PIPELINE config',command=self.save_pipeline_config).pack(side='left',padx=2)


        self.status=ttk.Label(top,text='Choose two raw CSVs and optionally an MP4 video',foreground='#3e647e')

        self.status.pack(side='right',padx=8)

        # A defined graph/video height leaves room for all controls in scrollable content.
        middle=ttk.PanedWindow(outer,orient='horizontal',height=620)
        middle.pack(fill='x',pady=6)

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

        cbox.pack(side='left');cbox.bind('<<ComboboxSelected>>',self.switch_channel)

        ttk.Label(row1,text='Filter:').pack(side='left',padx=(15,4))

        mbox=ttk.Combobox(row1,textvariable=self.method,values=FILTERS,state='readonly',width=26)

        mbox.pack(side='left');mbox.bind('<<ComboboxSelected>>',lambda _:self.schedule_plot())

        ttk.Checkbutton(row1,text='Subtract stationary gyro bias',variable=self.bias,command=self.schedule_plot).pack(side='left',padx=14)

        ttk.Checkbutton(row1,text='Show candidate foot-plant intervals',variable=self.stance,command=self.schedule_plot).pack(side='left')
        ttk.Checkbutton(row1,text='Show combined candidates (green diamonds)',
                        variable=self.show_combined,command=self.schedule_plot).pack(side='left',padx=8)

        roles=ttk.Frame(controls);roles.pack(fill='x',pady=3)
        self.channel_role_info=ttk.Label(roles,text='',foreground='#2a5b73',wraplength=900)
        self.channel_role_info.pack(side='left',padx=8)
        self.choose_impact_button=ttk.Button(roles,text='Use this acceleration signal for IMPACT detection',
                                             command=self.set_impact_source)
        self.choose_impact_button.pack(side='right',padx=5)
        r2=ttk.Frame(controls);r2.pack(fill='x',pady=3)

        self.scale_field(r2,'Low cutoff (Hz)',self.cutoff,1,45)

        self.scale_field(r2,'High cutoff (Hz)',self.high_cut,5,90)

        self.scale_field(r2,'Window (ms)',self.window,15,250)

        self.scale_field(r2,'Order',self.order,2,6,step=1)

        self.scale_field(r2,'Notch (Hz)',self.notch,10,90)

        impact_row=ttk.Frame(controls);impact_row.pack(fill='x',pady=3)
        ttk.Checkbutton(impact_row,text='Show RMS envelope (green)',
                        variable=self.show_rms,command=self.schedule_plot).pack(side='left',padx=5)
        ttk.Checkbutton(impact_row,text='Show impact peaks (orange)',
                        variable=self.show_impact_peaks,command=self.schedule_plot).pack(side='left',padx=5)
        self.scale_field(impact_row,'RMS window (ms)',self.rms_window,15,150)
        self.peak_scale,self.peak_label=self.scale_field(impact_row,'RMS peak prominence',self.peak_prominence,0.25,200)
        self.scale_field(impact_row,'Min peak spacing (s)',self.peak_spacing,0.2,1.5)

        r3=ttk.Frame(controls);r3.pack(fill='x',pady=3)

        self.scale_field(r3,'Stationary gyro (°/s)',self.gyro_threshold,5,80)

        self.scale_field(r3,'Gravity tolerance (m/s²)',self.accel_tol,.25,3)

        self.scale_field(r3,'Accel variance',self.variance,.1,2)

        self.scale_field(r3,'Min stationary (ms)',self.min_stance,30,300)
        energy_controls=ttk.Frame(controls);energy_controls.pack(fill='x',pady=3)
        ttk.Checkbutton(energy_controls,text='Use GYRO RMS for swing evidence (same rule for all steps)',
                        variable=self.use_gyro_rms,command=self.schedule_plot).pack(side='left',padx=5)
        self.scale_field(energy_controls,'Min swing RMS (°/s)',self.gyro_rms_threshold,10,120)
        self.scale_field(energy_controls,'Min swing duration (s)',self.gyro_rms_min_duration,.20,.70)
        ttk.Label(energy_controls,text='Experimental: RMS motion + impact + stable stance',
                  foreground='#64748b').pack(side='left',padx=8)
        fpa_box=ttk.LabelFrame(outer,text='IMU-only step review | raw-six-axis relative orientation available below')
        fpa_box.pack(fill='x',padx=5,pady=8)
        fpa_controls=ttk.Frame(fpa_box);fpa_controls.pack(fill='x',padx=6,pady=5)
        ttk.Label(fpa_controls,text='Sensor-fused yaw FPA disabled. New 3D ORIENTATION analysis in raw-six-axis lab below.').pack(side='left',padx=4)
        chooser=ttk.Frame(fpa_box);chooser.pack(fill='x',padx=6,pady=4)
        ttk.Label(chooser,text='Selected foot:').pack(side='left')
        ttk.Combobox(chooser,textvariable=self.fpa_side,values=('left','right'),state='readonly',width=8).pack(side='left',padx=5)
        ttk.Button(chooser,text='Previous step',command=lambda:self.select_step(-1)).pack(side='left',padx=3)
        ttk.Button(chooser,text='Next step',command=lambda:self.select_step(1)).pack(side='left',padx=3)
        ttk.Button(chooser,text='Show selected step',command=self.describe_step).pack(side='left',padx=3)
        self.fpa_text=ttk.Label(fpa_box,text='Load IMU data to inspect detected steps. 3D ORIENTATION reconstructs relative sensor direction, not anatomical FPA.',
            wraplength=1320,justify='left')
        self.fpa_text.pack(fill='x',padx=8,pady=5)


        # A separate research view: its six-axis filters NEVER modify the
        # footstep detector's saved impact/swing/stationary settings.
        lab=ttk.LabelFrame(outer,text='NEW: RAW 6-AXIS FPA FILTER LAB | separate from footstep detection')
        lab.pack(fill='x',padx=5,pady=8)
        ttk.Label(lab,wraplength=1310,justify='left',text=(
            'Goal: preserve physical foot rotations (gyro XYZ), obtain a useful gravity direction '
            '(accel XYZ) and inspect raw versus filtered traces for BOTH shoes. '
            'NO yaw/pitch/roll, NO magnetometer, NO video timestamps. '
            'Orientation reconstruction is now available as a SEPARATE diagnostic plot. '
            'Estimated SENSOR-axis heading is not anatomical foot angle or stride-relative FPA.'
        )).pack(fill='x',padx=8,pady=4)
        labfilters=ttk.Frame(lab);labfilters.pack(fill='x',padx=5,pady=4)
        ttk.Label(labfilters,text='Gyro low-pass (Hz)').pack(side='left',padx=2)
        ttk.Spinbox(labfilters,from_=1,to=40,increment=1,textvariable=self.fpa_gyro_hz,width=6).pack(side='left',padx=2)
        ttk.Label(labfilters,text='Accel low-pass (Hz)').pack(side='left',padx=(15,2))
        ttk.Spinbox(labfilters,from_=1,to=40,increment=1,textvariable=self.fpa_accel_hz,width=6).pack(side='left',padx=2)
        ttk.Label(labfilters,text='Order').pack(side='left',padx=(15,2))
        ttk.Combobox(labfilters,state='readonly',values=(2,3,4,5,6),textvariable=self.fpa_filter_order,width=4).pack(side='left',padx=2)
        ttk.Checkbutton(labfilters,text='Filter gyro XYZ',variable=self.fpa_filter_gyro).pack(side='left',padx=10)
        ttk.Checkbutton(labfilters,text='Filter accel XYZ',variable=self.fpa_filter_accel).pack(side='left',padx=6)
        labbias=ttk.Frame(lab);labbias.pack(fill='x',padx=5,pady=4)
        ttk.Checkbutton(labbias,text='Subtract initial stationary gyro bias',variable=self.fpa_correct_bias).pack(side='left',padx=2)
        ttk.Label(labbias,text='Bias window IMU seconds:').pack(side='left',padx=(16,2))
        ttk.Spinbox(labbias,from_=0,to=120,increment=.5,textvariable=self.fpa_bias_start,width=7).pack(side='left',padx=2)
        ttk.Label(labbias,text='to').pack(side='left',padx=3)
        ttk.Spinbox(labbias,from_=0,to=120,increment=.5,textvariable=self.fpa_bias_end,width=7).pack(side='left',padx=2)
        ttk.Label(labbias,text='Graph range (IMU seconds)').pack(side='left',padx=(15,2))
        ttk.Spinbox(labbias,from_=0,to=600,increment=1,textvariable=self.fpa_zoom_start,width=7).pack(side='left',padx=2)
        ttk.Label(labbias,text='to').pack(side='left',padx=3)
        ttk.Spinbox(labbias,from_=0,to=600,increment=1,textvariable=self.fpa_zoom_end,width=7).pack(side='left',padx=2)
        labbuttons=ttk.Frame(lab);labbuttons.pack(fill='x',padx=8,pady=7)
        ttk.Button(labbuttons,text='PLOT all 12 raw + filtered signals',command=self.show_raw6_lab).pack(side='left',padx=4)
        ttk.Button(labbuttons,text='Export FPA-lab filtered CSVs',command=self.export_raw6_lab).pack(side='left',padx=4)
        orient_controls=ttk.Frame(lab);orient_controls.pack(fill='x',padx=8,pady=4)
        ttk.Label(orient_controls,text='3D ORIENTATION — sensor axis for diagnostic heading').pack(side='left',padx=3)
        ttk.Combobox(orient_controls,state='readonly',textvariable=self.fpa_forward_axis,
                     values=('+X','-X','+Y','-Y'),width=4).pack(side='left',padx=3)
        ttk.Label(orient_controls,text='Gravity-correction tau (s)').pack(side='left',padx=(10,2))
        ttk.Spinbox(orient_controls,from_=0.05,to=15.,increment=.25,
                    textvariable=self.fpa_tilt_tau,width=6).pack(side='left',padx=2)
        ttk.Label(orient_controls,text='Quiet gyro < (deg/s)').pack(side='left',padx=(10,2))
        ttk.Spinbox(orient_controls,from_=1,to=50,increment=1,
                    textvariable=self.fpa_static_gyro,width=5).pack(side='left',padx=2)
        ttk.Label(orient_controls,text='|a|-g tolerance (m/s²)').pack(side='left',padx=(10,2))
        ttk.Spinbox(orient_controls,from_=.1,to=3.,increment=.1,
                    textvariable=self.fpa_gravity_tol,width=5).pack(side='left',padx=2)
        orient_buttons=ttk.Frame(lab);orient_buttons.pack(fill='x',padx=8,pady=5)
        ttk.Button(orient_buttons,text='PLOT 3D ORIENTATION — both feet',
                   command=self.show_raw6_orientation).pack(side='left',padx=4)
        ttk.Button(orient_buttons,text='Export orientation + diagnostic CSVs',
                   command=self.export_raw6_orientation).pack(side='left',padx=4)
        ttk.Label(orient_buttons,text='Relative SENSOR axis only — not FPA; no video/Euler/magnetometer.',
                  foreground='#6b7280').pack(side='left',padx=10)
        stride_frame=ttk.Frame(lab);stride_frame.pack(fill='x',padx=8,pady=7)
        ttk.Label(stride_frame,text='OFFLINE STEPS 1–3: provisional mount-forward axis').pack(side='left',padx=3)
        ttk.Label(stride_frame,text='LEFT').pack(side='left',padx=3)
        ttk.Combobox(stride_frame,textvariable=self.fpa_mount_left,
                     values=('+X','-X','+Y','-Y'),state='readonly',width=4).pack(side='left',padx=3)
        ttk.Label(stride_frame,text='RIGHT').pack(side='left',padx=3)
        ttk.Combobox(stride_frame,textvariable=self.fpa_mount_right,
                     values=('+X','-X','+Y','-Y'),state='readonly',width=4).pack(side='left',padx=3)
        ttk.Button(stride_frame,text='PLOT STEP 1–3: contacts + orientation',
                   command=self.show_offline_stage123).pack(side='left',padx=7)
        ttk.Button(stride_frame,text='EXPORT STEP 1–3 diagnostic CSVs',
                   command=self.export_offline_stage123).pack(side='left',padx=6)
        calrow=ttk.Frame(lab);calrow.pack(fill='x',padx=8,pady=5)
        ttk.Label(calrow,text='SEPARATE FUNCTIONAL CALIBRATION:').pack(side='left',padx=3)
        ttk.Checkbutton(calrow,text='Current loaded trial was straight forward walking (no turns)',
                        variable=self.fpa_confirm_straight).pack(side='left',padx=3)
        ttk.Checkbutton(calrow,text='Both printed +Y axes faced toes',
                        variable=self.fpa_confirm_mount_both).pack(side='left',padx=3)
        calbuttons=ttk.Frame(lab);calbuttons.pack(fill='x',padx=8,pady=2)
        ttk.Button(calbuttons,text='CREATE calibration JSONs from loaded raw CSVs',
                   command=self.create_independent_calibration).pack(side='left',padx=4)
        ttk.Button(calbuttons,text='LOAD separate calibration folder',
                   command=self.load_independent_calibration).pack(side='left',padx=4)
        ttk.Label(calbuttons,textvariable=self.fpa_calibration_info,wraplength=560).pack(side='left',padx=8)
        stage4row=ttk.Frame(lab);stage4row.pack(fill='x',padx=8,pady=5)
        ttk.Label(stage4row,text='OFFLINE STEP 4 — landing shoe direction vs INITIAL taped 0° line:').pack(side='left',padx=3)
        ttk.Button(stage4row,text='PLOT signed FPA DIAGNOSTICS',
                   command=self.show_offline_stage4).pack(side='left',padx=5)
        ttk.Button(stage4row,text='EXPORT Stage 1–4 CSVs',
                   command=self.export_offline_stage4).pack(side='left',padx=5)
        ttk.Label(lab,text='Reference-line FPA is ALWAYS active. Experimental protocol: 0° taped initial alignment, 5 s stationary, straight walkway.',
                  foreground='#265d6b',wraplength=1250).pack(anchor='w',padx=9,pady=4)
        verifyrow=ttk.Frame(lab);verifyrow.pack(fill='x',padx=8,pady=2)
        ttk.Label(verifyrow,text='ONLY check after physically verifying sensor axis toward TOES:').pack(side='left',padx=3)
        ttk.Checkbutton(verifyrow,text='LEFT mount confirmed',variable=self.fpa_verify_mount_left).pack(side='left',padx=6)
        ttk.Checkbutton(verifyrow,text='RIGHT mount confirmed',variable=self.fpa_verify_mount_right).pack(side='left',padx=6)
        ttk.Label(lab,text='Stage 4 filters ONLY ax,ay,az,gx,gy,gz upstream; never yaw, quaternion or FPA. '
                        'Opposite-facing axes are REJECTED, not silently flipped. '
                        'Candidates without physically verified mounts are diagnostic only. '
                        'Separate calibration must be CREATED/LOADED; no independently validated anatomical FPA.',
                  foreground='#9b6034',wraplength=1250).pack(anchor='w',padx=9,pady=2)
        ttk.Label(lab,text='Steps 1–3 show A/B/C contact evidence. Stage 4 automatically reports taped-reference FPA; B is provisional and C is review-only.',
                  foreground='#9b6034',wraplength=1250).pack(anchor='w',padx=9,pady=2)
        self.fpa_lab_status=ttk.Label(lab,text='Load left and right IMU CSVs, then click PLOT. '
            'Video and detector settings never enter this laboratory.',wraplength=1250,justify='left')
        self.fpa_lab_status.pack(fill='x',padx=8,pady=6)

        self.metrics=ttk.Label(outer,text='Load CSV files to compare raw and processed signals.',wraplength=1350)

        self.metrics.pack(anchor='w',fill='x',pady=3)



    def _page_mousewheel(self,event):
        # Windows uses event.delta in units of 120.  Do not steal the wheel from
        # the embedded Matplotlib graph or a combobox's pop-down.
        widget=event.widget
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
            plot_widget=self.plotcanvas.get_tk_widget()
            if widget==plot_widget or str(widget).startswith(str(plot_widget)+'.'):
                return
        except (AttributeError, tk.TclError):
            pass
        amount=-int(event.delta/120) if event.delta else 0
        if amount==0 and event.delta:
            amount=-1 if event.delta>0 else 1
        if amount:
            self.page_canvas.yview_scroll(amount*3,'units')

    def _page_scroll_linux(self,event,direction):
        self.page_canvas.yview_scroll(direction*3,'units')

    def scale_field(self,parent,title,var,lo,hi,step=None):
        """Slider and editable exact numeric value, restored with each axis."""
        box=ttk.Frame(parent)
        box.pack(side='left',fill='x',expand=True,padx=5)
        display=tk.StringVar()
        label=ttk.Label(box,text=title,anchor='w')
        label.pack(anchor='w')
        line=ttk.Frame(box)
        line.pack(fill='x')
        resolution=step if step is not None else (0.1 if hi >= 5 else 0.01)
        slider=tk.Scale(line,from_=lo,to=hi,resolution=resolution,
                        variable=var,orient='horizontal',showvalue=False,length=125,
                        command=lambda _value:self._slider_moved())
        slider.pack(side='left',fill='x',expand=True)
        value_entry=ttk.Entry(line,textvariable=display,width=8,justify='right')
        value_entry.pack(side='left',padx=(3,0))

        def format_value():
            v=float(var.get())
            return str(int(round(v))) if step==1 else f'{v:.2f}'.rstrip('0').rstrip('.')

        def on_var_change(*_):
            if not self.root.winfo_exists():
                return
            # An entry is committed only on Return/FocusOut; slider motion
            # must always refresh the numeric readout, even after entry focus.
            display.set(format_value())
            if var is self.peak_prominence:
                title_unit='RMS peak prominence (°/s)' if self.active_channel in (*GYRO,CHANNELS[0]) else 'RMS peak prominence (m/s²)'
                label.config(text=title_unit)
            if not self._switching_channel:
                self.schedule_plot()
        var.trace_add('write',on_var_change)

        def commit(_event=None):
            try:
                new=float(display.get())
                if not math.isfinite(new) or new < lo or new > hi:
                    raise ValueError
                if step==1:
                    new=int(round(new))
                var.set(new)
                self._snapshot_current_profile()
            except (ValueError,tk.TclError):
                self.root.bell()
            display.set(format_value())
            self.root.focus_set()
        value_entry.bind('<Return>',commit)
        value_entry.bind('<FocusOut>',commit)
        display.set(format_value())
        self._slider_fields.append((var,display,format_value))
        return slider,label

    def _slider_moved(self):
        if not self._switching_channel:
            self.schedule_plot()

    def _refresh_slider_entries(self):
        for var,textvar,formatter in self._slider_fields:
            textvar.set(formatter())

    def choose_csvs(self):

        l=filedialog.askopenfilename(title='Select LEFT raw CSV',filetypes=[('CSV','*.csv')])

        if not l: return

        r=filedialog.askopenfilename(title='Select RIGHT raw CSV',filetypes=[('CSV','*.csv')])

        if r: self.open_csvs(l,r)



    def open_csvs(self,l,r):

        try:

            self.data={'left':load_imu(l),'right':load_imu(r)}
            self.fpa_rows={}
            self.combined_rows={}
            self.selected_step=0

            self.current_t=0.

            self.status.config(text=f'L {self.data["left"]["rows"]} / R {self.data["right"]["rows"]} rows')

            self.schedule_plot()

        except Exception as e: messagebox.showerror('CSV load failed',str(e))



    def choose_video(self):

        path=filedialog.askopenfilename(title='Open reference walking video',filetypes=[('Videos','*.mp4 *.mov *.avi *.mkv'),('All files','*.*')])

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

        if event.xdata is None:return
        side=None
        for name,ax in zip(('left','right'),self.axes):
            if event.inaxes is ax or event.inaxes is getattr(ax,'_impact_twin',None):
                side=name;break
        if side is None:return
        self.current_t=float(event.xdata)
        self.select_nearest_marker(self.current_t,side)

        self.seek_video(self.current_t+self.offset.get())

        self.draw_cursor()



    def schedule_plot(self):

        if self.plot_job is not None:

            try:self.root.after_cancel(self.plot_job)

            except Exception:pass

        self.plot_job=self.root.after(160,self.refresh_plot)



    @staticmethod
    def _default_profile(channel, cfg):
        impact = cfg['impact']
        gyro_channel = channel == CHANNELS[0] or channel in GYRO
        acc_impact_channel = channel in ACC or channel == CHANNELS[1]
        if channel == 'az_mps2':
            method='Butterworth band-pass'
            lo=float(impact['low_hz']); hi=float(impact['high_hz'])
        elif channel == CHANNELS[1]:
            method='Butterworth low-pass';lo=15.0;hi=23.0
        elif acc_impact_channel:
            method='Butterworth band-pass';lo=11.0;hi=23.0
        else:
            method='Butterworth low-pass';lo=10.0;hi=23.0
        return dict(method=method,cutoff=lo,high_cut=hi,window=75.0,
            order=4,notch=50.0,rms_window=float(impact['rms_window_ms']) if channel=='az_mps2' else 75.0,
            prominence=float(impact['prominence_mps2']) if channel=='az_mps2' else (22.0 if gyro_channel else 1.0),
            spacing=float(impact['min_spacing_s']) if channel=='az_mps2' else 0.30)

    def _initialize_signal_profiles(self):
        self.impact_source=self.shared_config['impact'].get('axis','az_mps2')
        if self.impact_source not in (*ACC,CHANNELS[1]):
            self.impact_source='az_mps2'
        self.channel_profiles={name:self._default_profile(name,self.shared_config) for name in CHANNELS}
        saved=self.shared_config.get('per_signal_filters',{})
        for name,profile in saved.items():
            if name in self.channel_profiles and isinstance(profile,dict):
                self.channel_profiles[name].update({key:value for key,value in profile.items()
                                                    if key in self.channel_profiles[name]})
        self.active_channel=self.current_channel.get()

    def _snapshot_current_profile(self):
        if self._switching_channel:return
        self.channel_profiles[self.active_channel]=dict(
            method=self.method.get(),cutoff=float(self.cutoff.get()),
            high_cut=float(self.high_cut.get()),window=float(self.window.get()),
            order=int(self.order.get()),notch=float(self.notch.get()),
            rms_window=float(self.rms_window.get()),prominence=float(self.peak_prominence.get()),
            spacing=float(self.peak_spacing.get()))

    def _restore_profile(self,channel):
        profile=self.channel_profiles[channel]
        self._switching_channel=True
        try:
            # A high prominence range is needed for gyro (deg/s);
            # acceleration uses m/s² and needs finer increments.
            gyro=channel in GYRO or channel==CHANNELS[0]
            if hasattr(self,'peak_scale'):
                self.peak_scale.configure(from_=0.5 if gyro else .1,
                                          to=250 if gyro else 20,
                                          resolution=0.5 if gyro else .1)
                self.peak_label.config(text=('RMS peak prominence (°/s): ' if gyro
                                             else 'RMS peak prominence (m/s²): ') + f"{profile['prominence']:.1f}")
            for key,var in (('method',self.method),('cutoff',self.cutoff),
                    ('high_cut',self.high_cut),('window',self.window),('order',self.order),
                    ('notch',self.notch),('rms_window',self.rms_window),
                    ('prominence',self.peak_prominence),('spacing',self.peak_spacing)):
                var.set(profile[key])
        finally:
            self._switching_channel=False
        self._update_channel_role()

    def switch_channel(self,_event=None):
        selected=self.current_channel.get()
        if selected==self.active_channel:return
        self._snapshot_current_profile()
        self.active_channel=selected
        self._restore_profile(selected)
        self.schedule_plot()

    def set_impact_source(self):
        selected=self.current_channel.get()
        if selected not in ACC and selected!=CHANNELS[1]:
            messagebox.showinfo('Impact source','Choose an acceleration axis or Acceleration magnitude |a|.')
            return
        self._snapshot_current_profile()
        self.impact_source=selected
        self._update_channel_role()
        self.schedule_plot()

    def _update_channel_role(self):
        if not hasattr(self,'channel_role_info'):return
        ch=self.current_channel.get()
        if ch==self.impact_source:
            role='IMPACT DETECTOR — selected RMS and band-pass settings affect footstep candidates'
        elif ch==CHANNELS[0]:
            role='GYRO MOVEMENT — LP cutoff AND RMS window feed swing detection when gyro RMS option is enabled; orange peaks show movement, not impacts'
        elif ch==CHANNELS[1]:
            role='STANCE ACCEL — low-pass cutoff changes the stable-stance detector; RMS peaks are diagnostic only'
        else:
            role='SIGNAL EXPLORATION — RMS/peaks displayed; not used by detector until assigned to IMPACT'
        self.channel_role_info.config(text=f'{role}  |  Active impact: {self.impact_source}')
        self.choose_impact_button.config(state='normal' if ch in (*ACC,CHANNELS[1]) else 'disabled')

    def _apply_pipeline_config_to_controls(self):
        c=self.shared_config
        self._restore_profile(self.active_channel)
        self.gyro_threshold.set(c['stationary']['gyro_threshold_dps'])
        self.accel_tol.set(c['stationary']['gravity_tolerance_mps2'])
        self.variance.set(c['stationary']['accel_variance_mps4'])
        self.min_stance.set(c['stationary']['min_duration_ms'])
        self.use_gyro_rms.set(c.get('swing_segmentation',{}).get('use_gyro_rms_for_swing',True))
        self.gyro_rms_threshold.set(c.get('swing_segmentation',{}).get('gyro_rms_threshold_dps',40.0))
        self.gyro_rms_min_duration.set(c.get('swing_segmentation',{}).get('gyro_rms_min_swing_s',0.45))

    def _active_pipeline_config(self):
        # Store all profiles; only active IMPACT, GYRO MAGNITUDE and
        # ACCEL MAGNITUDE have detector roles. Other profiles are diagnostics.
        import copy
        self._snapshot_current_profile()
        cfg=copy.deepcopy(self.shared_config)
        cfg['per_signal_filters']=copy.deepcopy(self.channel_profiles)
        cfg['impact']['axis']=self.impact_source
        i=self.channel_profiles[self.impact_source]
        # The shared impact detector uses this signal's selected filter.
        cfg['impact'].update(method=i['method'],low_hz=float(i['cutoff']),high_hz=float(i['high_cut']),
            order=int(i['order']),window_ms=float(i['window']),notch_hz=float(i['notch']),
            rms_window_ms=float(i['rms_window']),
            prominence_mps2=float(i['prominence']),min_spacing_s=float(i['spacing']))
        cfg['stationary'].update(gyro_threshold_dps=float(self.gyro_threshold.get()),
            gravity_tolerance_mps2=float(self.accel_tol.get()),
            accel_variance_mps4=float(self.variance.get()),min_duration_ms=float(self.min_stance.get()),
            gyro_lowpass_hz=float(self.channel_profiles[CHANNELS[0]]['cutoff'])
                if self.channel_profiles[CHANNELS[0]]['method']=='Butterworth low-pass'
                else float(cfg['stationary'].get('gyro_lowpass_hz',cfg['stationary']['detector_lowpass_hz'])),
            accel_lowpass_hz=float(self.channel_profiles[CHANNELS[1]]['cutoff'])
                if self.channel_profiles[CHANNELS[1]]['method']=='Butterworth low-pass'
                else float(cfg['stationary'].get('accel_lowpass_hz',cfg['stationary']['detector_lowpass_hz'])))
        cfg.setdefault('swing_segmentation',{}).update(
            use_gyro_rms_for_swing=bool(self.use_gyro_rms.get()),
            gyro_rms_threshold_dps=float(self.gyro_rms_threshold.get()),
            gyro_rms_window_ms=float(self.channel_profiles[CHANNELS[0]]['rms_window']),
            gyro_rms_min_swing_s=float(self.gyro_rms_min_duration.get()))
        # No special first-step exception: the identical rule applies to every swing.
        cfg.setdefault('combined_detection',{})['startup_recovery_enabled']=False
        return cfg

    def save_pipeline_config(self):
        path=filedialog.asksaveasfilename(defaultextension='.json',
            initialfile='fpa_processing_config.json',filetypes=[('JSON','*.json')])
        if not path:return
        Path(path).write_text(json.dumps(self._active_pipeline_config(),indent=2),encoding='utf-8')
        messagebox.showinfo('Saved shared configuration',
           'To use these exact settings for both offline scripts, save the file as\n'
           'fpa_processing_config.json in your project folder.\n\n'+path)

    def processed(self,side):
        d=self.data[side]
        r=shared.run(d['t'],d['accel'],d['gyro'],self._active_pipeline_config())
        self.shared_results=getattr(self,'shared_results',{})
        self.shared_results[side]=r
        segmentation=detect_swings(r)
        # Raw impact/stance signal processing remains frozen. This diagnostic
        # view uses the same Phase13 A/B/C event rules as the production Stage4.
        self.fpa_rows[side]=[]
        self.combined_rows[side]=detect_multievidence(r,segmentation)
        w=r['gyro']-r['bias'] if self.bias.get() else r['gyro']
        opts=dict(method=self.method.get(),cutoff=self.cutoff.get(),
                  high_cut=self.high_cut.get(),window_ms=self.window.get(),
                  order=self.order.get(),notch_hz=self.notch.get())
        filtered_g=filter_signal(w,d['fs'],**opts)
        filtered_a=filter_signal(d['accel'],d['fs'],**opts)
        return dict(gyro_raw=w,gyro=filtered_g,accel_raw=d['accel'],accel=filtered_a,
            accel_mag_filtered=filter_signal(np.linalg.norm(d['accel'],axis=1),d['fs'],**opts),
            mask=r['mask'],intervals=[(s['start_s'],s['end_s']) for s in r['stationary']],
            bias=r['bias'],avar=r['accel_variance'],v21=segmentation)

    def calculate_impacts(self,side,p=None):
        r=getattr(self,'shared_results',{}).get(side)
        if r is None: r=shared.run(self.data[side]['t'],self.data[side]['accel'],
                                   self.data[side]['gyro'],self._active_pipeline_config())
        idx=np.array([x['index'] for x in r['contacts']],dtype=int)
        return dict(envelope=r['impact_envelope'],indices=idx,
            properties={'prominences':np.array([x['prominence_mps2'] for x in r['contacts']])},
            filtered=r['impact_bandpassed'],axis=self._active_pipeline_config()['impact']['axis'])

    def get_channel(self,d,p,processed):

        name=self.current_channel.get()

        if name == CHANNELS[0]:return np.linalg.norm(p['gyro'] if processed else p['gyro_raw'],axis=1)

        if name == CHANNELS[1]:return p['accel_mag_filtered'] if processed else np.linalg.norm(p['accel_raw'],axis=1)

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

                if getattr(ax,'_impact_twin',None) is not None:
                    try:ax._impact_twin.remove()
                    except Exception:pass
                    ax._impact_twin=None
                ax.clear()

                ax.plot(d['t'],raw,color='0.65',lw=.75,label='Raw',alpha=.72)

                ax.plot(d['t'],flt,color='#157da5',lw=1.15,label='Filtered')

                if self.show_rms.get() or self.show_impact_peaks.get():
                    # Plot positive RMS energy against a dedicated right-hand axis.
                    # This prevents the envelope from disappearing against the
                    # raw acceleration's +/- 40 m/s^2 vertical scale.
                    impact=self.calculate_impacts(side)
                    # All selected-signal RMS/peaks are calculated independently
                    # of the footstep detector, except when viewing its source.
                    profile=self.channel_profiles[self.current_channel.get()]
                    if self.current_channel.get()==self.impact_source:
                        shown=impact
                    else:
                        env,peaks,properties=impact_envelope_and_peaks(
                            d['t'],flt,d['fs'],rms_window_ms=float(profile['rms_window']),
                            min_prominence=float(profile['prominence']),
                            min_gap_s=float(profile['spacing']))
                        shown=dict(envelope=env,indices=peaks,properties=properties,
                                   filtered=flt,axis=self.current_channel.get())
                    p['impact']=impact
                    p['selected_rms']=shown
                    ax_energy=ax.twinx()
                    if self.show_rms.get():
                        ax_energy.plot(d['t'],shown['envelope'],color='#14856c',
                                       lw=1.5,alpha=.88,label='Selected-signal RMS')
                    idx=shown['indices']
                    if len(idx) and self.show_impact_peaks.get():
                        ax_energy.scatter(d['t'][idx],shown['envelope'][idx],
                                          color='#d18b2e',marker='v',s=38,
                                          zorder=6,label='Selected-signal RMS peaks')
                    if self.show_combined.get():
                        for status,color,label in (
                            ('accepted_candidate','#087a38','A: impact + stance'),
                            ('stance_supported_candidate','#d97706','B: stance-supported'),
                            ('soft_stance_review','#a855f7','C: soft-stance review')):
                            accepted=[e for e in self.combined_rows.get(side,[])
                                      if e['status']==status and e['contact_index'] is not None]
                            if accepted:
                                accepted_idx=np.array([e['contact_index'] for e in accepted],dtype=int)
                                ax_energy.scatter(d['t'][accepted_idx],shown['envelope'][accepted_idx],
                                    color=color,edgecolors='white',linewidths=.8,marker='D',s=72,
                                    zorder=9,label=label)
                    unit='°/s' if self.current_channel.get() in (*GYRO,CHANNELS[0]) else 'm/s²'
                    ax_energy.set_ylabel('Selected RMS ('+unit+')',color='#14856c',fontsize=9)
                    ax_energy.tick_params(axis='y',colors='#14856c',labelsize=8)
                    ax_energy.set_ylim(bottom=0)
                    ax_energy.grid(False)
                    # Explicitly remove old axes from previous refresh.
                    ax._impact_twin=ax_energy
                if self.stance.get():
                    for i,st in enumerate(p['v21']['stands']):
                        ax.axvspan(st['start_s'],st['end_s'],color='#3d9a77',alpha=.20,
                            label='V2.1 consolidated stance' if not i else None)
                    recovered_swing_ids={e['swing_id'] for e in self.combined_rows.get(side,[])
                                         if e['status']=='recovered_startup_candidate'}
                    for i,sw in enumerate(p['v21']['swings']):
                        if sw['valid_swing']:
                            ax.axvspan(sw['start_s'],sw['end_s'],color='#2563eb',alpha=.075,
                                label='V2.1 swing candidate' if i==0 else None)
                        elif sw['swing_id'] in recovered_swing_ids:
                            ax.axvspan(sw['start_s'],sw['end_s'],color='#d97706',alpha=.16,
                                label='Startup rescue to review')
                        else:
                            ax.axvspan(sw['start_s'],sw['end_s'],color='#dc2626',alpha=.12)
                    for st in p['v21']['stands']:
                        if st['reference_s'] is not None:
                            ax.axvline(st['reference_s'],color='#107c50',alpha=.4,lw=.75,ls=':')

                # FPA labels are shown ONLY for impact+swing+stance-supported
                # candidates. The old independent purple labels could appear
                # unrelated to the orange RMS markers and were misleading.
                if self.show_combined.get():
                    for event in self.combined_rows.get(side,[]):
                        if event['status'] not in ('accepted_candidate','stance_supported_candidate','soft_stance_review') or event['contact_time_s'] is None:
                            continue
                        xx=event['contact_time_s']
                        # Event is IMU-only. No old yaw-derived FPA labels.
                        quality=event.get('confidence','A_CONFIRMED')
                        color=('#087a38' if quality=='A_CONFIRMED' else '#d97706' if quality=='B_PROVISIONAL' else '#a855f7')
                        ax.axvline(xx,color=color,lw=1,alpha=.55)
                        txt='A' if quality=='A_CONFIRMED' else 'B' if quality=='B_PROVISIONAL' else 'C?'
                        ax.text(xx,.92,txt,
                            transform=ax.get_xaxis_transform(),color=color,
                            fontsize=8,rotation=45,va='top')
                ax.grid(alpha=.18)

                ax.set_ylabel(f'{side.upper()}\n'+('°/s' if self.current_channel.get() in GYRO or self.current_channel.get()==CHANNELS[0] else 'm/s²'))

                for mark in self.labels:

                    if mark['side']==side:ax.axvline(mark['imu_time_s'],color='#ce602f',alpha=.65,lw=1,ls='--')

                ax.legend(loc='upper right',fontsize=8,ncol=3)

                impact_count=f', {len(self.shared_results[side]["contacts"])} contact candidates, {sum(x["status"]=="matched_candidate" for x in self.shared_results[side]["pairs"])} matched'
                combined_n=sum(x['status']=='accepted_candidate' for x in self.combined_rows.get(side,[]))
                recovered_n=sum(x['confidence']=='B_PROVISIONAL' for x in self.combined_rows.get(side,[]))
                soft_n=sum(x['confidence']=='C_REVIEW_ONLY' for x in self.combined_rows.get(side,[]))
                stats.append(f'{side}: {len(p["v21"]["stands"])} consolidated stances, {sum(x["valid_swing"] for x in p["v21"]["swings"])} swing candidates{impact_count}, {combined_n} A strict + {recovered_n} B stance-supported + {soft_n} C review, bias [{", ".join(f"{x:.2f}" for x in p["bias"])}] °/s')

            self.axes[-1].set_xlabel('IMU time since first recorded sample (s) — click to seek video')

            self.axes[-1].set_xlim(0,max(self.data['left']['t'][-1],self.data['right']['t'][-1]))

            self.figure.suptitle(self.current_channel.get()+' | '+self.method.get()+' | selected RMS + combined detector (UNVALIDATED)',fontsize=11)

            suffix=f' | IMPACT {self.impact_source} ({self.channel_profiles[self.impact_source]["method"]})' + f' | GYRO LP {self.channel_profiles[CHANNELS[0]]["cutoff"]:.0f} Hz' + f' | STANCE ACC LP {self.channel_profiles[CHANNELS[1]]["cutoff"]:.0f} Hz'

            self.metrics.config(text=' | '.join(stats)+' | video offset '+f'{self.offset.get():+.2f}s'+suffix)

            self.draw_cursor(redraw=False)

            self.describe_step()
            self.plotcanvas.draw_idle()

        except Exception as e:

            self.metrics.config(text='Filter error: '+str(e))



    def _raw6_lab_config(self):
        return dict(gyro_hz=float(self.fpa_gyro_hz.get()),
            accel_hz=float(self.fpa_accel_hz.get()),order=int(self.fpa_filter_order.get()),
            bias_start_s=float(self.fpa_bias_start.get()),bias_end_s=float(self.fpa_bias_end.get()),
            correct_gyro_bias=bool(self.fpa_correct_bias.get()),
            filter_gyro=bool(self.fpa_filter_gyro.get()),filter_accel=bool(self.fpa_filter_accel.get()))

    def _apply_raw6_lab_config(self,options):
        match={'gyro_hz':self.fpa_gyro_hz,'accel_hz':self.fpa_accel_hz,
            'order':self.fpa_filter_order,'bias_start_s':self.fpa_bias_start,
            'bias_end_s':self.fpa_bias_end,'correct_gyro_bias':self.fpa_correct_bias,
            'filter_gyro':self.fpa_filter_gyro,'filter_accel':self.fpa_filter_accel}
        for key,var in match.items():
            if key in options:var.set(options[key])

    def _compute_raw6_lab(self):
        if not self.data or any(side not in self.data for side in ('left','right')):
            raise ValueError('Load both left and right raw CSVs first.')
        cfg=self._raw6_lab_config()
        return {side:filter_raw_six_axis(self.data[side]['t'],
            self.data[side]['accel'],self.data[side]['gyro'],**cfg)
            for side in ('left','right')}

    def show_raw6_lab(self):
        """Plot ALL six raw measured channels for EACH foot, with filter overlay.

        Deliberately avoids self.processed(), shared.run(), legacy heading,
        detected steps, manual video marks, video offsets and known angles.
        """
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        try:
            results=self._compute_raw6_lab()
            z0=float(self.fpa_zoom_start.get())
            z1=float(self.fpa_zoom_end.get())
            if z0<0 or z0>=z1:raise ValueError('Graph range must satisfy 0 <= start < end')
        except Exception as e:
            messagebox.showerror('Raw 6-axis filter lab',str(e))
            return
        if self.fpa_lab_window is None or not self.fpa_lab_window.winfo_exists():
            self.fpa_lab_window=tk.Toplevel(self.root)
            self.fpa_lab_window.title('ACIT4035 | RAW 6-axis FPA filter lab — left + right')
            self.fpa_lab_window.geometry('1400x890')
            self.fpa_lab_window.minsize(900,570)
            frame=ttk.Frame(self.fpa_lab_window)
            frame.pack(fill='both',expand=True)
            figure=Figure(figsize=(14,11),dpi=100)
            axes=figure.subplots(6,2,sharex='col')
            figure.subplots_adjust(left=.075,right=.975,top=.91,bottom=.075,wspace=.17,hspace=.20)
            canvas=FigureCanvasTkAgg(figure,master=frame)
            canvas.get_tk_widget().pack(fill='both',expand=True)
            NavigationToolbar2Tk(canvas,frame,pack_toolbar=False).pack(fill='x')
            self.fpa_lab_figure=figure
            self.fpa_lab_axes=axes
            self.fpa_lab_canvas=canvas
        names=list(GYRO_COLUMNS)+list(ACC_COLUMNS)
        cfg=self._raw6_lab_config()
        for col,side in enumerate(('left','right')):
            result=results[side]
            raw=np.column_stack((result['gyro_raw'],result['accel_raw']))
            processed=np.column_stack((result['gyro_filtered'],result['accel_filtered']))
            t=result['time_s']
            for row,name in enumerate(names):
                ax=self.fpa_lab_axes[row,col]
                ax.clear()
                ax.plot(t,raw[:,row],lw=.65,alpha=.6,color='#777777',label='Raw measurement')
                ax.plot(t,processed[:,row],lw=1.1,alpha=.95,color='#087d9c',label='FPA lab filtered')
                ax.axvspan(result['bias_start_s'],result['bias_end_s'],color='#f0be50',alpha=.12)
                ax.set_ylabel(name.split('_')[0] + (' (°/s)' if row < 3 else ' (m/s²)'),fontsize=9)
                ax.set_xlim(z0,min(z1,float(t[-1])))
                ax.grid(alpha=.2)
                if row==0:
                    ax.set_title(side.upper()+f'   |   {result["fs_hz"]:.1f} samples/s',fontsize=11)
                    ax.legend(loc='upper right',fontsize=8)
                if row==5:ax.set_xlabel('IMU time (s)')
        self.fpa_lab_figure.suptitle('RAW 6-axis signals vs FPA-LAB FILTERED — no video, no fused Euler angles, no step-detector edits\n'
            f'Gyro {cfg["gyro_hz"]:g} Hz / accel {cfg["accel_hz"]:g} Hz / Butterworth order {cfg["order"]} / gyro bias {"on" if cfg["correct_gyro_bias"] else "off"}',fontsize=11)
        self.fpa_lab_canvas.draw_idle()
        captions=[]
        for side,r in results.items():
            label=(f'{side.upper()}: gyro bias XYZ [{", ".join(f"{v:+.3f}" for v in r["gyro_bias_dps"])}] °/s; '
                f'standing 3D gyro RMS {r["rms_gyro_before_dps"]:.2f} -> {r["rms_gyro_after_dps"]:.2f} °/s')
            if r['bias_window_movement_warning']:
                label+=' [WARNING: initial bias interval may contain motion]'
            captions.append(label)
        self.fpa_lab_status.config(text=' | '.join(captions)+' | The six filtered traces are NOT yet FPA angles.')
        self.fpa_lab_window.lift()

    def _raw6_orientation_config(self):
        return dict(forward_axis=self.fpa_forward_axis.get(),
                    tilt_correction_tau_s=float(self.fpa_tilt_tau.get()),
                    static_gyro_threshold_dps=float(self.fpa_static_gyro.get()),
                    accel_gravity_tolerance_mps2=float(self.fpa_gravity_tol.get()))

    def _apply_raw6_orientation_config(self,options):
        mapping={'forward_axis':self.fpa_forward_axis,
                 'tilt_correction_tau_s':self.fpa_tilt_tau,
                 'static_gyro_threshold_dps':self.fpa_static_gyro,
                 'accel_gravity_tolerance_mps2':self.fpa_gravity_tol}
        for key,var in mapping.items():
            if key in options: var.set(options[key])

    def _compute_raw6_orientation(self):
        filtered=self._compute_raw6_lab()  # ONLY IMU timestamps/accel/gyro.
        opts=self._raw6_orientation_config()
        return {side:estimate_six_axis_orientation(filtered[side],**opts)
                for side in ('left','right')}

    def show_raw6_orientation(self):
        """Plot purely IMU-derived orientation; does not call step/video paths."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        try:
            results=self._compute_raw6_orientation()
            z0=float(self.fpa_zoom_start.get())
            z1=float(self.fpa_zoom_end.get())
            if z0<0 or z0>=z1:raise ValueError('Graph range must satisfy 0 <= start < end')
        except Exception as e:
            messagebox.showerror('Raw-six-axis orientation',str(e));return
        if self.fpa_orientation_window is None or not self.fpa_orientation_window.winfo_exists():
            self.fpa_orientation_window=tk.Toplevel(self.root)
            self.fpa_orientation_window.title('ACIT4035 | QUATERNION orientation from RAW six axes — NOT FPA')
            self.fpa_orientation_window.geometry('1450x875')
            self.fpa_orientation_window.minsize(900,580)
            frame=ttk.Frame(self.fpa_orientation_window);frame.pack(fill='both',expand=True)
            figure=Figure(figsize=(14,10),dpi=100)
            axes=figure.subplots(4,2,sharex='col')
            figure.subplots_adjust(left=.08,right=.975,top=.89,bottom=.075,wspace=.19,hspace=.29)
            canvas=FigureCanvasTkAgg(figure,master=frame)
            canvas.get_tk_widget().pack(fill='both',expand=True)
            NavigationToolbar2Tk(canvas,frame,pack_toolbar=False).pack(fill='x')
            self.fpa_orientation_figure=figure
            self.fpa_orientation_axes=axes
            self.fpa_orientation_canvas=canvas
        for col,side in enumerate(('left','right')):
            r=results[side];t=r['time_s']
            axs=self.fpa_orientation_axes[:,col]
            for ax in axs:
                ax.clear();ax.grid(alpha=.18)
                ax.set_xlim(z0,min(z1,float(t[-1])))
                ax.axvspan(*r['initial_reference_s'],color='#e7bd55',alpha=.16,lw=0)
            axs[0].plot(t,r['gyro_only_sensor_axis_azimuth_deg'],color='#888888',lw=1,
                        alpha=.8,label='Gyro-only integration')
            axs[0].plot(t,r['relative_sensor_axis_azimuth_deg'],color='#087d9c',lw=1.3,
                        label='Gyro + trusted-gravity tilt correction')
            axs[0].axhline(0,color='#999999',lw=.7,ls=':')
            axs[0].set_ylabel('Relative SENSOR-axis\nazimuth (deg)')
            axs[0].legend(loc='upper right',fontsize=8)
            axs[0].set_title(side.upper()+f' | axis {r["sensor_axis"]} | initial yaw ARBITRARY',fontsize=11)
            axs[1].plot(t,r['rotation_magnitude_from_initial_deg'],color='#7952a6',lw=1)
            axs[1].set_ylabel('3D rotation from\ninitial pose (deg)')
            axs[2].plot(t,r['gyro_only_gravity_residual_deg'],color='#888888',alpha=.6,lw=.8,
                        label='Gyro-only')
            axs[2].plot(t,r['gravity_alignment_residual_deg'],color='#087d9c',lw=1,
                        label='Gravity-corrected')
            axs[2].set_ylabel('Accel vs up\nresidual (deg)')
            axs[2].legend(loc='upper right',fontsize=7)
            axs[3].plot(t,r['gyro_magnitude_dps'],color='#94532f',lw=.9,label='3D gyro magnitude')
            axs[3].axhline(self.fpa_static_gyro.get(),color='#9c6721',ls=':',lw=.9,
                           label='Quiet threshold')
            axs[3].set_ylabel('Angular speed\n(deg/s)')
            axs[3].legend(loc='upper right',fontsize=8)
            # Pale-green regions indicate where gravity correction was permitted,
            # NOT footstep detector stance labels; shaded only if >=60 ms.
            valid=r['trusted_tilt_correction']
            boundaries=np.diff(np.r_[False,valid,False].astype(int))
            starts=np.where(boundaries==1)[0]
            stops=np.where(boundaries==-1)[0]
            for start,stop in zip(starts,stops):
                if (t[stop-1]-t[start]) < .06: continue
                if t[start]>z1 or t[stop-1]<z0: continue
                for ax in axs:ax.axvspan(t[start],t[stop-1],color='#4d9c76',alpha=.085,lw=0)
            axs[-1].set_xlabel('IMU time (s)')
        opts=self._raw6_orientation_config()
        self.fpa_orientation_figure.suptitle(
            'IMU-ONLY 3D ORIENTATION — quaternion gyro integration + gated accelerometer tilt correction\n'
            'Relative SENSOR-axis angle, NOT foot progression angle; NO absolute heading; NO video or fused Euler angles\n'
            f'Axis {opts["forward_axis"]}  |  correction tau {opts["tilt_correction_tau_s"]:.2g}s  |  green = trusted accelerometer, amber = zero reference',
            fontsize=11)
        self.fpa_orientation_canvas.draw_idle()
        parts=[]
        for side,r in results.items():
            residual=r['median_trusted_tilt_residual_deg']
            parts.append(f'{side.upper()}: gravity correction permitted for {100*r["trusted_fraction"]:.1f}% of samples; '
                         f'quiet gravity residual {residual:.2f} deg' if residual is not None
                         else f'{side.upper()}: no gravity-trusted samples')
        self.fpa_lab_status.config(text=' | '.join(parts) +
            ' | Sensor-axis azimuth is NOT toe-out/FPA until shoe-forward axis and progression direction are calibrated.')
        self.fpa_orientation_window.lift()

    def _compute_offline_stage123(self):
        """Reuse EXACT Studio detector settings, independently from FPA lab filters.

        The downstream steps process only six measured IMU channels and time.
        This method does not access video frames or video offsets.
        """
        if not self.data or any(side not in self.data for side in ('left','right')):
            raise ValueError('Load BOTH left and right raw IMU CSVs')
        cfg=self._active_pipeline_config()
        lab=self._raw6_lab_config()
        orient=self._raw6_orientation_config()
        mounts={'left':self.fpa_mount_left.get(),'right':self.fpa_mount_right.get()}
        results={}
        for side in ('left','right'):
            raw=self.data[side]
            results[side]=process_offline_stage123(
                raw['t'],raw['accel'],raw['gyro'],side=side,
                detector_config=cfg,lab_config=lab,orientation_config=orient,
                mount_forward=mounts[side])
        return results,cfg,lab,orient,mounts

    def show_offline_stage123(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        try:
            results,_,_,_,_=self._compute_offline_stage123()
            fig=plot_stage123(results)
        except Exception as e:
            messagebox.showerror('Offline IMU stages 1–3',str(e));return
        if self.fpa_stage123_window is None or not self.fpa_stage123_window.winfo_exists():
            self.fpa_stage123_window=tk.Toplevel(self.root)
            self.fpa_stage123_window.title('OFFLINE Stage 1–3 | IMU-only accepted stances and orientation')
            self.fpa_stage123_window.geometry('1450x925')
            self.fpa_stage123_window.minsize(950,600)
            frame=ttk.Frame(self.fpa_stage123_window);frame.pack(fill='both',expand=True)
            self.fpa_stage123_canvas=FigureCanvasTkAgg(fig,master=frame)
            self.fpa_stage123_canvas.get_tk_widget().pack(fill='both',expand=True)
            NavigationToolbar2Tk(self.fpa_stage123_canvas,frame,pack_toolbar=False).pack(fill='x')
        else:
            old=self.fpa_stage123_canvas
            old.get_tk_widget().destroy()
            container=old.get_tk_widget().master
            self.fpa_stage123_canvas=FigureCanvasTkAgg(fig,master=container)
            self.fpa_stage123_canvas.get_tk_widget().pack(fill='both',expand=True)
        self.fpa_stage123_figure=fig
        self.fpa_stage123_canvas.draw_idle()
        parts=[]
        for side,x in results.items():
            count=len(x['accepted_links'])
            pc=x['calibration']
            parts.append(f"{side.upper()}: {count} candidate IMU contacts (A/B/C); "
                         f"PCA={pc['reason']}")
        self.fpa_lab_status.config(text=' | '.join(parts) +
            ' | Foot axis sign NOT verified, NO FPA, no video, no fused Euler angles.')
        self.fpa_stage123_window.lift()

    def export_offline_stage123(self):
        try:
            results,cfg,lab,orient,mounts=self._compute_offline_stage123()
        except Exception as e:
            messagebox.showerror('Offline stages 1–3',str(e));return
        folder=filedialog.askdirectory(title='Save offline step 1–3 IMU-only diagnostics')
        if not folder:return
        try:
            report=export_stage123(results,folder,detector_config=cfg,
                                   lab_config=lab,orientation_config=orient,mount_axes=mounts)
            fig=plot_stage123(results)
            fig.savefig(Path(folder)/'offline_stage123_diagnostics.png',dpi=140)
            messagebox.showinfo('Offline stage 1–3 export',
                'Exported accepted IMU contacts, provisional PCA axes, and orientation timelines, '
                'and orientation traces. NO FPA or calibrated accuracy claimed. '
                'Original IMU files and detector config are unchanged.')
        except Exception as e:
            messagebox.showerror('Offline stages 1–3 export failed',str(e))

    def create_independent_calibration(self):
        if not self.fpa_confirm_straight.get():
            messagebox.showwarning('Calibration confirmation',
                'Confirm the loaded trial was a straight, uninterrupted forward walking calibration (no turns).')
            return
        if not self.fpa_confirm_mount_both.get():
            messagebox.showwarning('Mount confirmation',
                'Confirm the physical +Y axis faced the toes on BOTH shoes during calibration.')
            return
        try:
            from fpa_frozen_profile import frozen_settings
            cfg,lab,orient,mounts=frozen_settings()
            if not self.data or any(side not in self.data for side in ('left','right')):
                raise ValueError('Load independent raw6 calibration samples first')
            results={}
            for side in ('left','right'):
                raw=self.data[side]
                results[side]=process_offline_stage123(raw['t'],raw['accel'],raw['gyro'],
                    side=side,detector_config=cfg,lab_config=lab,
                    orientation_config=orient,mount_forward=mounts[side])
            pair=build_calibration(results,cfg,lab,orient,mounts,
                forward_walk_confirmed=True,mounting_verified=True)
        except Exception as e:
            messagebox.showerror('Calibration rejected',str(e));return
        folder=filedialog.askdirectory(title='Select output folder for independent left/right calibration JSONs')
        if not folder:return
        try:
            save_pair(pair,folder)
            self.fpa_calibration_folder=Path(folder)
            self.fpa_calibration_info.set('Calibration loaded: '+str(folder))
            messagebox.showinfo('Independent calibration saved',
                'Two independent calibration JSONs saved. Load a DIFFERENT walking trial before Stage 4. '
                'Research-only; no independent anatomical accuracy validation has been performed.')
        except Exception as e:
            messagebox.showerror('Calibration save failed',str(e))

    def load_independent_calibration(self):
        folder=filedialog.askdirectory(title='Choose folder containing left_foot_calibration.json and right_foot_calibration.json')
        if not folder:return
        try:
            from fpa_frozen_profile import frozen_settings
            cfg,lab,orient,mounts=frozen_settings()
            load_pair(folder,cfg,lab,orient,mounts)
            self.fpa_calibration_folder=Path(folder)
            self.fpa_calibration_info.set('Calibration loaded: '+str(folder))
        except Exception as e:
            messagebox.showerror('Calibration incompatible',str(e))

    def _compute_offline_stage4(self):
        """Use EXACT same frozen reference-line Stage 4 as Windows/Pi browser."""
        from fpa_frozen_profile import frozen_settings
        from fpa_unified_pipeline import process_arrays
        if self.fpa_calibration_folder is None:
            raise ValueError('Load an independent, FROZEN-PROFILE left/right calibration folder first')
        if not self.data or any(side not in self.data for side in ('left','right')):
            raise ValueError('Load both raw six-channel recordings')
        cfg,lab,orient,mounts=frozen_settings()
        external=load_pair(self.fpa_calibration_folder,cfg,lab,orient,mounts)
        raw={side:((self.data[side]['shared_host_t'] if self.data[side].get('shared_host_t') is not None else self.data[side]['t']),self.data[side]['accel'],self.data[side]['gyro'])
             for side in ('left','right')}
        verified={'left':self.fpa_verify_mount_left.get(),
                  'right':self.fpa_verify_mount_right.get()}
        results,stage4=process_arrays(raw,calibration=external,
                mounting_verified=verified,
                reference_line_mode=True,
                shared_host_clock_verified=all(self.data[s].get('shared_host_t') is not None for s in ('left','right')))
        return results,stage4,cfg,lab,orient,mounts

    def show_offline_stage4(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
        try:
            results,stage4,_,_,_,_=self._compute_offline_stage4()
            fig=plot_stage4(results,stage4)
        except Exception as e:
            messagebox.showerror('Offline Stage 4',str(e));return
        if self.fpa_stage4_window is None or not self.fpa_stage4_window.winfo_exists():
            self.fpa_stage4_window=tk.Toplevel(self.root)
            self.fpa_stage4_window.title('OFFLINE Stage 4 | IMU-only signed FPA diagnostics, unvalidated')
            self.fpa_stage4_window.geometry('1420x920')
            self.fpa_stage4_window.minsize(930,570)
            frame=ttk.Frame(self.fpa_stage4_window);frame.pack(fill='both',expand=True)
            review_frame=ttk.LabelFrame(frame,text='C · REVIEW-ONLY FPA ESTIMATES (unconfirmed contacts)')
            review_frame.pack(fill='x',padx=9,pady=5)
            self.fpa_stage4_review_text=tk.Text(review_frame,height=4,wrap='none',font=('Consolas',10))
            self.fpa_stage4_review_text.pack(fill='x',padx=6,pady=4)
            self.fpa_stage4_review_text.configure(state='disabled')
            diff_frame=ttk.LabelFrame(frame,text='SYNCHRONOUS TWO-IMU RELATIVE HEADING · DOUBLE SUPPORT')
            diff_frame.pack(fill='x',padx=9,pady=5)
            self.fpa_stage4_differential_text=tk.Text(diff_frame,height=4,wrap='none',font=('Consolas',10))
            self.fpa_stage4_differential_text.pack(fill='x',padx=6,pady=4)
            self.fpa_stage4_differential_text.configure(state='disabled')
            crp_frame=ttk.LabelFrame(frame,text='EXPLORATORY TWO-IMU CONTINUOUS RELATIVE PHASE (CRP)')
            crp_frame.pack(fill='x',padx=9,pady=5)
            self.fpa_stage4_crp_text=tk.Text(crp_frame,height=3,wrap='none',font=('Consolas',10))
            self.fpa_stage4_crp_text.pack(fill='x',padx=6,pady=4)
            self.fpa_stage4_crp_text.configure(state='disabled')
            self.fpa_stage4_canvas=FigureCanvasTkAgg(fig,master=frame)
            self.fpa_stage4_canvas.get_tk_widget().pack(fill='both',expand=True)
            NavigationToolbar2Tk(self.fpa_stage4_canvas,frame,pack_toolbar=False).pack(fill='x')
        else:
            old=self.fpa_stage4_canvas
            holder=old.get_tk_widget().master
            old.get_tk_widget().destroy()
            self.fpa_stage4_canvas=FigureCanvasTkAgg(fig,master=holder)
            self.fpa_stage4_canvas.get_tk_widget().pack(fill='both',expand=True)
        self.fpa_stage4_figure=fig
        self.fpa_stage4_canvas.draw_idle()
        review=[]
        for side,item in stage4.items():
            for row in item['rows']:
                if row.get('detector_confidence')=='C_REVIEW_ONLY':
                    angle=row.get('fpa_deg')
                    label=f'{angle:+.2f}°' if angle is not None else 'Unavailable'
                    review.append((float(row['contact_time_s']),side,label))
        review.sort()
        lines=['Foot    Contact (s)   Provisional reference-line FPA']
        lines.extend(f'{side.upper():5s}   {tm:8.2f}     {angle}' for tm,side,angle in review)
        if not review:lines.append('No C review-only candidates in this recording.')
        if self.fpa_stage4_review_text is not None:
            self.fpa_stage4_review_text.configure(state='normal')
            self.fpa_stage4_review_text.delete('1.0','end')
            self.fpa_stage4_review_text.insert('1.0','\n'.join(lines))
            self.fpa_stage4_review_text.configure(state='disabled')
        sync=results['left'].get('synchronous_relative',{})
        summary=sync.get('summary',{})
        def show_angle(value):
            return ('unavailable' if value is None else f'{float(value):+.2f}°')
        lines=[f"Synchronous relative heading: {summary.get('status','UNAVAILABLE')} · {summary.get('samples',0)} shared-clock samples",
               f"Median absolute synchronous heading separation during walking: {show_angle(summary.get('median_abs_moving_relative_heading_deg'))}; peak absolute separation: {show_angle(summary.get('max_abs_moving_relative_heading_deg'))}",
               'Both quaternions sampled at the SAME host time; no sequential landing pairs.']
        for w in sync.get('double_support_periods',[]):
            lines.append(f"Both stationary {w['start_s']:7.2f}–{w['end_s']:7.2f}s: {w['median_relative_heading_deg']:+.2f}° relative heading")
        if summary.get('status')!='OK':
            lines.append('Unavailable: '+str(summary.get('reason')))
        elif not sync.get('double_support_periods'):
            lines.append('No confirmed double-support window after the taped zero hold.')
        if self.fpa_stage4_differential_text is not None:
            self.fpa_stage4_differential_text.configure(state='normal')
            self.fpa_stage4_differential_text.delete('1.0','end')
            self.fpa_stage4_differential_text.insert('1.0','\n'.join(lines))
            self.fpa_stage4_differential_text.configure(state='disabled')
        crp=results['left'].get('bilateral_crp',{}).get('summary',{})
        crp_lines=[f"Status: {crp.get('status','UNAVAILABLE')} · {crp.get('samples',0)} paired host-timestamp CRP samples",
                   f"Circular mean CRP: {show_angle(crp.get('circular_mean_relative_phase_deg'))}; mean deviation from 180°: {show_angle(crp.get('mean_abs_antiphase_deviation_deg'))}",
                   'Exploratory zero-phase, gait-band Hilbert estimate. Not validated clinical phase or independent ground truth.']
        if crp.get('reason'):crp_lines.append('Unavailable: '+str(crp['reason']))
        if self.fpa_stage4_crp_text is not None:
            self.fpa_stage4_crp_text.configure(state='normal')
            self.fpa_stage4_crp_text.delete('1.0','end')
            self.fpa_stage4_crp_text.insert('1.0','\n'.join(crp_lines))
            self.fpa_stage4_crp_text.configure(state='disabled')
        parts=[]
        for side,item in stage4.items():
            rows=item['rows']
            statuses={}
            for r in rows:
                status=r['fpa_status'];statuses[status]=statuses.get(status,0)+1
            parts.append(f"{side.upper()}: {len(rows)} IMU step rows; " +
                         ', '.join(f'{count} {name.lower()}' for name,count in statuses.items()))
        self.fpa_lab_status.config(text=' | '.join(parts) +
             ' | FPA is the landing angle relative to the physical taped 0° walkway; no displacement reconstruction.')
        self.fpa_stage4_window.lift()

    def export_offline_stage4(self):
        try:
            results,stage4,cfg,lab,orient,mounts=self._compute_offline_stage4()
        except Exception as e:
            messagebox.showerror('Offline Stage 4',str(e));return
        folder=filedialog.askdirectory(title='Export RAW6 Stage 1–4 diagnostic records')
        if not folder:return
        try:
            export_stage123(results,folder,detector_config=cfg,lab_config=lab,
                            orientation_config=orient,mount_axes=mounts)
            plot_stage123(results).savefig(Path(folder)/'offline_stage123_diagnostics.png',dpi=140)
            manifest=export_stage4(stage4,folder,detector_config=cfg,lab_config=lab,
                                   orientation_config=orient,mount_axes=mounts,
                                   synchronous=results['left'].get('synchronous_relative'))
            from fpa_continuous_relative_phase import CRP_FIELDS
            crp=results['left'].get('bilateral_crp',{})
            with (Path(folder)/'bilateral_continuous_relative_phase.csv').open('w',newline='',encoding='utf-8') as out_f:
                writer=csv.DictWriter(out_f,fieldnames=CRP_FIELDS)
                writer.writeheader();writer.writerows(crp.get('rows',[]))
            (Path(folder)/'bilateral_continuous_relative_phase_summary.json').write_text(
                json.dumps(crp.get('summary',{}),indent=2),encoding='utf-8')
            plot_stage4(results,stage4).savefig(Path(folder)/'offline_stage4_fpa_diagnostics.png',dpi=140)
            msg='; '.join(f"{side}: {manifest['results'][side]['research_fpa_released_rows']} research estimates, "
                          f"{manifest['results'][side]['candidate_fpa_rows']} diagnostic candidates"
                          for side in ('left','right'))
            messagebox.showinfo('Offline Stage 4 exports',
                f'Exported Stage 1–4 to {folder}\n{msg}\n'
                'Phase 16 relative heading and exploratory continuous relative phase also exported. Original recordings, detector configuration and saved profiles unchanged. '
                'No clinically validated FPA claimed.')
        except Exception as e:
            messagebox.showerror('Offline Stage 4 export failed',str(e))

    def export_raw6_orientation(self):
        try: results=self._compute_raw6_orientation()
        except Exception as e:
            messagebox.showerror('Orientation export',str(e));return
        folder=filedialog.askdirectory(title='Export six-axis orientation diagnostic CSVs')
        if not folder:return
        try:
            summary={'filter_config':self._raw6_lab_config(),
                     'orientation_config':self._raw6_orientation_config(),
                     'warning':'RELATIVE SENSOR AXIS ONLY, not anatomical foot direction, stride path or validated FPA. '
                               'Unobservable absolute yaw and accumulated heading drift remain.',
                     'uses_only_raw_six_imu_channels':True}
            for side,r in results.items():
                data={'imu_time_s':r['time_s'],
                      'relative_sensor_axis_azimuth_deg':r['relative_sensor_axis_azimuth_deg'],
                      'gyro_only_relative_sensor_axis_azimuth_deg':r['gyro_only_sensor_axis_azimuth_deg'],
                      'three_d_rotation_from_initial_deg':r['rotation_magnitude_from_initial_deg'],
                      'gravity_alignment_residual_deg':r['gravity_alignment_residual_deg'],
                      'gyro_only_gravity_residual_deg':r['gyro_only_gravity_residual_deg'],
                      'trusted_accelerometer_correction':r['trusted_tilt_correction'].astype(int),
                      'gyro_magnitude_dps':r['gyro_magnitude_dps'],
                      'accel_magnitude_mps2':r['accel_magnitude_mps2']}
                for j,k in enumerate(('x','y','z','w')):
                    data['quaternion_'+k]=r['quaternion_xyzw'][:,j]
                    data['gyro_only_quaternion_'+k]=r['gyro_only_quaternion_xyzw'][:,j]
                pd.DataFrame(data).to_csv(Path(folder)/(side+'_raw6_orientation.csv'),index=False)
                summary[side]={'sensor_axis':r['sensor_axis'],
                               'initial_reference_s':r['initial_reference_s'],
                               'initial_accel_mps2':r['initial_accel_mps2'].tolist(),
                               'trusted_fraction':r['trusted_fraction'],
                               'median_trusted_tilt_residual_deg':r['median_trusted_tilt_residual_deg']}
            Path(folder,'raw6_orientation_diagnostics.json').write_text(
                json.dumps(summary,indent=2),encoding='utf-8')
            messagebox.showinfo('Orientation export complete',
                'Six-axis quaternion and relative SENSOR-axis diagnostics saved. '
                'NOT FPA. Original CSVs and step detector unchanged.')
        except Exception as e:messagebox.showerror('Orientation export failed',str(e))

    def export_raw6_lab(self):
        try:results=self._compute_raw6_lab()
        except Exception as e:
            messagebox.showerror('Raw 6-axis export',str(e));return
        folder=filedialog.askdirectory(title='Export filtered raw-6-axis FPA-lab data')
        if not folder:return
        try:
            for side,r in results.items():
                columns={'imu_time_s':r['time_s']}
                for j,name in enumerate(GYRO_COLUMNS):
                    columns[name+'_raw']=r['gyro_raw'][:,j]
                    columns[name+'_fpa_filtered']=r['gyro_filtered'][:,j]
                for j,name in enumerate(ACC_COLUMNS):
                    columns[name+'_raw']=r['accel_raw'][:,j]
                    columns[name+'_fpa_filtered']=r['accel_filtered'][:,j]
                pd.DataFrame(columns).to_csv(Path(folder)/(side+'_raw6_fpa_lab.csv'),index=False)
            Path(folder,'fpa_raw6_filter_settings.json').write_text(
                json.dumps(self._raw6_lab_config(),indent=2),encoding='utf-8')
            messagebox.showinfo('Export complete','Independent raw-six-axis filter data saved. '
                'Frozen signal filters and original raw CSVs are unchanged; Phase13 A/B/C contact rules apply.')
        except Exception as e:
            messagebox.showerror('Export failed',str(e))

    def _accepted_for_side(self,side):
        return [e for e in self.combined_rows.get(side,[])
                if e['status'] in ('accepted_candidate','stance_supported_candidate','soft_stance_review')
                and e['contact_time_s'] is not None]

    def select_step(self,delta):
        accepted=self._accepted_for_side(self.fpa_side.get())
        if not accepted:
            self.describe_step()
            return
        self.selected_step=(self.selected_step+delta)%len(accepted)
        self.describe_step()

    def select_nearest_marker(self,t,side):
        accepted=self._accepted_for_side(side)
        if accepted:
            index=min(range(len(accepted)),key=lambda k:abs(accepted[k]['contact_time_s']-t))
            if abs(accepted[index]['contact_time_s']-t)<=.25:
                self.fpa_side.set(side)
                self.selected_step=index
                self.describe_step()
                return
        reviews=[e for e in self.combined_rows.get(side,[])
                 if e['status']=='needs_review' and e['contact_time_s'] is not None]
        if reviews:
            e=min(reviews,key=lambda x:abs(x['contact_time_s']-t))
            if abs(e['contact_time_s']-t)<=.25:
                self.fpa_text.config(text=f"{side.upper()} at {t:.2f}s: NOT ACCEPTED. "
                    f"Combined: {e['reason']}; V2.1: {e.get('v21_rejection_details','not provided')}")
                return
        self.fpa_text.config(text=f"{side.upper()} at {t:.2f}s: no accepted combined footstep within 0.25 s. "
            "An RMS impact by itself is not a confirmed landing.")

    def describe_step(self):
        side=self.fpa_side.get()
        accepted=self._accepted_for_side(side)
        if not accepted:
            self.fpa_text.config(text=f'{side.upper()}: no combined accepted step candidates. '
                'Review the swing, impact, and stance evidence; do not force an FPA.')
            return
        self.selected_step=min(self.selected_step,len(accepted)-1)
        event=accepted[self.selected_step]
        # The Stage4 tab reports the actual canonical Phase13 FPA using the
        # independent foot-axis calibration and explicit taped-start protocol.
        provisional=('PROVISIONAL/REVIEW ONLY | ' if event.get('confidence')!='A_CONFIRMED' else '')
        self.fpa_text.config(text=(
            f"{side.upper()} {provisional}candidate step {self.selected_step+1}/{len(accepted)} "
            f"| RMS impact {event['contact_time_s']:.2f}s "
            f"| stance {event['stance_start_s']:.2f}–{event['stance_end_s']:.2f}s "
            f"| {event['reason']}.\n"
            'IMU step evidence only in this sandbox; use Offline Stage 4 for calibrated '
            'taped-reference FPA (assumes walking progressed parallel to the taped line).'))

    def export_fpa(self):
        messagebox.showinfo('Raw six-axis FPA pending',
            'Relative IMU orientation is available via Export orientation + diagnostic CSVs. '
            'Use Offline Stage 4 export for aligned-start walkway-referenced FPA, with A/B/C detector quality.')

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

        path=filedialog.asksaveasfilename(defaultextension='.csv',initialfile='foot_plant_video_marks.csv',filetypes=[('CSV','*.csv')])

        if not path:return

        with open(path,'w',newline='',encoding='utf-8') as f:

            w=csv.DictWriter(f,fieldnames=list(self.labels[0]));w.writeheader();w.writerows(self.labels)

        messagebox.showinfo('Saved',path)



    def apply_recovered_profile(self,which):
        """Apply prior trial-specific detector settings; never modify raw-FPA filters."""
        import copy
        self._snapshot_current_profile()
        if which not in ('73','69'):
            raise ValueError('Unknown recovered profile')
        impact_lo,impact_hi=(11.0,23.0) if which=='73' else (10.0,23.0)
        stationary_ms=87.0 if which=='73' else 160.0
        cfg=self.shared_config
        cfg['impact'].update(axis='az_mps2',method='Butterworth band-pass',
                             low_hz=impact_lo,high_hz=impact_hi,order=4,
                             rms_window_ms=75.0,prominence_mps2=1.0,min_spacing_s=0.30)
        cfg['stationary'].update(gyro_threshold_dps=50.0,min_duration_ms=stationary_ms,
                                 gyro_lowpass_hz=10.0,accel_lowpass_hz=15.0)
        self.impact_source='az_mps2'
        for name,method,lo,hi in (
            ('az_mps2','Butterworth band-pass',impact_lo,impact_hi),
            (CHANNELS[0],'Butterworth low-pass',10.0,23.0),
            (CHANNELS[1],'Butterworth low-pass',15.0,23.0)):
            self.channel_profiles[name].update(method=method,cutoff=lo,
                 high_cut=hi,order=4,rms_window=75.0)
        self.channel_profiles['az_mps2'].update(prominence=1.0,spacing=0.30)
        self.min_stance.set(stationary_ms)
        self.gyro_threshold.set(50.0)
        self._restore_profile(self.active_channel)
        self.schedule_plot()
        messagebox.showinfo('Historical detector settings',
            ('IMG_2473 / recording 73: 11–23 Hz impact, 75 ms RMS, min stationary 87 ms.'
             if which=='73' else
             'IMG_2469 / recording 69 recovery: 10–23 Hz impact, 75 ms RMS, min stationary 160 ms.')
            + '\\n\\nThis changes the current detector profile only. Save settings and '
              'Save PIPELINE config for a reproducible run. Previously created '
              'calibration files may need regeneration if the detector configuration changed.')

    def save_settings(self):

        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='fpa_filter_settings.json',filetypes=[('JSON','*.json')])

        if not path:return

        payload={'schema':'ACIT4035_SIGNAL_PROFILES_V1','per_signal_filters':self._active_pipeline_config()['per_signal_filters'],
                 'impact_source':self.impact_source,'fpa_raw6_lab':self._raw6_lab_config(),
                 'fpa_raw6_orientation_lab':self._raw6_orientation_config(),
                 'fpa_stage123_mount_axes':{'left':self.fpa_mount_left.get(),'right':self.fpa_mount_right.get()},
                 'fpa_stage4_mount_verification':{'left':self.fpa_verify_mount_left.get(),'right':self.fpa_verify_mount_right.get()},
                 'shared_config':self._active_pipeline_config(),'channel':self.current_channel.get(),'filter':self.method.get(),'cutoff_hz':self.cutoff.get(),

                 'high_cut_hz':self.high_cut.get(),'window_ms':self.window.get(),'butterworth_order':self.order.get(),

                 'notch_hz':self.notch.get(),'gyro_bias_correction':self.bias.get(),'video_offset_s':self.offset.get(),
                 'use_gyro_rms_for_swing':self.use_gyro_rms.get(),
                 'gyro_rms_threshold_dps':self.gyro_rms_threshold.get(),
                 'gyro_rms_min_swing_s':self.gyro_rms_min_duration.get(),

                 'stance_gyro_threshold_dps':self.gyro_threshold.get(),'stance_accel_tolerance_mps2':self.accel_tol.get(),

                 'stance_accel_variance_limit':self.variance.get(),'min_stance_ms':self.min_stance.get(),

                  'impact_rms_enabled':self.show_rms.get(),
                  'impact_peak_markers_enabled':self.show_impact_peaks.get(),
                  'impact_rms_window_ms':self.rms_window.get(),
                  'impact_min_prominence_mps2':self.peak_prominence.get(),
                  'impact_min_peak_spacing_s':self.peak_spacing.get(),
                  'note':'Experimental parameters. Impact events are unvalidated candidates, not heel-strike ground truth.'}

        Path(path).write_text(json.dumps(payload,indent=2),encoding='utf-8')

        messagebox.showinfo('Saved',path)



    def load_settings(self):
        path=filedialog.askopenfilename(title='Load signal filter settings',filetypes=[('JSON','*.json')])
        if not path:return
        try:
            payload=json.loads(Path(path).read_text(encoding='utf-8'))
            configs=payload.get('per_signal_filters') or payload.get('shared_config',{}).get('per_signal_filters')
            if not isinstance(configs,dict):
                raise ValueError('This settings file has no per-signal filter profiles.')
            self._snapshot_current_profile()
            for ch,profile in configs.items():
                if ch in self.channel_profiles and isinstance(profile,dict):
                    self.channel_profiles[ch].update({key:val for key,val in profile.items()
                        if key in self.channel_profiles[ch]})
            chosen=payload.get('impact_source') or payload.get('shared_config',{}).get('impact',{}).get('axis')
            if chosen in (*ACC,CHANNELS[1]):self.impact_source=chosen
            shared_cfg=payload.get('shared_config',{})
            stationary=shared_cfg.get('stationary',{})
            self.gyro_threshold.set(stationary.get('gyro_threshold_dps',self.gyro_threshold.get()))
            self.accel_tol.set(stationary.get('gravity_tolerance_mps2',self.accel_tol.get()))
            self.variance.set(stationary.get('accel_variance_mps4',self.variance.get()))
            self.min_stance.set(stationary.get('min_duration_ms',self.min_stance.get()))
            swings=shared_cfg.get('swing_segmentation',{})
            self.use_gyro_rms.set(swings.get('use_gyro_rms_for_swing',True))
            self.gyro_rms_threshold.set(swings.get('gyro_rms_threshold_dps',40.0))
            self.gyro_rms_min_duration.set(swings.get('gyro_rms_min_swing_s',0.45))
            self._restore_profile(self.active_channel)
            if isinstance(payload.get('fpa_raw6_lab'),dict):
                self._apply_raw6_lab_config(payload['fpa_raw6_lab'])
            if isinstance(payload.get('fpa_stage123_mount_axes'),dict):
                m=payload['fpa_stage123_mount_axes']
                if m.get('left') in ('+X','-X','+Y','-Y'):self.fpa_mount_left.set(m['left'])
                if m.get('right') in ('+X','-X','+Y','-Y'):self.fpa_mount_right.set(m['right'])
            if isinstance(payload.get('fpa_stage4_mount_verification'),dict):
                v=payload['fpa_stage4_mount_verification']
                self.fpa_verify_mount_left.set(v.get('left') is True)
                self.fpa_verify_mount_right.set(v.get('right') is True)
            if isinstance(payload.get('fpa_raw6_orientation_lab'),dict):
                self._apply_raw6_orientation_config(payload['fpa_raw6_orientation_lab'])
            self.schedule_plot()
            messagebox.showinfo('Loaded',f'Loaded independent signal profiles from\n{path}')
        except Exception as e:
            messagebox.showerror('Load settings failed',str(e))

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

                impacts=self.calculate_impacts(side)
                out['impact_rms_mps2']=impacts['envelope']
                out['impact_candidate']=0
                out.loc[impacts['indices'],'impact_candidate']=1
                out.to_csv(Path(folder)/f'{side}_processed_studio.csv',index=False)

            messagebox.showinfo('Saved',f'Processed CSVs saved in {folder}\nOriginal raw files unchanged.')

        except Exception as e:messagebox.showerror('Export error',str(e))



    def export_combined(self):
        """Export both accepted and rejected swing decisions for auditability."""
        if not self.data:
            messagebox.showinfo('No sensor data','Load left and right CSVs first.')
            return
        path=filedialog.asksaveasfilename(defaultextension='.csv',
            initialfile='fpa_combined_footstep_candidates.csv',filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            event_rows=[]
            for side in ('left','right'):
                event_rows.extend(export_combined_rows(
                    self.combined_rows.get(side,[]),self.fpa_rows.get(side,[]),side))
            event_rows.sort(key=lambda row:(row['contact_time_s'] if row['contact_time_s'] is not None else row['swing_end_s']))
            if not event_rows:
                messagebox.showwarning('No swing events','No events detected for export.')
                return
            with open(path,'w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(event_rows[0]))
                writer.writeheader()
                writer.writerows(event_rows)
            accepted=sum(row['status']=='accepted_candidate' for row in event_rows)
            recovered=sum(row['status']=='recovered_startup_candidate' for row in event_rows)
            messagebox.showinfo('Saved',
                f'Exported {accepted} strict + {recovered} startup-provisional candidate steps and '
                f'{len(event_rows)-accepted-recovered} events needing review.\\n{path}')
        except Exception as e:
            messagebox.showerror('Export error',str(e))

    def export_impact_events(self):
        if not self.data:
            messagebox.showinfo('No sensor data','Load left and right CSVs first.')
            return
        path=filedialog.asksaveasfilename(defaultextension='.csv',
            initialfile='fpa_impact_candidates.csv',filetypes=[('CSV','*.csv')])
        if not path:return
        try:
            events=[]
            for side,d in self.data.items():
                impacts=self.calculate_impacts(side)
                for j,i in enumerate(impacts['indices']):
                    events.append(dict(side=side,imu_time_s=round(float(d['t'][i]),5),
                                       video_time_s=round(float(d['t'][i]+self.offset.get()),5),
                                       rms_peak_mps2=round(float(impacts['envelope'][i]),5),
                                       prominence_mps2=round(float(impacts['properties']['prominences'][j]),5),
                                       axis=impacts['axis'],
                                       confirmed_by_video=False))
            events.sort(key=lambda event:event['imu_time_s'])
            fields=['side','imu_time_s','video_time_s','rms_peak_mps2',
                    'prominence_mps2','axis','confirmed_by_video']
            with open(path,'w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader();writer.writerows(events)
            messagebox.showinfo('Saved',f'Exported {len(events)} unvalidated event candidates to\n{path}')
        except Exception as e:
            messagebox.showerror('Export error',str(e))

    def close(self):

        try:
            self.root.unbind_all('<MouseWheel>')
            self.root.unbind_all('<Button-4>')
            self.root.unbind_all('<Button-5>')
        except tk.TclError:
            pass
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
