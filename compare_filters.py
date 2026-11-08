"""ACIT4035 | Offline IMU filter gallery. No input files are modified.

Usage:
 py compare_filters.py --input "recordings\\fpa_trials\\...\\left_raw.csv"
 py compare_filters.py --input left_raw.csv --channel gy --zoom 4 8 --show
 py compare_filters.py --input left_raw.csv --channel gyro_mag --export butter_15

Only diagnostic/display signals are filtered. Do not filter raw count columns;
filter choices for event detection need not match those for integration or FPA.
Requires: numpy, scipy, pandas, matplotlib.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt, savgol_filter, find_peaks

SIGNALS = ["gx_dps", "gy_dps", "gz_dps", "ax_mps2", "ay_mps2", "az_mps2"]
CHANNELS = {
    "gx": "gx_dps", "gy": "gy_dps", "gz": "gz_dps",
    "ax": "ax_mps2", "ay": "ay_mps2", "az": "az_mps2",
    "gyro_mag": "gyro_mag", "acc_mag": "acc_mag",
}
COLORS = {"raw": "#737d8c", "filtered": "#087e8b"}


def odd_samples(ms: float, fs: float, minimum: int = 3) -> int:
    n = max(minimum, round(ms * fs / 1000))
    return n if n % 2 else n + 1


def ema_offline(x: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Causal EMA, for comparison only (has phase lag)."""
    alpha = 1 - np.exp(-2 * np.pi * cutoff_hz / fs)
    result = np.empty_like(x)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    return result


def variants(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    out = {"raw": x.copy()}
    for hz in [5, 10, 15, 20, 30]:
        if hz >= fs / 2:
            continue
        sos = butter(4, hz, btype="lowpass", fs=fs, output="sos")
        out[f"butter_{hz}"] = sosfiltfilt(sos, x)
    for ms in [35, 75, 151]:
        window = odd_samples(ms, fs, minimum=5)
        if window < len(x):
            out[f"savgol_{ms}ms"] = savgol_filter(x, window, polyorder=3, mode="interp")
    for ms in [25, 75]:
        n = odd_samples(ms, fs)
        if n < len(x):
            out[f"median_{ms}ms"] = median_filter(x, size=n, mode="reflect")
    # Two-stage combination: remove isolated spikes, then attenuate fast noise.
    n = odd_samples(25, fs)
    if n < len(x) and fs > 30:
        out["median25_butter15"] = sosfiltfilt(
            butter(4, 15, btype="lowpass", fs=fs, output="sos"),
            median_filter(x, size=n, mode="reflect"))
    for ms in [25, 75]:
        sigma = (ms / 1000 * fs) / 2.355  # ms ~= Gaussian FWHM
        out[f"gaussian_{ms}ms"] = gaussian_filter1d(x, sigma=sigma, mode="reflect")
    out["ema_15"] = ema_offline(x, 15, fs)
    return out


def choose_time(df: pd.DataFrame):
    if "timestamp_midpoint_monotonic_s" in df:
        t = df["timestamp_midpoint_monotonic_s"].to_numpy(dtype=float, copy=True)
    elif "time_s" in df:
        t = df["time_s"].to_numpy(dtype=float, copy=True)
    elif "host_elapsed_s" in df:
        t = df["host_elapsed_s"].to_numpy(dtype=float, copy=True)
    else:
        raise ValueError("Missing time: expected timestamp_midpoint_monotonic_s, time_s or host_elapsed_s")
    if len(t) < 30 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Timestamps must be finite, increasing, and contain at least 30 samples")
    t -= t[0]
    fs = 1 / float(np.median(np.diff(t)))
    return t, fs


def get_channel(df, channel):
    for col in SIGNALS:
        if col not in df:
            raise ValueError(f"Missing required column {col}; this tool expects a raw FPA reader CSV")
    # For derived magnitudes, filter the component axes FIRST, then take norm.
    # This is not mathematically equivalent to filtering magnitude directly.
    if channel == "gyro_mag":
        return np.linalg.norm(df[SIGNALS[:3]].to_numpy(float), axis=1), SIGNALS[:3]
    if channel == "acc_mag":
        return np.linalg.norm(df[SIGNALS[3:]].to_numpy(float), axis=1), SIGNALS[3:]
    return df[CHANNELS[channel]].to_numpy(float), [CHANNELS[channel]]


def filtered_channel(df, channel, fs):
    raw, columns = get_channel(df, channel)
    if len(columns) == 1:
        return variants(raw, fs)
    processed = [variants(df[col].to_numpy(float), fs) for col in columns]
    return {key: np.linalg.norm(np.column_stack([v[key] for v in processed]), axis=1)
            for key in processed[0]}


def metrics(t, signals, zoom, fs):
    mask = (t >= zoom[0]) & (t <= zoom[1])
    raw = signals["raw"]
    baseline = raw[mask]
    raw_std = max(float(np.std(baseline)), 1e-12)
    raw_area = float(np.trapezoid(np.abs(baseline), t[mask]))
    rows = []
    for name, x in signals.items():
        y = x[mask]
        rows.append({
            "filter": name,
            "window_rms_change": float(np.sqrt(np.mean((y - baseline) ** 2))),
            "window_rms_change_pct_raw_sd": float(100 * np.sqrt(np.mean((y - baseline) ** 2)) / raw_std),
            "window_peak_abs_retention_pct": float(100 * np.max(np.abs(y)) / max(np.max(np.abs(baseline)), 1e-12)),
            "window_abs_area_retention_pct": float(100 * np.trapezoid(np.abs(y), t[mask]) / max(raw_area, 1e-12)),
        })
    return rows


def draw_grid(t, signals, zoom, output, channel):
    items = list(signals.items())
    ncols = 3
    nrows = int(np.ceil(len(items) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(17, 3.2 * nrows), sharex=True)
    axs = np.array(axs).ravel()
    mask = (t >= zoom[0]) & (t <= zoom[1])
    raw = signals["raw"][mask]
    for ax, (name, y) in zip(axs, items):
        ax.plot(t[mask], raw, color=COLORS["raw"], alpha=.42, lw=.85, label="Raw")
        if name != "raw":
            ax.plot(t[mask], y[mask], color=COLORS["filtered"], lw=1.4, label="Filtered")
        ax.set_title(name.replace("_", " "), fontsize=11)
        ax.grid(alpha=.18)
    for ax in axs[len(items):]:
        ax.set_visible(False)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"ACIT4035 — filter comparison | {channel} | {zoom[0]:.2f}–{zoom[1]:.2f} s\n"
                 "Gray = raw; teal = processed | Offline processing (except EMA)", fontsize=14)
    fig.supxlabel("Seconds from recording start")
    fig.tight_layout(rect=[0, 0.02, 1, .94])
    fig.savefig(output, dpi=165, bbox_inches="tight")
    return fig


def draw_overlay(t, signals, zoom, output, channel):
    mask = (t >= zoom[0]) & (t <= zoom[1])
    fig, ax = plt.subplots(figsize=(15, 5.5))
    keys = ["raw", "butter_5", "butter_15", "butter_30", "savgol_75ms", "median25_butter15"]
    keys = [k for k in keys if k in signals]
    for key in keys:
        kw = dict(lw=1.5, alpha=.9)
        if key == "raw":
            kw = dict(lw=.9, color="#777777", alpha=.5)
        ax.plot(t[mask], signals[key][mask], label=key, **kw)
    ax.set(title=f"Filter overlay — {channel}", xlabel="Time (s)", ylabel="Measured units")
    ax.grid(alpha=.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=165)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Batch compare IMU signal filters, without changing raw recordings")
    parser.add_argument("--input", required=True, type=Path, help="Input FPA reader left_raw.csv/right_raw.csv")
    parser.add_argument("--out", type=Path, help="Output directory (default: beside CSV, <name>_filter_comparison)")
    parser.add_argument("--channel", choices=list(CHANNELS), default="gx", help="Which channel to compare")
    parser.add_argument("--zoom", nargs=2, type=float, metavar=("START", "END"), help="Time window in seconds")
    parser.add_argument("--export", help="Optional filter name to save processed gyro/accel axes in a separate CSV")
    parser.add_argument("--show", action="store_true", help="Open Matplotlib windows after generating files")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"File not found: {args.input}")
    df = pd.read_csv(args.input)
    t, fs = choose_time(df)
    if not (20 <= fs <= 2000):
        raise ValueError(f"Implausible sample rate {fs:.2f} Hz; check timestamps")
    zoom = tuple(args.zoom) if args.zoom else (max(0., .15 * t[-1]), min(t[-1], .80 * t[-1]))
    if not (0 <= zoom[0] < zoom[1] <= t[-1] + 1e-7):
        parser.error(f"Invalid --zoom window for recording of {t[-1]:.2f}s")
    out = args.out or args.input.with_name(args.input.stem + "_filter_comparison")
    out.mkdir(parents=True, exist_ok=True)
    signals = filtered_channel(df, args.channel, fs)
    rows = metrics(t, signals, zoom, fs)
    pd.DataFrame(rows).to_csv(out / "filter_comparison_metrics.csv", index=False)
    figs = [draw_grid(t, signals, zoom, out / "filter_gallery.png", args.channel),
            draw_overlay(t, signals, zoom, out / "filter_overlay.png", args.channel)]
    # Save wide-format signal overview for the currently chosen channel.
    pd.DataFrame({"time_s": t, **signals}).to_csv(out / f"{args.channel}_all_filters.csv", index=False)
    result = {"source": str(args.input), "channel": args.channel, "rows": len(df),
              "estimated_fs_hz": fs, "duration_s": float(t[-1]),
              "gap_p95_ms": float(1000 * np.percentile(np.diff(t), 95)),
              "zoom_s": zoom, "filter_variants": list(signals),
              "note": "Comparison scores indicate changes from raw, not scientific FPA accuracy."
             }
    (out / "comparison_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.export:
        if args.export not in signals:
            parser.error(f"Unknown --export {args.export!r}. Choose one of: {', '.join(signals)}")
        processed = df.copy(deep=True)
        for col in SIGNALS:
            filt = variants(df[col].to_numpy(dtype=float), fs)
            processed[col + "_processed"] = filt[args.export]
        # Original columns are preserved exactly; processed signals get suffixes.
        path = out / f"processed_{args.export}.csv"
        processed.to_csv(path, index=False)
        print(f"Processed signal export: {path}")
    print(f"Loaded {len(df)} rows at ~{fs:.2f} Hz, {t[-1]:.2f}s duration")
    print(f"Compared {len(signals)} variants for {args.channel} in {zoom[0]:.2f}–{zoom[1]:.2f}s")
    print(f"Results: {out}")
    print("Use filter_gallery.png to compare each method and filter_overlay.png to inspect peaks.")
    if args.show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
