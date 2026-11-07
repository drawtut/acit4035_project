#!/usr/bin/env python3
"""
Compare moving-average, median and high-pass filters (and combinations) on
one walking recording from the left/right foot IMUs.

Run in Spyder (Run / F5) or:  python plot_filters.py [RECORDINGS_FOLDER]

Combinations (applied left to right, median first so spikes are removed
before they get smeared by the average):
    raw | MA | Median | HP | Median+MA | Median+HP | MA+HP | Median+MA+HP

Filters
  MA      centred moving average over MA_WINDOW samples (no time shift)
  Median  centred median over MEDIAN_WINDOW samples (removes 1-sample spikes)
  HP      Butterworth high-pass, zero-phase (filtfilt), HP_CUTOFF_HZ.
          Removes slow parts: the gravity/static offset in acceleration and
          the bias in angular rate.

Output (in plots/filters_<recording>/ next to the CSVs):
  <side>_acceleration.png, <side>_angular_rate.png
      one panel per combination, faint = raw, bold = filtered
  filter_metrics.csv  + printed table:
      noise_still   std while the foot is still (lower = smoother)
      peak_kept     % of the raw peak |value| that survives
      offset        mean value (HP should bring this to ~0)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- Settings ---------------------------------------------------------------
DEFAULT_FOLDER = r"C:\Users\nokkv\Downloads\acit4035_project-N-kkvi-Testing\acit4035_project-N-kkvi-Testing"
# Recordings to run, all with the same filter parameters below
RECORDINGS = ["20261028_174828", "20261028_175255",
              "20261028_180457", "20261028_183828"]
# Which signals to filter: any of "acceleration", "angular_rate", "yaw"
SIGNALS_TO_RUN = ["yaw"]

MA_WINDOW = 3          # samples (~0.33 s at ~9 Hz)
MEDIAN_WINDOW = 3      # samples
HP_CUTOFF_HZ = 0.3     # below the step rhythm (~0.8-1 Hz), above slow drift
HP_ORDER = 2
STILL_GYRO_DPS = 10.0  # |gyro| below this counts as "still" for noise metric
# ---------------------------------------------------------------------------

ALL_SIGNALS = {
    "acceleration": (["ax_mps2", "ay_mps2", "az_mps2"], "m/s²"),
    "angular_rate": (["gx_dps", "gy_dps", "gz_dps"], "°/s"),
    # yaw is unwrapped first (no +180/-180 jumps), otherwise averaging across
    # a wrap gives nonsense values
    "yaw": (["yaw_deg_unwrapped"], "°"),
}
SIGNALS = {k: ALL_SIGNALS[k] for k in SIGNALS_TO_RUN}
AXIS_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]
LABELS = {"yaw_deg_unwrapped": "yaw"}


def moving_average(x):
    return x.rolling(MA_WINDOW, center=True, min_periods=1).mean()


def median(x):
    return x.rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()


def highpass(x, fs):
    # filtfilt assumes evenly spaced samples; the recorder's spacing jitters
    # a little around ~0.11 s, which is acceptable for this comparison.
    b, a = butter(HP_ORDER, HP_CUTOFF_HZ / (fs / 2), btype="high")
    return pd.Series(filtfilt(b, a, x.to_numpy()), index=x.index)


COMBOS = {
    "raw": [],
    "MA": ["ma"],
    "Median": ["med"],
    "HP": ["hp"],
    "Median+MA": ["med", "ma"],
    "Median+HP": ["med", "hp"],
    "MA+HP": ["ma", "hp"],
    "Median+MA+HP": ["med", "ma", "hp"],
}


def apply(x, steps, fs):
    for s in steps:
        x = {"ma": moving_average, "med": median}[s](x) if s != "hp" else highpass(x, fs)
    return x


def find_files(folder, rec):
    hits = {}
    for side in ("left", "right"):
        found = [p for p in Path(folder).rglob(f"{rec}_{side}.csv")
                 if "plots" not in p.parts]
        if not found:
            return None
        hits[side] = found[0]
    return hits


def load(path):
    df = pd.read_csv(path)
    sensor = ["ax_mps2", "ay_mps2", "az_mps2", "gx_dps", "gy_dps", "gz_dps",
              "mx_raw", "my_raw", "mz_raw"]
    df = df[~(df[sensor] == 0).all(axis=1)].reset_index(drop=True)
    df["yaw_deg_unwrapped"] = np.degrees(np.unwrap(np.radians(df.yaw_deg)))
    df.attrs["fs"] = 1 / np.median(np.diff(df.time_s))
    return df


def still_mask(df):
    """No motion within 2 samples either side."""
    gyro = np.sqrt(df.gx_dps**2 + df.gy_dps**2 + df.gz_dps**2)
    return gyro.rolling(5, center=True, min_periods=1).max() < STILL_GYRO_DPS


def shade_still(ax, df):
    t = df.time_s.to_numpy()
    m = still_mask(df).to_numpy()
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    for i, (a, b) in enumerate(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1)):
        ax.axvspan(t[a], t[b], color="grey", alpha=0.15, lw=0,
                   label="foot still" if i == 0 else None)


def plot_signal(df, side, sig, outdir, rec):
    cols, unit = SIGNALS[sig]
    fs = df.attrs["fs"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
    for ax, (name, steps) in zip(axes.T.flat, COMBOS.items()):
        for c, color in zip(cols, AXIS_COLORS):
            if steps:
                ax.plot(df.time_s, df[c], color=color, lw=0.7, alpha=0.25)
            ax.plot(df.time_s, apply(df[c], steps, fs), color=color, lw=1.3,
                    label=LABELS.get(c, c.split("_")[0][-1]))
        ax.axhline(0, color="k", lw=0.5)
        if sig == "yaw":
            shade_still(ax, df)
        ax.set_title(name, loc="left", fontsize=11, weight="bold")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"{rec} | {side} foot | {sig.replace('_', ' ')}\n"
                 f"MA {MA_WINDOW} samples, median {MEDIAN_WINDOW} samples, "
                 f"high-pass {HP_CUTOFF_HZ} Hz (order {HP_ORDER}), fs ≈ {fs:.1f} Hz  |  "
                 f"faint = raw, bold = filtered", fontsize=12)
    fig.tight_layout()
    path = outdir / f"{side}_{sig}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def metrics(df, side):
    fs = df.attrs["fs"]
    # still = no motion within 2 samples either side, so smoothing that
    # spreads a swing into neighbouring samples is not counted as noise
    still = still_mask(df)
    rows = []
    for sig, (cols, unit) in SIGNALS.items():
        for name, steps in COMBOS.items():
            noise, peak, offset, shift, step = [], [], [], [], []
            for c in cols:
                raw = df[c]
                f = apply(raw, steps, fs)
                # noise: sample-to-sample jitter while still
                # (only between neighbouring samples that are both still)
                pair = still & still.shift(1, fill_value=False)
                noise.append(f.diff()[pair].std() / np.sqrt(2))
                if sig == "yaw":
                    # how far the filter moves the heading while the foot is
                    # on the ground (where heading is actually measured), and
                    # the largest jump between two samples (raw swing spikes)
                    shift.append((f - raw)[still].abs().mean())
                    step.append(f.diff().abs().max())
                else:
                    peak.append(100 * f.abs().max() / raw.abs().max())
                    offset.append(f.mean())
            row = {"side": side, "signal": sig, "combo": name,
                   "noise_still": np.mean(noise), "unit": unit}
            if sig == "yaw":
                row.update({"shift_while_still": np.mean(shift),
                            "max_step": np.mean(step)})
            else:
                row.update({"peak_kept_%": np.mean(peak),
                            "offset": np.mean(np.abs(offset))})
            rows.append(row)
    return rows


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    all_rows = []
    base = None
    for rec in RECORDINGS:
        files = find_files(folder, rec)
        if files is None:
            print(f"warning: {rec}_left.csv / _right.csv not found, skipped")
            continue
        base = files["left"].parent / "plots"
        outdir = base / f"filters_{rec}"
        outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        print(f"\n=== {rec} ===")
        for side, path in files.items():
            df = load(path)
            for sig in SIGNALS:
                print("saved", plot_signal(df, side, sig, outdir, rec))
            rows += [{"recording": rec, **r} for r in metrics(df, side)]
        pd.DataFrame(rows).to_csv(outdir / "filter_metrics.csv", index=False)
        all_rows += rows
    if not all_rows:
        raise SystemExit(f"None of the recordings were found in {folder}")

    table = pd.DataFrame(all_rows)
    table.to_csv(base / "filter_metrics_all.csv", index=False)
    print("\nsaved", base / "filter_metrics_all.csv")
    pd.set_option("display.width", 200)
    for col, title in (("peak_kept_%", "Peak kept (%)"), ("noise_still", "Noise while still"),
                       ("shift_while_still", "Yaw: mean shift while foot is still (deg)"),
                       ("max_step", "Yaw: largest jump between samples (deg)")):
        if col not in table or table[col].isna().all():
            continue
        print(f"\n{title}, mean of both feet and all axes:")
        print(table.pivot_table(index=["signal", "combo"], columns=["recording", "side"],
                                values=col, sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
