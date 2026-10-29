"""Create and save graphs from fpa_reader.py trial CSV files.

fpa_reader.py writes one pair of files per trial into ./recordings:
    <TAG>_left.csv  and  <TAG>_right.csv     (TAG = YYYYMMDD_HHMMSS)

This script reads those pairs and saves PNG graphs into PLOTS_DIR:
    <TAG>_left_sensors.png    accel / gyro / mag / Euler for the left foot
    <TAG>_right_sensors.png   same for the right foot
    <TAG>_heading.png         reference-relative heading over time, stance shaded
    <TAG>_stance_heading.png  median heading per stance phase + left/right box plot
    <TAG>_summary.csv         per-stance values used in the stance plot

Usage
    python imu_graph_generator.py                 # newest trial in recordings/
    python imu_graph_generator.py 20261028_183828 # one specific trial
    python imu_graph_generator.py --all           # every trial in recordings/
    python imu_graph_generator.py --show          # also open the plot windows

NOTE: ref_heading_deg is a provisional, reference-relative foot heading,
not validated FPA (same caveat as fpa_reader.py).
Requirements: numpy, pandas, matplotlib
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# --- CONFIGURATION -----------------------------------------------------------
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
PLOTS_DIR = Path(
    r"C:\Users\nokkv\Downloads\acit4035_project-main\acit4035_project-main\plots"
)
STANCE_MIN_SECONDS = 0.12   # same as fpa_reader.py
STANCE_MIN_SAMPLES = 2      # need at least 2 samples to call it a stance phase
DPI = 200
SIDE_COLORS = {"left": "#2563eb", "right": "#ea580c"}   # matches fpa_reader UI
AXIS_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]           # x / y / z
# -----------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"^(\d{8}_\d{6})_(left|right)\.csv$")
REQUIRED_COLUMNS = {
    "time_s", "ax_mps2", "ay_mps2", "az_mps2", "gx_dps", "gy_dps", "gz_dps",
    "mx_raw", "my_raw", "mz_raw", "roll_deg", "pitch_deg", "yaw_deg",
    "ref_heading_deg", "stance_candidate",
}
SENSOR_PANELS = [
    (["ax_mps2", "ay_mps2", "az_mps2"], ["ax", "ay", "az"], "Acceleration (m/s²)"),
    (["gx_dps", "gy_dps", "gz_dps"], ["gx", "gy", "gz"], "Angular velocity (°/s)"),
    (["mx_raw", "my_raw", "mz_raw"], ["mx", "my", "mz"], "Magnetometer (raw)"),
    (["roll_deg", "pitch_deg", "yaw_deg"], ["roll", "pitch", "yaw"], "Euler angles (°)"),
]


# --- Data loading ------------------------------------------------------------
def find_tags(recordings_dir: Path) -> list[str]:
    """Return tags that have BOTH a left and a right CSV, oldest first."""
    sides: dict[str, set[str]] = {}
    for path in recordings_dir.glob("*.csv"):
        match = TAG_PATTERN.match(path.name)
        if match:
            sides.setdefault(match.group(1), set()).add(match.group(2))
    return sorted(tag for tag, found in sides.items() if found == {"left", "right"})


def load_side(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  missing: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df.sort_values("time_s").reset_index(drop=True)


def stance_intervals(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Index ranges [start, end) of contiguous stance_candidate runs.

    Same rule as fpa_reader.py: the run must last >= STANCE_MIN_SECONDS.
    """
    if df.empty:
        return []
    times = df["time_s"].to_numpy()
    flags = df["stance_candidate"].to_numpy().astype(bool)
    edges = np.diff(np.r_[False, flags, False].astype(int))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(s, e) for s, e in zip(starts, ends)
            if e - s >= STANCE_MIN_SAMPLES
            and times[e - 1] - times[s] >= STANCE_MIN_SECONDS]


def stance_table(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """One row per stance phase with its median reference-relative heading."""
    rows = []
    for n, (s, e) in enumerate(stance_intervals(df), start=1):
        seg = df.iloc[s:e]
        heading = seg["ref_heading_deg"].dropna()
        rows.append({
            "side": side, "stance_no": n,
            "t_start_s": seg["time_s"].iloc[0], "t_end_s": seg["time_s"].iloc[-1],
            "duration_s": seg["time_s"].iloc[-1] - seg["time_s"].iloc[0],
            "n_samples": len(seg),
            "median_heading_deg": heading.median() if len(heading) else np.nan,
            "std_heading_deg": heading.std() if len(heading) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def shade_stance(ax, df, color, alpha=0.10):
    t = df["time_s"].to_numpy()
    for s, e in stance_intervals(df):
        ax.axvspan(t[s], t[e - 1], color=color, alpha=alpha, lw=0)


# --- Figures -----------------------------------------------------------------
def plot_sensors(df: pd.DataFrame, side: str, tag: str, out: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Trial {tag} | {side.upper()} foot | sensor streams "
                 f"(shaded = stance candidate)", fontsize=13)
    t = df["time_s"].to_numpy()
    for ax, (cols, names, ylabel) in zip(axes, SENSOR_PANELS):
        for col, name, color in zip(cols, names, AXIS_COLORS):
            ax.plot(t, df[col], label=name, color=color, lw=1.1)
        shade_stance(ax, df, SIDE_COLORS[side])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8, framealpha=0.8)
    axes[-1].set_xlabel("Time since recording started (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    return fig


def plot_heading(data: dict[str, pd.DataFrame], tag: str, out: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for side, df in data.items():
        if df.empty:
            continue
        color = SIDE_COLORS[side]
        ax.plot(df["time_s"], df["ref_heading_deg"], color=color, lw=1.4,
                label=f"{side.title()} foot")
        shade_stance(ax, df, color, alpha=0.08)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set(title=f"Trial {tag} | reference-relative foot heading "
                 "(provisional, NOT validated FPA)",
           xlabel="Time (s)", ylabel="Heading change from calibration (°)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    return fig


def plot_stance_heading(table: pd.DataFrame, tag: str, out: Path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(f"Trial {tag} | median heading per stance phase", fontsize=13)

    box_data, box_labels, box_colors = [], [], []
    for side in ("left", "right"):
        sub = table[table["side"] == side].dropna(subset=["median_heading_deg"])
        if sub.empty:
            continue
        color = SIDE_COLORS[side]
        mid_t = (sub["t_start_s"] + sub["t_end_s"]) / 2
        ax1.errorbar(mid_t, sub["median_heading_deg"], yerr=sub["std_heading_deg"],
                     fmt="o", color=color, ms=6, capsize=3, lw=1,
                     label=f"{side.title()} (n={len(sub)})")
        box_data.append(sub["median_heading_deg"].to_numpy())
        box_labels.append(f"{side.title()}\nmedian {sub['median_heading_deg'].median():+.1f}°")
        box_colors.append(color)

    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set(xlabel="Stance mid-time (s)", ylabel="Heading vs calibration (°)")
    ax1.grid(alpha=0.3)
    if box_data:
        ax1.legend(loc="best")
        bp = ax2.boxplot(box_data, patch_artist=True, widths=0.55,
                         showfliers=False)  # every point is drawn below
        ax2.set_xticks(range(1, len(box_labels) + 1), box_labels)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set(facecolor=color, alpha=0.35, edgecolor=color)
        for i, (vals, color) in enumerate(zip(box_data, box_colors), start=1):
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
            ax2.scatter(i + jitter, vals, color=color, s=18, zorder=3)
    else:
        ax2.text(0.5, 0.5, "No stance phases found", ha="center",
                 va="center", transform=ax2.transAxes)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("Per-stance median heading (°)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    return fig


# --- Driver ------------------------------------------------------------------
def process_trial(tag: str, recordings_dir: Path, plots_dir: Path) -> bool:
    print(f"\nTrial {tag}")
    data = {side: load_side(recordings_dir / f"{tag}_{side}.csv")
            for side in ("left", "right")}
    if all(df.empty for df in data.values()):
        print("  no data, skipped")
        return False

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for side, df in data.items():
        if not df.empty:
            saved.append(plots_dir / f"{tag}_{side}_sensors.png")
            plot_sensors(df, side, tag, saved[-1])

    saved.append(plots_dir / f"{tag}_heading.png")
    plot_heading(data, tag, saved[-1])

    table = pd.concat([stance_table(df, side) for side, df in data.items()
                       if not df.empty], ignore_index=True)
    saved.append(plots_dir / f"{tag}_stance_heading.png")
    plot_stance_heading(table, tag, saved[-1])
    saved.append(plots_dir / f"{tag}_summary.csv")
    table.round(3).to_csv(saved[-1], index=False)

    for side in ("left", "right"):
        sub = table[table["side"] == side] if not table.empty else table
        if len(sub):
            print(f"  {side:5s}: {len(sub)} stance phases, median heading "
                  f"{sub['median_heading_deg'].median():+.1f}° "
                  f"(range {sub['median_heading_deg'].min():+.1f} to "
                  f"{sub['median_heading_deg'].max():+.1f}°)")
    for path in saved:
        print(f"  saved {path}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag", nargs="?",
                        help="trial tag, e.g. 20261028_183828 (default: newest)")
    parser.add_argument("--all", action="store_true", help="process every trial")
    parser.add_argument("--show", action="store_true", help="open plot windows")
    parser.add_argument("--recordings-dir", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use("Agg")   # save only, no windows
    import matplotlib.pyplot as plt

    tags = find_tags(args.recordings_dir)
    if args.tag:
        tags = [args.tag]
    elif not tags:
        print(f"No <tag>_left.csv / <tag>_right.csv pairs in {args.recordings_dir}")
        return 1
    elif not args.all:
        tags = tags[-1:]

    ok = sum(process_trial(t, args.recordings_dir, args.plots_dir) for t in tags)
    print(f"\nDone: {ok}/{len(tags)} trial(s) -> {args.plots_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
