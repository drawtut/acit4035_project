"""Plot trial data from saved fpa_reader CSV files and save them to the plots directory.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
TAG = "20261028_183828"
RECORDINGS_DIR = (
    Path(__file__).resolve().parent / "recordings"
)  # Or your absolute path
PLOTS_DIR = Path(
    r"C:\Users\nokkv\Downloads\acit4035_project-main\acit4035_project-main\plots"
)
STANCE_MIN_SECONDS = 0.12


def stance_intervals(df):
  """Extract contiguous stance phases from stance_candidate flags."""
  if df.empty:
    return []
  times = df["time_s"].values
  flags = df["stance_candidate"].values.astype(bool)
  edges = np.diff(np.r_[False, flags, False].astype(int))
  starts = np.where(edges == 1)[0]
  ends = np.where(edges == -1)[0]
  results = []
  for start, end in zip(starts, ends):
    if end <= len(times):
      t0, t1 = times[start], times[end - 1]
      if t1 - t0 >= STANCE_MIN_SECONDS:
        results.append((float(t0), float(t1)))
  return results


def plot_and_save_trial(tag):
  # Ensure plots directory exists
  PLOTS_DIR.mkdir(parents=True, exist_ok=True)

  left_path = RECORDINGS_DIR / f"{tag}_left.csv"
  right_path = RECORDINGS_DIR / f"{tag}_right.csv"

  if not left_path.exists() or not right_path.exists():
    print(f"Error: Could not find CSV files in: {RECORDINGS_DIR}")
    print(f"Looking for: {left_path.name} and {right_path.name}")
    return

  left_df = pd.read_csv(left_path)
  right_df = pd.read_csv(right_path)
  recordings = {"left": left_df, "right": right_df}

  # --- Figure 1: Multi-panel Sensor Data ---
  fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
  fig.suptitle(f"Foot-mounted IMU trial {tag} | Sensor Streams")
  labels = [
      (["ax_mps2", "ay_mps2", "az_mps2"], "Acceleration (m/s²)"),
      (["gx_dps", "gy_dps", "gz_dps"], "Angular velocity (°/s)"),
      (["mx_raw", "my_raw", "mz_raw"], "Magnetometer (raw counts)"),
      (["roll_deg", "pitch_deg", "yaw_deg"], "Euler angles (°)"),
  ]

  for side in ("left", "right"):
    df = recordings[side]
    if df.empty:
      continue
    t = df["time_s"].values
    for ax, (cols, ylabel) in zip(axes, labels):
      for column in cols:
        ax.plot(
            t,
            df[column],
            label=f"{side[0].upper()} {column}",
            lw=1.1,
            alpha=0.8,
        )
      ax.set_ylabel(ylabel)
      ax.grid(alpha=0.25)
    for a, b in stance_intervals(df):
      axes[3].axvspan(
          a,
          b,
          color=("tab:blue" if side == "left" else "tab:orange"),
          alpha=0.09,
      )

  for ax in axes:
    ax.legend(loc="upper right", ncol=3, fontsize=7)
  axes[-1].set_xlabel("Time since recording started (s)")
  fig.tight_layout()

  # Save Figure 1
  fig1_path = PLOTS_DIR / f"{tag}_sensors.png"
  fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
  print(f"Saved: {fig1_path}")

  # --- Figure 2: Reference-Relative Foot Heading ---
  fig2, ax = plt.subplots(figsize=(11, 4))
  for side, color in [("left", "tab:blue"), ("right", "tab:orange")]:
    df = recordings[side]
    if df.empty:
      continue
    ax.plot(
        df["time_s"],
        df["ref_heading_deg"],
        label=f"{side.title()} reference-relative foot heading",
        color=color,
    )
    for a, b in stance_intervals(df):
      ax.axvspan(a, b, color=color, alpha=0.08)

  ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
  ax.set(
      title=(
          "Provisional straight-path foot heading (recreated from CSV files)"
      ),
      xlabel="Time (s)",
      ylabel="Heading change from calibration (°)",
  )
  ax.grid(alpha=0.3)
  ax.legend()
  fig2.tight_layout()

  # Save Figure 2
  fig2_path = PLOTS_DIR / f"{tag}_heading.png"
  fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
  print(f"Saved: {fig2_path}")

  plt.show()


if __name__ == "__main__":
  plot_and_save_trial(TAG)