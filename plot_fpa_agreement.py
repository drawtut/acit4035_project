"""Scatter of IMU-estimated FPA against the footprint reference.

Reads the Data sheet of ACIT4035_FPA_validation.xlsx, so if you correct a
footprint value in the workbook, re-running this picks the change up.

HOW TO USE
  1. Put this file next to ACIT4035_FPA_validation.xlsx, or edit WORKBOOK below.
  2. Run it. The figure is written next to the script as fpa_agreement.png/.pdf.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------- EDIT THIS ---------------------------------
WORKBOOK = Path(__file__).resolve().parent / "ACIT4035_FPA_validation.xlsx"
OUTPUT   = Path(__file__).resolve().parent / "fpa_agreement"
COLOURS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]  # P1..P5
# ---------------------------------------------------------------------------


def load():
    d = pd.read_excel(WORKBOOK, sheet_name="Data")
    d = d[d["Footprint FPA (deg)"].notna()].copy()
    # Anonymise, preserving the order the participants appear in the sheet.
    names = list(dict.fromkeys(d["Participant"]))
    d["label"] = d["Participant"].map({n: f"P{i+1}" for i, n in enumerate(names)})
    return d


def main():
    d = load()
    x = d["Footprint FPA (deg)"].to_numpy(float)
    y = d["IMU FPA (deg)"].to_numpy(float)

    slope, intercept = np.polyfit(x, y, 1)
    rho = np.corrcoef(x, y)[0, 1]

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "xtick.major.width": 0.8, "ytick.major.width": 0.8})
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    lim = (np.floor(min(x.min(), y.min()) / 5) * 5 - 2,
           np.ceil(max(x.max(), y.max()) / 5) * 5 + 2)

    line = np.array(lim)
    ax.plot(line, slope * line + intercept, color="black", lw=1.0,
            zorder=3, label="Regression line")

    for i, label in enumerate(sorted(d["label"].unique())):
        k = d["label"] == label
        ax.scatter(x[k], y[k], s=18, color=COLOURS[i % len(COLOURS)],
                   edgecolors="none", alpha=0.85, zorder=2, label=label)

    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("FPA: Footprint reference (deg)")
    ax.set_ylabel("FPA: IMU-Measurements (deg)")

    sign = "+" if intercept >= 0 else "-"
    ax.text(0.97, 0.06,
            f"$\\rho$ = {rho:.2f}\ny = {slope:.2f}x {sign} {abs(intercept):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)

    ax.legend(frameon=False, fontsize=7.5, loc="upper left",
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.savefig(f"{OUTPUT}.png", dpi=400)
    fig.savefig(f"{OUTPUT}.pdf")
    print(f"n = {len(x)}   rho = {rho:.3f}   y = {slope:.3f}x {sign} {abs(intercept):.3f}")
    print("written:", f"{OUTPUT}.png")


if __name__ == "__main__":
    main()
