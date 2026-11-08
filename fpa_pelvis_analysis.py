"""Foot progression angle (FPA) from one foot IMU and one pelvis (waist) IMU.

Method (one FPA value per stride of the instrumented foot)
-----------------------------------------------------------
1. Heading of each sensor = direction of its forward axis projected onto the
   horizontal plane (forward axes are set in fpa_settings.py).
2. Calibration: at the start the subject stands still for a few seconds with
   the foot along the walking line and the pelvis facing forward. The
   foot-minus-pelvis heading in that pose is defined as FPA = 0.
3. Stance phases of the foot are found from its gyro and accelerometer.
   Foot heading for a step = median heading in the middle of the stance.
4. Walking direction for that step = pelvis heading averaged over one full
   stride centred on the stance. The pelvis twists about +/-5 deg every
   stride; a full-stride average cancels that.
5. FPA = foot heading - walking direction - calibration, positive = toe-out.
6. Drift: the subject stands still again at the end. Any change in the
   foot-minus-pelvis heading between the start and end poses is gyro drift
   (6-axis mode) and is removed linearly over the walk.
7. Strides where the pelvis turns more than TURN_LIMIT_DEG are flagged.

Run on a saved recording (in Spyder, or):
    python fpa_pelvis_analysis.py                 newest recording
    python fpa_pelvis_analysis.py 20261108_113000 a specific recording tag
    python fpa_pelvis_analysis.py TAG --use-settings
        re-analyse with the current fpa_settings.py instead of the settings
        saved with the recording (use this after fixing a wrong axis)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import fpa_settings as S

G = 9.80665
DEFAULT_FOLDER = Path(__file__).resolve().parent / "recordings"


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def runs(mask):
    """(start, end) index pairs of consecutive True values, end inclusive."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def heading_from_euler(euler_deg, forward_axis):
    """Heading (deg, wrapped, counter-clockwise positive seen from above) of a
    sensor's forward axis, and how horizontal that axis is (0-1).

    euler_deg: (N, 3) roll, pitch, yaw from the WT901, same XYZ convention as
    the rest of the project."""
    euler_deg = np.atleast_2d(np.asarray(euler_deg, dtype=float))
    axis = np.asarray(forward_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    v = Rotation.from_euler("xyz", euler_deg, degrees=True).apply(axis)
    v = np.atleast_2d(v)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0])), np.hypot(v[:, 0], v[:, 1])


def side_sign(side, fpa_sign=S.FPA_SIGN):
    """+1/-1 so that toe-out is positive. Toe-out turns the left foot
    counter-clockwise (positive heading) and the right foot clockwise."""
    side = side.lower()
    if side not in ("left", "right"):
        raise ValueError(f"FOOT_SIDE must be 'left' or 'right', got {side!r}")
    return fpa_sign * (1 if side == "left" else -1)


def _mag(df, prefix, unit):
    return np.sqrt(sum(df[f"{prefix}{a}_{unit}"] ** 2 for a in "xyz")).to_numpy()


def _clean(df):
    """Drop failed reads (every sensor value exactly 0) and repeated
    timestamps, and sort by time."""
    cols = ["ax_mps2", "ay_mps2", "az_mps2", "gx_dps", "gy_dps", "gz_dps"]
    df = df[~(df[cols] == 0).all(axis=1)]
    df = df.sort_values("time_s").drop_duplicates("time_s")
    return df.reset_index(drop=True)


def _window(seconds, fs):
    """Odd number of samples covering about `seconds` (at least 1)."""
    n = max(1, int(round(seconds * fs)))
    return n if n % 2 else n + 1


def stance_core(a, b):
    """Index range of the middle 50 % of a stance (a..b inclusive). Avoids the
    heel-strike and push-off ends; at least one sample."""
    n = b - a + 1
    lo = a + n // 4
    hi = b - n // 4
    return np.arange(lo, max(lo, hi) + 1)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(foot: pd.DataFrame, pelvis: pd.DataFrame, side: str,
            foot_axis=S.FOOT_FORWARD_AXIS, pelvis_axis=S.PELVIS_FORWARD_AXIS,
            fpa_sign=S.FPA_SIGN) -> dict:
    """Return {'samples': DataFrame, 'strides': DataFrame, 'info': dict,
    'warnings': list}."""
    warnings = []
    foot, pelvis = _clean(foot), _clean(pelvis)
    if len(foot) < 10 or len(pelvis) < 10:
        raise ValueError("Too few samples to analyse.")
    sign = side_sign(side, fpa_sign)

    t = foot.time_s.to_numpy()
    tp = pelvis.time_s.to_numpy()
    dt = max(float(np.median(np.diff(t))), 1e-3)
    fs = 1.0 / dt

    fh_w, f_horiz = heading_from_euler(foot[["roll_deg", "pitch_deg", "yaw_deg"]], foot_axis)
    ph_w, p_horiz = heading_from_euler(pelvis[["roll_deg", "pitch_deg", "yaw_deg"]], pelvis_axis)
    fh = np.degrees(np.unwrap(np.radians(fh_w)))
    ph = np.degrees(np.unwrap(np.radians(ph_w)))
    ph_on_t = np.interp(t, tp, ph)                 # pelvis heading at foot times
    rel = fh - ph_on_t                             # foot minus pelvis

    f_gyro = pd.Series(_mag(foot, "g", "dps"))
    p_gyro = pd.Series(np.interp(t, tp, _mag(pelvis, "g", "dps")))
    f_acc = pd.Series(_mag(foot, "a", "mps2"))

    # --- start / end still periods (both sensors still) ---
    w_still = _window(0.3, fs)
    still = ((f_gyro.rolling(w_still, center=True, min_periods=1).median() < S.STILL_GYRO_DPS)
             & (p_gyro.rolling(w_still, center=True, min_periods=1).median() < S.STILL_GYRO_DPS)
             ).to_numpy()
    still_runs = [(a, b) for a, b in runs(still) if t[b] - t[a] >= S.MIN_CAL_S]
    start_win = next((r for r in still_runs if t[r[0]] <= t[0] + 1.0), None)
    end_win = next((r for r in reversed(still_runs) if t[r[1]] >= t[-1] - 1.0), None)
    if end_win is not None and start_win is not None and end_win[0] <= start_win[1]:
        end_win = None                              # whole recording is one still period
    if start_win is None:
        n = max(3, int(round(1.0 / dt)))
        start_win = (0, min(n, len(t)) - 1)
        warnings.append("No still period found at the start: calibration used the "
                        "first second and may be unreliable. Stand still for "
                        f"{S.STILL_SECONDS:.0f} s before walking.")
    sa, sb = start_win
    start_ref = float(np.median(rel[sa:sb + 1]))
    t_ws = t[sb]                                    # walking starts after this
    t_sm = float(np.median(t[sa:sb + 1]))           # time the start reference represents

    if end_win is not None:
        ea, eb = end_win
        end_ref = float(np.median(rel[ea:eb + 1]))
        t_we = t[ea]                                # walking ends before this
        t_em = float(np.median(t[ea:eb + 1]))
        drift_total = end_ref - start_ref
    else:
        ea = eb = None
        end_ref = np.nan
        t_we = t[-1]
        t_em = t[-1]
        drift_total = 0.0
        warnings.append("No still period found at the end: drift was NOT corrected. "
                        f"Stand still for {S.STILL_SECONDS:.0f} s before stopping.")

    def drift(tt):
        """Linear drift between the middle of the start and end still periods."""
        if t_em <= t_sm:
            return np.zeros_like(np.asarray(tt, dtype=float))
        return drift_total * np.clip((np.asarray(tt) - t_sm) / (t_em - t_sm), 0, 1)

    # --- mounting check: forward axes must be roughly horizontal ---
    f_h0 = float(np.median(f_horiz[sa:sb + 1]))
    p_h0 = float(np.median(np.interp(t[sa:sb + 1], tp, p_horiz)))
    for name, val, setting in (("foot", f_h0, "FOOT_FORWARD_AXIS"),
                               ("pelvis", p_h0, "PELVIS_FORWARD_AXIS")):
        if val < S.MIN_HORIZONTAL:
            warnings.append(f"The {name} forward axis is nearly vertical "
                            f"(horizontal component {val:.2f}). {setting} in "
                            "fpa_settings.py is probably the wrong axis; heading "
                            "and FPA are unreliable.")

    # --- stance phases during the walk ---
    # a short median filter (~50 ms) stops heel-strike vibration at high sample
    # rates from splitting one stance into several; at 10 Hz it is 1 sample
    w_st = _window(0.05, fs)
    g_s = f_gyro.rolling(w_st, center=True, min_periods=1).median().to_numpy()
    a_s = f_acc.rolling(w_st, center=True, min_periods=1).median().to_numpy()
    stance_flag = (g_s < S.STANCE_GYRO_DPS) & (np.abs(a_s - G) < S.STANCE_ACCEL_TOL)
    stances = []
    for a, b in runs(stance_flag):
        dur = t[b] - t[a] + dt
        if not (S.MIN_STANCE_S <= dur <= S.MAX_STANCE_S):
            continue
        if t[a] <= t_ws or t[b] >= t_we:
            continue
        stances.append((a, b))

    rows = []
    if stances:
        mids = []
        for a, b in stances:
            core = stance_core(a, b)
            mids.append((float(t[core].mean()), float(np.median(fh[core])),
                         float(np.ptp(fh[core])), a, b))
        T = np.array([m[0] for m in mids])
        half = float(np.median(np.diff(T))) / 2 if len(T) > 1 else 0.55
        for k, (tk, fhk, spread, a, b) in enumerate(mids):
            lo = tk - min((tk - T[k - 1]) / 2, 1.5 * half) if k > 0 else tk - half
            hi = tk + min((T[k + 1] - tk) / 2, 1.5 * half) if k < len(T) - 1 else tk + half
            grid = np.linspace(lo, hi, 41)
            pg = np.interp(grid, tp, ph)
            pelvis_mean = float(pg.mean())
            turn = float(pg[-1] - pg[0])
            fpa = float(sign * wrap180(fhk - pelvis_mean - start_ref - drift(tk)))
            notes = []
            if abs(turn) > S.TURN_LIMIT_DEG:
                notes.append("turning")
            if spread > 25:
                notes.append("unstable stance heading")
            rows.append({"stride": k + 1, "time_s": tk,
                         "stance_start_s": t[a], "stance_end_s": t[b],
                         "fpa_deg": fpa, "foot_heading_deg": fhk - fh[sa:sb + 1].mean(),
                         "walk_direction_deg": pelvis_mean - ph_on_t[sa:sb + 1].mean(),
                         "pelvis_turn_deg": turn, "drift_removed_deg": float(drift(tk)),
                         "valid": not notes, "note": ", ".join(notes)})
    strides = pd.DataFrame(rows, columns=[
        "stride", "time_s", "stance_start_s", "stance_end_s", "fpa_deg",
        "foot_heading_deg", "walk_direction_deg", "pelvis_turn_deg",
        "drift_removed_deg", "valid", "note"])

    valid = strides[strides.valid] if len(strides) else strides
    if len(valid) < 3:
        warnings.append(f"Only {len(valid)} valid strides found. Walk at least "
                        "8-10 steps in a straight line.")
    if abs(drift_total) > 15:
        warnings.append(f"Large heading drift between start and end ({drift_total:+.0f} deg). "
                        "Check that the sensors are in 6-axis mode, that the end pose "
                        "matches the start pose, and keep trials short.")

    samples = pd.DataFrame({
        "time_s": t,
        "foot_heading_deg": fh - fh[sa:sb + 1].mean(),
        "pelvis_heading_deg": ph_on_t - ph_on_t[sa:sb + 1].mean(),
        "foot_vs_pelvis_deg": sign * wrap180(rel - start_ref - drift(t)),
        "drift_deg": drift(t),
        "stance": stance_flag,
        "still": still,
    })
    info = {
        "side": side, "sign": sign, "n_strides": int(len(strides)),
        "n_valid": int(len(valid)),
        "fpa_mean": float(valid.fpa_deg.mean()) if len(valid) else np.nan,
        "fpa_sd": float(valid.fpa_deg.std()) if len(valid) > 1 else np.nan,
        "fpa_median": float(valid.fpa_deg.median()) if len(valid) else np.nan,
        "drift_total_deg": float(drift_total),
        "start_still_s": (float(t[sa]), float(t[sb])),
        "end_still_s": (float(t[ea]), float(t[eb])) if ea is not None else None,
        "foot_axis_horizontal": f_h0, "pelvis_axis_horizontal": p_h0,
        "sample_rate_hz": 1 / dt,
    }
    return {"samples": samples, "strides": strides, "info": info, "warnings": warnings}


def summary_text(result) -> str:
    i = result["info"]
    if i["n_valid"]:
        sd = f" ± {i['fpa_sd']:.1f}" if not np.isnan(i["fpa_sd"]) else ""
        kind = "toe-out" if i["fpa_mean"] >= 0 else "toe-in"
        line = (f"{i['side'].capitalize()} foot FPA: {i['fpa_mean']:+.1f}°{sd} ({kind}), "
                f"{i['n_valid']} valid of {i['n_strides']} strides")
    else:
        line = f"{i['side'].capitalize()} foot: no valid strides"
    return line + f" | drift removed {i['drift_total_deg']:+.1f}°"


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(result, title=""):
    import matplotlib.pyplot as plt
    s, st, info = result["samples"], result["strides"], result["info"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    def shade(ax, legend=True):
        tt = s.time_s.to_numpy()
        for i, (a, b) in enumerate(runs(s.still.to_numpy())):
            ax.axvspan(tt[a], tt[b], color="#2ca02c", alpha=0.12, lw=0,
                       label="standing still" if (legend and i == 0) else None)
        for i, (a, b) in enumerate(zip(st.stance_start_s, st.stance_end_s)):
            ax.axvspan(a, b, color="grey", alpha=0.18, lw=0,
                       label="stance used" if (legend and i == 0) else None)

    ax = axes[0]
    shade(ax)
    ax.plot(s.time_s, s.foot_heading_deg, color="#1f77b4", lw=1.2, label="foot heading")
    ax.plot(s.time_s, s.pelvis_heading_deg, color="#ff7f0e", lw=1.2, label="pelvis heading")
    if len(st):
        ax.plot(st.time_s, st.walk_direction_deg, "o-", color="#d62728", ms=4, lw=1,
                label="walking direction (pelvis, stride mean)")
    ax.set_ylabel("deg")
    ax.set_title("Sensor headings, relative to the start pose", loc="left")

    ax = axes[1]
    shade(ax, legend=False)
    ax.plot(s.time_s, s.foot_vs_pelvis_deg, color="#7f7f7f", lw=1,
            label="foot vs pelvis (every sample; meaningful only in stance)")
    ax.plot(s.time_s, info["sign"] * s.drift_deg, color="#9467bd", ls="--", lw=1.2,
            label="drift removed")
    if len(st):
        ax.plot(st.time_s, st.fpa_deg, "o", color="#d62728", ms=5, label="FPA per stride")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("deg (toe-out +)")
    lim = max(30, float(np.nanpercentile(np.abs(st.fpa_deg), 95)) * 1.5) if len(st) else 30
    ax.set_ylim(-lim, lim)
    ax.set_title("Foot relative to pelvis, calibrated and drift-corrected", loc="left")

    ax = axes[2]
    if len(st):
        v, bad = st[st.valid], st[~st.valid]
        ax.plot(v.time_s, v.fpa_deg, "o-", color="#d62728", ms=6, lw=1.2, label="valid stride")
        if len(bad):
            ax.plot(bad.time_s, bad.fpa_deg, "x", color="k", ms=8, label="flagged (turning / unstable)")
        if info["n_valid"]:
            m, sd = info["fpa_mean"], info["fpa_sd"]
            ax.axhline(m, color="#d62728", ls="--", lw=1, label=f"mean {m:+.1f}°")
            if not np.isnan(sd):
                ax.axhspan(m - sd, m + sd, color="#d62728", alpha=0.1, lw=0, label=f"± SD {sd:.1f}°")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("FPA (deg, toe-out +)")
    ax.set_xlabel("time [s]")
    ax.set_title("Foot progression angle, one value per stride", loc="left")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
    head = summary_text(result)
    if result["warnings"]:
        head += f"\n⚠ {len(result['warnings'])} warning(s), see console / status"
    fig.suptitle((title + "\n" if title else "") + head, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def recording_paths(folder: Path, tag: str):
    return (folder / f"{tag}_foot.csv", folder / f"{tag}_pelvis.csv",
            folder / f"{tag}_meta.json")


def analyze_recording(folder: Path, tag: str, use_settings=False, show=False):
    foot_csv, pelvis_csv, meta_json = recording_paths(folder, tag)
    meta = {}
    if meta_json.exists() and not use_settings:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
    side = meta.get("foot_side", S.FOOT_SIDE)
    result = analyze(pd.read_csv(foot_csv), pd.read_csv(pelvis_csv), side,
                     foot_axis=meta.get("foot_forward_axis", S.FOOT_FORWARD_AXIS),
                     pelvis_axis=meta.get("pelvis_forward_axis", S.PELVIS_FORWARD_AXIS),
                     fpa_sign=meta.get("fpa_sign", S.FPA_SIGN))
    out_csv = folder / f"{tag}_fpa_strides.csv"
    out_png = folder / f"{tag}_fpa.png"
    result["strides"].to_csv(out_csv, index=False)
    import matplotlib.pyplot as plt
    fig = plot(result, title=f"Recording {tag}")
    fig.savefig(out_png, dpi=130)
    print(summary_text(result))
    for w in result["warnings"]:
        print("WARNING:", w)
    print("saved", out_csv)
    print("saved", out_png)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return result, out_csv, out_png


def newest_tag(folder: Path):
    files = sorted(folder.glob("*_foot.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No *_foot.csv recordings found in {folder}")
    return files[-1].name[: -len("_foot.csv")]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tag", nargs="?", help="recording tag, e.g. 20261108_113000 (default: newest)")
    p.add_argument("--folder", default=str(DEFAULT_FOLDER))
    p.add_argument("--use-settings", action="store_true",
                   help="use current fpa_settings.py instead of the saved recording settings")
    p.add_argument("--no-show", action="store_true", help="save the plot without opening it")
    args = p.parse_args()
    folder = Path(args.folder)
    tag = args.tag or newest_tag(folder)
    print(f"Analysing {tag} in {folder}")
    analyze_recording(folder, tag, use_settings=args.use_settings, show=not args.no_show)


if __name__ == "__main__":
    main()
