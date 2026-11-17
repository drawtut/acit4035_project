"""ACIT4035 experimental combined footstep candidate detector.

Uses shared.run() filtered az impact peaks and stationary mask plus
fpa_segmentation_v21.detect() gyro-supported swing boundaries.

One candidate contact per swing, ONLY if an impact is temporally associated
with a subsequent stable stance. No magnetometer or heading is used here.
These are *candidate* footsteps, not validated anatomical heel strikes.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "min_stance_duration_s": 0.10,
    "stance_check_window_s": 0.35,
    "min_stationary_fraction": 0.40,
    "impact_before_stance_s": 0.35,
    # Only for the FIRST valid swing after a >=1s aligned initial stance.
    # Fixed first-step recovery, distinct from the frozen global timing gate.
    "first_swing_impact_before_stance_s": 0.50,
    "impact_after_stance_s": 0.18,
    "min_swing_fraction_before_impact": 0.40,
    "min_impact_delay_after_swing_s": 0.10,
    # ONLY the first foot movement after prolonged initial standing may
    # qualify for a cautious, explicitly provisional edge-step recovery.
    "startup_recovery_enabled": False,
    "startup_min_initial_stance_s": 1.0,
    "startup_min_swing_duration_s": 0.10,
    "startup_min_peak_gyro_dps": 70.0,
    "startup_min_excursion_deg": 15.0,
    "startup_next_strict_step_max_gap_s": 3.5,
}


def settings(result, overrides=None):
    params = dict(DEFAULTS)
    params.update(result.get("config", {}).get("combined_detection", {}))
    if overrides:
        params.update(overrides)
    if not (0 <= params["min_stationary_fraction"] <= 1):
        raise ValueError("min_stationary_fraction must be 0..1")
    return params


def detect(result, segmentation, overrides=None):
    """Returns diagnostics for every gyro-identified swing, including rejects.

    A new stance must follow a valid swing, and exactly one acceleration RMS
    peak must be selected near the resulting swing-to-stance transition.
    The peak timestamp is an acceleration response, not guaranteed the first
    instant of foot-ground contact.
    """
    params = settings(result, overrides)
    t = np.asarray(result["t"], dtype=float)
    stationary_mask = np.asarray(result["mask"], dtype=bool)
    gyro_mag = np.asarray(result["gyro_magnitude"], dtype=float)
    envelope = np.asarray(result["impact_envelope"], dtype=float)
    contact_candidates = result["contacts"]
    stands = {int(s["stance_id"]): s for s in segmentation["stands"]}
    swings = segmentation["swings"]
    out = []

    for swing in swings:
        stance = stands.get(int(swing["to_stance_id"]))
        reasons = []
        if not swing["valid_swing"]:
            reasons.append("gyro_movement_not_valid_swing")
        if stance is None:
            reasons.append("no_subsequent_stance")
        stance_start = float(stance["start_s"]) if stance else float(swing["end_s"])
        stance_end = float(stance["end_s"]) if stance else None
        if stance:
            if float(stance["duration_s"]) < params["min_stance_duration_s"]:
                reasons.append("stance_too_short")

        # Verify the ORIGINAL stationary mask, not just a consolidated
        # green interval (which may contain short gaps and merged fragments).
        stable_fraction = 0.0
        median_post_gyro = float("nan")
        if stance:
            check_end = min(
                stance_start + params["stance_check_window_s"],
                float(t[-1]),
                stance_end
            )
            check = (t >= stance_start) & (t <= check_end)
            if np.any(check):
                stable_fraction = float(np.mean(stationary_mask[check]))
                median_post_gyro = float(np.median(gyro_mag[check]))
            if stable_fraction < params["min_stationary_fraction"]:
                reasons.append("weak_post_impact_stationarity")

        # Candidate impacts: late swing / early stance only.
        swing_start = float(swing["start_s"])
        swing_end = float(swing["end_s"])
        duration = max(float(swing["duration_s"]), 0.0)
        # First-step timing recovery only: after a verified long initial stand,
        # the impact-response peak can precede the filtered stationary-mask
        # onset by >0.35 s. It must STILL have a valid swing and stable stance.
        # Keep the original window for every later swing. No synthetic steps.
        from_stance = stands.get(int(swing["from_stance_id"]))
        eligible_initial = bool(
            int(swing["swing_id"]) == 1
            and int(swing["from_stance_id"]) == 1
            and from_stance is not None
            and float(from_stance["start_s"]) <= float(t[0]) + 0.25
            and float(from_stance["duration_s"]) >= 1.0
            and swing["valid_swing"]
        )
        original_lookback = float(params["impact_before_stance_s"])
        extended_lookback = float(params.get("first_swing_impact_before_stance_s", original_lookback))
        lookback = max(original_lookback, extended_lookback) if eligible_initial else original_lookback
        earliest = max(
            swing_start + params["min_swing_fraction_before_impact"] * duration,
            swing_start + params["min_impact_delay_after_swing_s"],
            stance_start - lookback,
        )
        latest = stance_start + params["impact_after_stance_s"]
        eligible = [
            c for c in contact_candidates
            if earliest <= float(c["time_s"]) <= latest
        ]
        if not eligible:
            reasons.append("no_impact_near_swing_to_stance")

        selected = None
        if eligible:
            # Prioritize a contact near the stance transition, but allow a
            # stronger impact to win when multiple peaks occur there.
            max_prom = max(float(c["prominence_mps2"]) for c in eligible)
            def candidate_score(c):
                dt = abs(float(c["time_s"]) - stance_start)
                closeness = np.exp(-dt / 0.16)
                strength = float(c["prominence_mps2"]) / max(max_prom, 1e-6)
                return 0.60 * closeness + 0.40 * strength
            selected = max(eligible, key=candidate_score)

        status = "accepted_candidate" if not reasons else "needs_review"
        early_initial_recovery = bool(
            status == "accepted_candidate" and eligible_initial and selected is not None
            and float(selected["time_s"]) < stance_start - original_lookback
        )
        out.append({
            "swing_id": int(swing["swing_id"]),
            "from_stance_id": int(swing["from_stance_id"]),
            "to_stance_id": int(swing["to_stance_id"]),
            "status": status,
            "reason": ";".join(reasons) if reasons else (
                "swing_impact_stance_supported;FIRST_STEP_EXTENDED_TIMING_WINDOW"
                if early_initial_recovery else "swing_impact_stance_supported"),
            "contact_time_s": float(selected["time_s"]) if selected else None,
            "contact_index": int(selected["index"]) if selected else None,
            "impact_rms_mps2": float(selected["rms_mps2"]) if selected else None,
            "impact_prominence_mps2": float(selected["prominence_mps2"]) if selected else None,
            "impact_candidates_near_transition": len(eligible),
            "swing_start_s": swing_start,
            "swing_end_s": swing_end,
            "swing_peak_gyro_dps": float(swing["gyro_peak_dps"]),
            "swing_duration_s": duration,
            "swing_excursion_deg": float(swing.get("integrated_angular_motion_deg", 0.0)),
            "v21_rejection_details": str(swing.get("reason", "")),
            "stance_start_s": stance_start if stance else None,
            "stance_end_s": stance_end,
            "stationary_fraction": stable_fraction,
            "post_stance_median_gyro_dps": median_post_gyro,
        })

    # Tight, reviewable exception: V2.1 can reject the FIRST rapid step
    # because it is shorter than its global 250 ms / 35 degree requirements.
    # Repositioning after established walking is NEVER recovered here.
    if params["startup_recovery_enabled"] and swings and out:
        first=out[0]
        first_swing=swings[0]
        initial=stands.get(int(first_swing["from_stance_id"]))
        later_strict=[e for e in out[1:]
                      if e["status"]=="accepted_candidate" and e["contact_time_s"] is not None]
        next_step_exists=bool(first["contact_time_s"] is not None and any(
            0.45 <= e["contact_time_s"]-first["contact_time_s"]
            <= params["startup_next_strict_step_max_gap_s"] for e in later_strict))
        only_swing_rejected=first["reason"]=="gyro_movement_not_valid_swing"
        first_stance_long=bool(initial is not None and
                              initial["duration_s"]>=params["startup_min_initial_stance_s"])
        first_swing_plausible=(
            first["swing_duration_s"] >= params["startup_min_swing_duration_s"] and
            first["swing_peak_gyro_dps"] >= params["startup_min_peak_gyro_dps"] and
            first["swing_excursion_deg"] >= params["startup_min_excursion_deg"])
        not_terminal=not first_swing.get("terminal_stopping_candidate",False)
        if (only_swing_rejected and first_stance_long and first_swing_plausible
                and next_step_exists and not_terminal):
            first["status"]="recovered_startup_candidate"
            first["reason"]="PROVISIONAL_first_short_swing_with_impact_and_stance;" + first["v21_rejection_details"]
    return out


def fpa_for_event(event, fpa_rows, max_stance_start_difference_s=0.14):
    """Map an accepted contact to the FPA already computed for its new stance.

    Never invent an FPA when no matching stance heading exists.
    """
    if event["status"] not in ("accepted_candidate", "recovered_startup_candidate"):
        return None
    start = event["stance_start_s"]
    if start is None:
        return None
    eligible = [
        (abs(float(r["stance_start_s"]) - start), r)
        for r in fpa_rows
        if r.get("stance_start_s") is not None
    ]
    if not eligible:
        return None
    delta, row = min(eligible, key=lambda pair: pair[0])
    if delta > max_stance_start_difference_s:
        return None
    return row


def export_rows(detections, fpa_rows, side):
    """Flatten diagnostic results for CSV, including unaccepted swings."""
    out = []
    for event in detections:
        rec = dict(side=side, **event)
        fpa = fpa_for_event(event, fpa_rows)
        rec["fpa_deg"] = fpa.get("fpa_deg") if fpa else None
        rec["fpa_heading_source"] = fpa.get("heading_source") if fpa else None
        rec["fpa_quality"] = fpa.get("quality") if fpa else "FPA unavailable or unaccepted"
        out.append(rec)
    return out
