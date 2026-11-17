"""Phase 15: bilateral *paired-landing* heading differential, research-only.

No new gyroscope integration, calibration, contact detection, or FPA correction.
The two FPA values refer to distinct, consecutive landing instants. This is
NOT instantaneous relative orientation, shared drift cancellation, or a
validated clinical gait-symmetry measure.

For independently zeroed taped-reference headings, with positive toe-out:
    inter-foot divergence = wrap(left FPA + right FPA)
    toe-out asymmetry = wrap(left FPA - right FPA)

Pair only adjacent opposite-foot events within MAX_PAIR_GAP_S; never require
strict alternation, invent a step, or reuse a contact. A missing angle or a
large temporal separation prevents a pair. Weakest detector evidence wins.
"""
from __future__ import annotations
import math
from collections import Counter

MAX_PAIR_GAP_S = 2.0
CONFIDENCE_ORDER = {'A_CONFIRMED': 0, 'B_PROVISIONAL': 1, 'C_REVIEW_ONLY': 2}
PAIR_FIELDS = [
    'pair_index', 'first_foot', 'second_foot', 'left_contact_s',
    'right_contact_s', 'contact_separation_s', 'pair_midpoint_s',
    'left_fpa_deg', 'right_fpa_deg', 'interfoot_divergence_deg',
    'toe_out_asymmetry_deg', 'left_confidence', 'right_confidence',
    'pair_confidence', 'temporal_interpretation',
]


def wrap_deg(angle):
    """Signed shortest angular difference; +180 is represented as +180."""
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 and angle > 0 else wrapped


def _good_float(value):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _confidence(row):
    status = row.get('detector_confidence', 'A_CONFIRMED')
    return status if status in CONFIDENCE_ORDER else 'C_REVIEW_ONLY'


def pair_landings(source, max_pair_gap_s=MAX_PAIR_GAP_S):
    """Pair sequential opposing footfalls from Stage 4 or dashboard events.

    Accepted input forms:
      stage = {'left': {'rows': [{contact_time_s, fpa_deg, ...}]}, ...}
      events = {'left': [{time_s, fpa_deg, ...}], ...}
    Per-foot relative timestamps may differ by one initial Modbus poll; the
    2.0s threshold is intentionally much larger than this synchronization error.
    The original sampling timestamps are NOT modified.
    """
    if not (0 < float(max_pair_gap_s) <= 10):
        raise ValueError('max_pair_gap_s must be between 0 and 10 seconds')
    events=[]
    for side in ('left', 'right'):
        branch=source.get(side, [])
        records=branch.get('rows', []) if isinstance(branch, dict) else branch
        for row in records:
            t=_good_float(row.get('contact_time_s', row.get('time_s')))
            if t is None:
                continue
            events.append((t,side,row))
    events.sort(key=lambda value:(value[0],value[1]))
    result=[]
    i=0
    while i<len(events)-1:
        t0,s0,rec0=events[i]
        t1,s1,rec1=events[i+1]
        a0=_good_float(rec0.get('fpa_deg'))
        a1=_good_float(rec1.get('fpa_deg'))
        if (s0 == s1 or t1-t0>max_pair_gap_s or
                a0 is None or a1 is None):
            i+=1
            continue
        by_side={s0:(t0,a0,rec0),s1:(t1,a1,rec1)}
        lt,lf,lr=by_side['left']
        rt,rf,rr=by_side['right']
        ca,cb=_confidence(lr),_confidence(rr)
        confidence=max((ca,cb),key=lambda value:CONFIDENCE_ORDER[value])
        result.append({
            'pair_index':len(result)+1,
            'first_foot':s0,
            'second_foot':s1,
            'left_contact_s':lt,
            'right_contact_s':rt,
            'contact_separation_s':round(abs(rt-lt),6),
            'pair_midpoint_s':(t0+t1)/2.0,
            'left_fpa_deg':lf,
            'right_fpa_deg':rf,
            'interfoot_divergence_deg':wrap_deg(lf+rf),
            'toe_out_asymmetry_deg':wrap_deg(lf-rf),
            'left_confidence':ca,
            'right_confidence':cb,
            'pair_confidence':confidence,
            'temporal_interpretation':'sequential_landings_not_simultaneous_foot_orientation',
        })
        i+=2
    return result


def summarize_pairs(pairs):
    """Descriptive only: no diagnosis, no offset correction and no GT fitting."""
    counts=Counter(r['pair_confidence'] for r in pairs)
    confirmed=[r for r in pairs if r['pair_confidence']=='A_CONFIRMED']
    def aggregate(rows,key):
        values=[float(r[key]) for r in rows]
        return round(sum(values)/len(values),4) if values else None
    return {
        'method':'consecutive_opposite_foot_landings_on_fixed_taped_reference',
        'quantity_not_measured':'instantaneous_interfoot_orientation_or_absolute_path_heading',
        'max_pair_gap_s':MAX_PAIR_GAP_S,
        'pair_count':len(pairs),
        'confirmed_pair_count':counts['A_CONFIRMED'],
        'stance_supported_pair_count':counts['B_PROVISIONAL'],
        'review_only_pair_count':counts['C_REVIEW_ONLY'],
        'mean_interfoot_divergence_deg_all_candidates':aggregate(pairs,'interfoot_divergence_deg'),
        'mean_toe_out_asymmetry_deg_all_candidates':aggregate(pairs,'toe_out_asymmetry_deg'),
        'mean_interfoot_divergence_deg_confirmed_only':aggregate(confirmed,'interfoot_divergence_deg'),
        'mean_toe_out_asymmetry_deg_confirmed_only':aggregate(confirmed,'toe_out_asymmetry_deg'),
        'reference_line_required':True,
        'clinical_validation':False,
    }
