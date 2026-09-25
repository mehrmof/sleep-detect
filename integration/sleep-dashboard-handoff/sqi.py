#!/usr/bin/env python3
"""Sleep Quality Index.

``sqi-1.0.0`` is the default and is unchanged. ``sqi-1.1.0`` adds a
cycle-structure sub-score. Pass ``index_version="sqi-1.1.0"`` to score it.
Whether that sub-score enters C6 is ``CYCLE_IN_ARCHITECTURE``, set from the
reference-versus-prediction rule in section 14 of ``SQI_REPORT.md``.

The index is computed from a hypnogram alone: a stage sequence and epoch
onsets. It does not read labels beyond that sequence, and it does not read
subject metadata. Age is optional and affects only the secondary,
age-adjusted architecture score.

Weights were fixed before any validation and are not fit to outcomes:

    efficiency 0.25, continuity 0.20, duration 0.20,
    fragmentation 0.15, latency 0.10, architecture 0.10

The primary total is the weighted arithmetic mean, reported on 0–100.
The weighted geometric mean is always returned beside it. v1 does not
age-adjust. Passing ``age`` also returns an age-adjusted variant.
"""

from __future__ import annotations

import csv
import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

INDEX_VERSION = "sqi-1.0.0"
INDEX_VERSION_11 = "sqi-1.1.0"
# Section 14. On the 158 development nights the cycle sub-score's
# reference-versus-prediction Spearman was 0.640, above the 0.50 floor,
# so sqi-1.1.0 folds it into C6. Weights are unchanged. sqi-1.0.0 does
# not read this flag.
CYCLE_IN_ARCHITECTURE = True
# Feinberg and Floyd, Psychophysiology, 1979: NREM periods at least 15 min,
# REM periods at least 5 min. They waived the 5 min minimum for the first
# REM period; this index applies it to every REM period.
MIN_NREM_PERIOD_MIN = 15.0
MIN_REM_PERIOD_MIN = 5.0
# Wake or unscored time at least this long ends a period. Shorter breaks
# stay out of the cycle duration and do not split it. The 15 min value
# matches their NREM-interruption rule. Applying it to wake is a project
# assumption.
PERIOD_BREAK_MIN = 15.0

# Locked before validation. See SQI_REPORT.md for the citations.
WEIGHTS: dict[str, float] = {
    "efficiency": 0.25,
    "continuity": 0.20,
    "duration": 0.20,
    "fragmentation": 0.15,
    "latency": 0.10,
    "architecture": 0.10,
}

EPOCH_SEC_DEFAULT = 30.0
FOLLOW_SEC = 600.0
MIN_WINDOW_MIN = 180.0
MAX_UNKNOWN_FRACTION = 0.25
MIN_COVERAGE = 0.70
MIN_SPT_MIN = 60.0
# Ohayon et al., Sleep, 2004: SWS declines about 2 percentage points per decade.
# Age 25 anchors the Carskadon and Dement young-adult band. The anchor age is
# a project assumption; the rate is theirs.
N3_REFERENCE_AGE = 25.0
N3_PER_DECADE = 2.0

REASON_WINDOW = "window_too_short"
REASON_UNKNOWN = "unknown_fraction"
REASON_NO_SLEEP = "no_sleep"
REASON_COVERAGE = "spt_coverage_low"
REASON_ONSETS = "invalid_epoch_onsets"
PROBABILITY_TOLERANCE = 1.0e-6

STAGE_NAMES = {
    "W": 0.0,
    "WAKE": 0.0,
    "N1": 1.0,
    "N2": 2.0,
    "N3": 3.0,
    "REM": 4.0,
    "R": 4.0,
}
UNSCORED_NAMES = {"", "?", "M", "MT", "MOVEMENT", "UNKNOWN", "UNSCORED", "NAN"}
SLEEP_CODES = {1.0, 2.0, 3.0, 4.0}
SCORED_CODES = {0.0, 1.0, 2.0, 3.0, 4.0}
PROB_COLUMNS = ("prob_W", "prob_N1", "prob_N2", "prob_N3", "prob_REM")


def map_piecewise(value: float, knots: list[tuple[float, float]]) -> float:
    """Map ``value`` through a piecewise-linear knot sequence.

    Outside the first and last knots the function holds the end value.
    Flat knot spans return the right-hand value.

    Args:
        value: Raw measure, in the same units as the knot x positions.
        knots: ``(x, y)`` pairs sorted by increasing ``x``. Each ``y`` is
            a score in ``[0, 1]``.

    Returns:
        Interpolated score, clipped to ``[0, 1]``.

    Raises:
        ValueError: ``knots`` is empty.
    """
    if not knots:
        raise ValueError("knots must not be empty")
    if value <= knots[0][0]:
        score = knots[0][1]
    elif value >= knots[-1][0]:
        score = knots[-1][1]
    else:
        score = knots[-1][1]
        for left, right in pairwise(knots):
            if value <= right[0]:
                span = right[0] - left[0]
                if span == 0.0:
                    score = right[1]
                else:
                    frac = (value - left[0]) / span
                    score = left[1] + frac * (right[1] - left[1])
                break
    return float(np.clip(score, 0.0, 1.0))


def score_duration(tst_hours: float) -> float:
    """Map total sleep time onto ``[0, 1]``.

    Zero at and below 5 h, one from 7 h through 9 h, 0.7 at and above 10 h.
    The 5 h floor and the 7–9 h plateau follow the adult NSF bands
    (Hirshkowitz et al., Sleep Health, 2015). The 0.7 long-sleep floor is a
    project assumption.

    Args:
        tst_hours: Total sleep time in hours.

    Returns:
        Duration sub-score in ``[0, 1]``.
    """
    return map_piecewise(
        tst_hours,
        [(0.0, 0.0), (5.0, 0.0), (7.0, 1.0), (9.0, 1.0), (10.0, 0.7), (24.0, 0.7)],
    )


def score_efficiency(se: float) -> float:
    """Map sleep efficiency onto ``[0, 1]``.

    One at and above 85% (Schutte-Rodin et al., J Clin Sleep Med, 2008).
    Zero at and below 65%. The 65% floor is a project assumption.

    Args:
        se: Sleep efficiency as a fraction, ``TST / TIB``.

    Returns:
        Efficiency sub-score in ``[0, 1]``.
    """
    return map_piecewise(se, [(0.0, 0.0), (0.65, 0.0), (0.85, 1.0), (2.0, 1.0)])


def score_latency(sol_min: float) -> float:
    """Map sleep-onset latency onto ``[0, 1]``.

    One at and below 15 min, zero at and above 60 min. Thirty minutes, the
    usual complaint threshold (Lichstein et al., Behav Res Ther, 2003;
    Schutte-Rodin et al., 2008), lands at two-thirds. Both outer anchors are
    project assumptions around that cited threshold.

    Args:
        sol_min: Minutes from the analysis-window start to sleep onset.

    Returns:
        Latency sub-score in ``[0, 1]``.
    """
    return map_piecewise(sol_min, [(0.0, 1.0), (15.0, 1.0), (60.0, 0.0), (1.0e6, 0.0)])


def score_continuity(waso_min: float) -> float:
    """Map wake after sleep onset onto ``[0, 1]``.

    One at and below 20 min, zero at and above 60 min. The 31 min insomnia
    criterion (Lichstein et al., 2003) therefore scores about 0.73 rather than
    nearly 1. The 20 min and 60 min anchors are project assumptions placed
    around that criterion.

    Args:
        waso_min: Wake minutes inside the sleep period.

    Returns:
        Continuity sub-score in ``[0, 1]``.
    """
    return map_piecewise(waso_min, [(0.0, 1.0), (20.0, 1.0), (60.0, 0.0), (1.0e6, 0.0)])


def score_fragmentation(sfi: float) -> float:
    """Map the stage-transition rate onto ``[0, 1]``.

    One at and below 10 transitions per hour of the sleep period, zero at and
    above 30. Laffan et al. (Sleep, 2010) report a 3-state median near 10 per
    hour of sleep; their quartiles are not 5-stage cut-points. Both numbers
    are project assumptions.

    Args:
        sfi: Stage transitions per hour of the sleep period.

    Returns:
        Fragmentation sub-score in ``[0, 1]``.
    """
    return map_piecewise(sfi, [(0.0, 1.0), (10.0, 1.0), (30.0, 0.0), (1.0e6, 0.0)])


def score_percent_band(
    percent: float,
    low: float,
    high: float,
    taper_end: float,
) -> float:
    """Map a stage percentage onto a plateau with an upper taper.

    Below ``low`` the score rises from 0 at 0%. It stays at 1 from ``low`` to
    ``high``, then falls to 0.5 at ``taper_end`` and holds 0.5. If ``low`` is
    not positive, the plateau includes 0%. The hold at 0.5, rather than a fall
    to 0, is a project assumption: extra N3 or REM is not treated as absent
    sleep.

    Args:
        percent: Stage time as a percentage of total sleep time.
        low: Left edge of the full-credit plateau.
        high: Right edge of the full-credit plateau.
        taper_end: Percentage at which the score reaches 0.5.

    Returns:
        Sub-score in ``[0, 1]``.
    """
    if low <= 0.0:
        knots = [(0.0, 1.0), (max(high, 0.0), 1.0), (taper_end, 0.5), (100.0, 0.5)]
    else:
        knots = [
            (0.0, 0.0),
            (low, 1.0),
            (high, 1.0),
            (taper_end, 0.5),
            (100.0, 0.5),
        ]
    return map_piecewise(percent, knots)


def score_rem_latency(rem_latency_min: float | None) -> float:
    """Map REM latency onto ``[0, 1]``.

    One from 60 min through 120 min. Zero at 0 min and at or above 240 min.
    Absent REM is zero. The 60 min edge is the short-latency region associated
    with depression (Kupfer, Biol Psychiatry, 1976). The 120 min and 240 min
    edges are project assumptions. A first REM near 80–100 min is the usual
    adult value (Carskadon and Dement, 2011).

    Args:
        rem_latency_min: Minutes from sleep onset to the first REM epoch, or
            ``None`` when no REM epoch falls inside the sleep period.

    Returns:
        REM-latency sub-score in ``[0, 1]``.
    """
    if rem_latency_min is None or math.isnan(rem_latency_min):
        return 0.0
    return map_piecewise(
        rem_latency_min,
        [(0.0, 0.0), (60.0, 1.0), (120.0, 1.0), (240.0, 0.0), (1.0e6, 0.0)],
    )


def score_cycle_count(n_cycles: float) -> float:
    """Map a completed-cycle count onto ``[0, 1]``.

    Zero at 0 cycles and at 1 cycle. Full credit from 4 through 6, which is
    the usual adult range (Carskadon and Dement, 2011). The score stays at 1
    through 7 and falls to 0 at 10. The rise from 1 to 4, the hold through 7,
    and the zero at 10 are project assumptions. "Tapering above 7" is read as
    a decline that starts after 7, so 7 itself is still full credit.

    Args:
        n_cycles: Number of completed NREM-REM cycles.

    Returns:
        Cycle-count sub-score in ``[0, 1]``.
    """
    return map_piecewise(
        n_cycles,
        [(0.0, 0.0), (1.0, 0.0), (4.0, 1.0), (6.0, 1.0), (7.0, 1.0), (10.0, 0.0), (1.0e6, 0.0)],
    )


def score_cycle_regularity(cv: float | None) -> float | None:
    """Map the coefficient of variation of cycle durations onto ``[0, 1]``.

    The score is ``1 - cv``, clipped to ``[0, 1]``. Fewer than two cycles
    leaves the coefficient undefined, and the term is omitted rather than
    scored 0. The formula is a project assumption. The coefficient uses the
    sample standard deviation.

    Args:
        cv: Sample coefficient of variation, or ``None`` when it is undefined.

    Returns:
        Regularity in ``[0, 1]``, or ``None`` when ``cv`` is missing.
    """
    if cv is None or math.isnan(cv):
        return None
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def score_n3_front(fraction: float | None) -> float | None:
    """Map the first-half share of N3 onto ``[0, 1]``.

    Full credit at or above 0.60. Zero at or below 0.35. Linear between.
    Both knots are project assumptions. The direction they reward, slow-wave
    sleep sitting early in the night, is Feinberg and Floyd (Psychophysiology,
    1979): the first NREM period normally holds about half of slow-wave sleep.
    A night with no N3 omits the term.

    Args:
        fraction: N3 minutes in the first half of the sleep period divided by
            all N3 minutes, or ``None`` when there is no N3.

    Returns:
        Front-loading sub-score in ``[0, 1]``, or ``None`` when omitted.
    """
    if fraction is None or math.isnan(fraction):
        return None
    return map_piecewise(fraction, [(0.0, 0.0), (0.35, 0.0), (0.60, 1.0), (1.0, 1.0)])


def n3_band_for_age(age: float | None) -> tuple[float, float, float]:
    """Return the N3 full-credit band and the upper taper point.

    Without age, the band is 13–23% of TST, the sum of the stage-3 and
    stage-4 ranges in Carskadon and Dement (2011), tapering to 0.5 at 40%.
    With age, both edges move by about 2 percentage points per decade away
    from age 25 (Ohayon et al., Sleep, 2004). The 40% taper moves with them.
    The reference age and the 40% taper are project assumptions.

    Args:
        age: Age in years, or ``None`` for the unadjusted band.

    Returns:
        ``(low, high, taper_end)`` in percentage points of TST.
    """
    low, high, taper_end = 13.0, 23.0, 40.0
    if age is None:
        return low, high, taper_end
    shift = -N3_PER_DECADE * (float(age) - N3_REFERENCE_AGE) / 10.0
    low += shift
    high += shift
    taper_end += shift
    low = max(low, 0.0)
    high = max(high, low)
    if taper_end <= high:
        taper_end = high + 17.0
    return low, high, taper_end


def aggregate(components: dict[str, float], weights: dict[str, float] | None = None) -> tuple[float, float]:
    """Combine six ``[0, 1]`` components on a 0–100 scale.

    The arithmetic mean lets a strong component offset a weak one. The
    geometric mean is zero when any positively weighted component is zero.

    Args:
        components: Keys ``duration``, ``efficiency``, ``latency``,
            ``continuity``, ``fragmentation``, and ``architecture``.
        weights: Optional override. The default is the prespecified set.

    Returns:
        ``(arithmetic, geometric)``, each on 0–100.
    """
    used = WEIGHTS if weights is None else weights
    names = (
        "efficiency",
        "continuity",
        "duration",
        "fragmentation",
        "latency",
        "architecture",
    )
    values = np.array([float(components[name]) for name in names], dtype=float)
    load = np.array([float(used[name]) for name in names], dtype=float)
    if not np.isclose(load.sum(), 1.0):
        load = load / load.sum()
    arithmetic = float(np.dot(load, values) * 100.0)
    if np.any((values <= 0.0) & (load > 0.0)):
        geometric = 0.0
    else:
        geometric = float(np.exp(np.dot(load, np.log(values))) * 100.0)
    return arithmetic, geometric


def score_stages(
    stages: np.ndarray,
    onset_sec: np.ndarray | None = None,
    *,
    epoch_sec: float | None = None,
    age: float | None = None,
    provenance: str = "unspecified",
    recording_id: str | None = None,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    index_version: str = INDEX_VERSION,
    include_cycle: bool | None = None,
) -> dict[str, object]:
    """Score one hypnogram.

    Unscored epochs (unknown, movement, non-finite) stay in the clock
    denominators. They are not sleep and they are not wake. They do not form
    stage transitions. A gap in ``onset_sec`` likewise stays in the span and
    is not filled with an invented stage.

    Sleep onset is the first sleep epoch followed by 10 minutes in which at
    least half the epochs are sleep and at least 20 epochs (on a 30 s grid)
    are present. Final awakening is the last sleep epoch under the mirrored
    rule. The sleep period runs from the onset epoch to the end of the final
    awakening epoch.

    Args:
        stages: Stage codes, one per epoch. Numeric codes are 0 wake, 1 N1,
            2 N2, 3 N3, 4 REM. Any other numeric value, NaN, or a name in
            ``{?, M, movement}`` is unscored. Names ``W, N1, N2, N3, REM``
            are accepted.
        onset_sec: Epoch start times in seconds. The default is a uniform
            grid of ``epoch_sec`` beginning at 0.
        epoch_sec: Epoch length when onsets are omitted. When onsets are
            present the length is their median positive step, unless this
            argument is set.
        age: Age in years. Used only for the secondary architecture score.
        provenance: ``reference``, ``prediction``, or another short label.
        recording_id: Optional identifier copied into the record.
        window_start_sec: Analysis-window start. The default is the first
            onset. An earlier value lengthens TIB and SOL without inventing
            epochs. This is how a lights-off denominator is supplied.
        window_end_sec: Analysis-window end. The default is the end of the
            last epoch.
        index_version: ``sqi-1.0.0`` or ``sqi-1.1.0``. The default is 1.0.0.
            Version 1.1.0 adds cycle fields. It changes C6 only when
            ``include_cycle`` says to.
        include_cycle: Whether the cycle sub-score enters C6. Used only by
            ``sqi-1.1.0``. ``None`` follows ``CYCLE_IN_ARCHITECTURE``.

    Returns:
        One flat record. Score fields are ``None`` when ``unavailable_reason``
        is set. Totals are on 0–100. Component scores are on ``[0, 1]``.

    Raises:
        ValueError: ``index_version`` is not 1.0.0 or 1.1.0.
    """
    if index_version not in (INDEX_VERSION, INDEX_VERSION_11):
        raise ValueError("index_version must be sqi-1.0.0 or sqi-1.1.0")
    use_cycle = index_version == INDEX_VERSION_11
    cycle_in_c6 = False
    if use_cycle:
        cycle_in_c6 = CYCLE_IN_ARCHITECTURE if include_cycle is None else include_cycle
    codes = _as_codes(stages)
    n_epochs = int(codes.size)
    step = _epoch_step(onset_sec, epoch_sec, n_epochs)
    onsets = _onsets(onset_sec, step, n_epochs)
    if n_epochs == 0:
        return _versioned(
            _blank(recording_id, provenance, age, REASON_WINDOW, step),
            index_version,
            cycle_in_c6,
        )

    # Epoch order is part of the hypnogram meaning. Reject malformed clocks;
    # sorting or deduplicating here would silently change that meaning.
    if not np.isfinite(onsets).all() or np.any(np.diff(onsets) <= 0.0):
        return _versioned(
            _blank(recording_id, provenance, age, REASON_ONSETS, step),
            index_version,
            cycle_in_c6,
        )

    start = float(onsets[0] if window_start_sec is None else window_start_sec)
    end = float(onsets[-1] + step if window_end_sec is None else window_end_sec)
    window_min = (end - start) / 60.0
    unscored = ~np.isin(codes, list(SCORED_CODES))
    unknown_fraction = float(unscored.mean()) if n_epochs else 1.0

    base = _versioned(
        _blank(recording_id, provenance, age, None, step),
        index_version,
        cycle_in_c6,
    )
    base["n_epochs"] = n_epochs
    base["n_unscored"] = int(unscored.sum())
    base["unknown_fraction"] = unknown_fraction
    base["window_min"] = window_min
    base["tib_min"] = window_min

    if window_min < MIN_WINDOW_MIN or end <= start:
        base["unavailable_reason"] = REASON_WINDOW
        return base
    if unknown_fraction > MAX_UNKNOWN_FRACTION:
        base["unavailable_reason"] = REASON_UNKNOWN
        return base

    onset_idx = _find_onset(codes, onsets, step)
    final_idx = _find_final(codes, onsets, step)
    if onset_idx is None or final_idx is None or final_idx < onset_idx:
        base["unavailable_reason"] = REASON_NO_SLEEP
        return base

    spt_start = float(onsets[onset_idx])
    spt_end = float(onsets[final_idx] + step)
    spt_min = (spt_end - spt_start) / 60.0
    inside = (onsets >= spt_start) & (onsets < spt_end)
    inside_codes = codes[inside]
    inside_onsets = onsets[inside]
    scored_inside = np.isin(inside_codes, list(SCORED_CODES))
    scored_min = float(scored_inside.sum() * step / 60.0)
    coverage = scored_min / spt_min if spt_min > 0.0 else 0.0
    base["coverage_fraction"] = coverage
    base["spt_min"] = spt_min

    if spt_min < MIN_SPT_MIN or coverage < MIN_COVERAGE:
        base["unavailable_reason"] = REASON_COVERAGE
        return base

    sleep = np.isin(inside_codes, list(SLEEP_CODES))
    wake = inside_codes == 0.0
    tst_min = float(sleep.sum() * step / 60.0)
    waso_min = float(wake.sum() * step / 60.0)
    tib_min = window_min
    se = tst_min / tib_min if tib_min > 0.0 else float("nan")
    sol_min = (spt_start - start) / 60.0
    transitions = _transitions(inside_codes, inside_onsets, step)
    wake_bouts = _wake_bouts(inside_codes, inside_onsets, step)
    spt_hours = spt_min / 60.0
    sfi = transitions / spt_hours if spt_hours > 0.0 else float("nan")
    awakening_index = wake_bouts / spt_hours if spt_hours > 0.0 else float("nan")

    if tst_min <= 0.0:
        base["unavailable_reason"] = REASON_NO_SLEEP
        return base

    pct = {
        "n1": 100.0 * float(((inside_codes == 1.0).sum()) * step / 60.0) / tst_min,
        "n2": 100.0 * float(((inside_codes == 2.0).sum()) * step / 60.0) / tst_min,
        "n3": 100.0 * float(((inside_codes == 3.0).sum()) * step / 60.0) / tst_min,
        "rem": 100.0 * float(((inside_codes == 4.0).sum()) * step / 60.0) / tst_min,
    }
    rem_latency = _rem_latency(inside_codes, inside_onsets, spt_start)
    n3_score = score_percent_band(pct["n3"], *n3_band_for_age(None))
    rem_score = _score_rem_percent(pct["rem"])
    rem_lat_score = score_rem_latency(rem_latency)
    architecture_terms = [n3_score, rem_score, rem_lat_score]
    cycle_payload = None
    if use_cycle:
        cycle_payload = _cycle_payload(inside_codes, inside_onsets, step, spt_start, spt_end)
        if cycle_in_c6:
            architecture_terms.append(float(cycle_payload["c_cycles"]))
    architecture = float(np.mean(architecture_terms))

    components = {
        "duration": score_duration(tst_min / 60.0),
        "efficiency": score_efficiency(se),
        "latency": score_latency(sol_min),
        "continuity": score_continuity(waso_min),
        "fragmentation": score_fragmentation(sfi),
        "architecture": architecture,
    }
    arithmetic, geometric = aggregate(components)

    age_architecture = None
    age_n3 = None
    age_arithmetic = None
    age_geometric = None
    if age is not None:
        age_n3 = score_percent_band(pct["n3"], *n3_band_for_age(float(age)))
        age_terms = [age_n3, rem_score, rem_lat_score]
        if cycle_in_c6 and cycle_payload is not None:
            age_terms.append(float(cycle_payload["c_cycles"]))
        age_architecture = float(np.mean(age_terms))
        age_components = dict(components)
        age_components["architecture"] = age_architecture
        age_arithmetic, age_geometric = aggregate(age_components)

    base.update(
        {
            "unavailable_reason": None,
            "sqi_arithmetic": arithmetic,
            "sqi_geometric": geometric,
            "sqi_arithmetic_age_adj": age_arithmetic,
            "sqi_geometric_age_adj": age_geometric,
            "c_duration": components["duration"],
            "c_efficiency": components["efficiency"],
            "c_latency": components["latency"],
            "c_continuity": components["continuity"],
            "c_fragmentation": components["fragmentation"],
            "c_architecture": architecture,
            "c_architecture_age_adj": age_architecture,
            "c_n3": n3_score,
            "c_rem": rem_score,
            "c_rem_latency": rem_lat_score,
            "c_n3_age_adj": age_n3,
            "tst_min": tst_min,
            "se": se,
            "sol_min": sol_min,
            "waso_min": waso_min,
            "sfi_per_h": sfi,
            "awakening_index": awakening_index,
            "spt_min": spt_min,
            "tib_min": tib_min,
            "pct_n1": pct["n1"],
            "pct_n2": pct["n2"],
            "pct_n3": pct["n3"],
            "pct_rem": pct["rem"],
            "rem_latency_min": rem_latency,
            "n_stage_transitions": transitions,
            "n_wake_bouts": wake_bouts,
            "coverage_fraction": coverage,
        }
    )
    if cycle_payload is not None:
        base.update(cycle_payload)
    return base


def score_prediction_csv(path: str | Path) -> list[dict[str, object]]:
    """Score a ``predict.py`` CSV, or any CSV with the same columns.

    Rows are grouped by ``recording_id`` when that column exists. A row with
    a blank stage and blank probabilities is unscored. Probabilities that are
    all missing stay unscored; they are not given a stage by ``argmax``.
    When no hard stage is present, all five probabilities must be finite,
    nonnegative, no greater than 1 (within ``1e-6``), and sum to 1 within
    an absolute tolerance of ``1e-6``. Grossly malformed rows are unscored;
    they are never silently renormalized.

    Args:
        path: CSV path. Required columns are ``onset_sec`` and either a stage
            column (``pred``, ``label``, or ``stage``) or ``prob_W`` through
            ``prob_REM``.

    Returns:
        One record per recording, in file order. Provenance is ``prediction``.
    """
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [score_stages(np.array([]), provenance="prediction")]
    groups: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for row in rows:
        key = row.get("recording_id") or ""
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    records = []
    for key in order:
        group = groups[key]
        stages, onsets = _rows_to_arrays(group)
        records.append(
            score_stages(
                stages,
                onsets,
                provenance="prediction",
                recording_id=key or None,
            )
        )
    return records


def _score_rem_percent(percent: float) -> float:
    """Map REM percent of TST, with a flat floor through 5%.

    Full credit is 20–25% (Carskadon and Dement, 2011). The score is 0 at and
    below 5%, rises to 1 at 20%, holds through 25%, and falls to 0.5 at 40%.
    The 5% floor and the 40% taper are project assumptions.

    Args:
        percent: REM minutes as a percentage of total sleep time.

    Returns:
        Sub-score in ``[0, 1]``.
    """
    return map_piecewise(
        percent,
        [
            (0.0, 0.0),
            (5.0, 0.0),
            (20.0, 1.0),
            (25.0, 1.0),
            (40.0, 0.5),
            (100.0, 0.5),
        ],
    )


def _as_codes(stages: np.ndarray) -> np.ndarray:
    """Convert a stage sequence to float codes, with NaN unscored.

    Args:
        stages: Numeric codes or stage names.

    Returns:
        Float array. Unscored epochs are NaN.
    """
    array = np.asarray(stages)
    if array.dtype.kind in "UO":
        return np.array([_parse_name(item) for item in array], dtype=float)
    values = array.astype(float)
    return values


def _parse_name(item: object) -> float:
    """Parse one stage name.

    Args:
        item: Cell value from a hypnogram.

    Returns:
        Stage code, or NaN when the cell is unscored.
    """
    if item is None:
        return float("nan")
    text = str(item).strip().upper()
    if text in UNSCORED_NAMES:
        return float("nan")
    if text in STAGE_NAMES:
        return STAGE_NAMES[text]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _epoch_step(
    onset_sec: np.ndarray | None,
    epoch_sec: float | None,
    n_epochs: int,
) -> float:
    """Resolve the epoch length in seconds.

    Args:
        onset_sec: Optional onset vector.
        epoch_sec: Optional caller override.
        n_epochs: Number of epochs.

    Returns:
        Epoch length in seconds.
    """
    if epoch_sec is not None:
        return float(epoch_sec)
    if onset_sec is not None and n_epochs >= 2:
        diffs = np.diff(np.asarray(onset_sec, dtype=float))
        positive = diffs[diffs > 0.0]
        if positive.size:
            return float(np.median(positive))
    return EPOCH_SEC_DEFAULT


def _onsets(onset_sec: np.ndarray | None, step: float, n_epochs: int) -> np.ndarray:
    """Return epoch onsets, synthesizing a uniform grid when needed.

    Args:
        onset_sec: Optional onset vector.
        step: Epoch length in seconds.
        n_epochs: Number of epochs.

    Returns:
        Onset of each epoch, in seconds.

    Raises:
        ValueError: ``onset_sec`` does not have one time per stage.
    """
    if n_epochs == 0:
        return np.array([], dtype=float)
    if onset_sec is None:
        return np.arange(n_epochs, dtype=float) * step
    onsets = np.asarray(onset_sec, dtype=float)
    if onsets.shape[0] != n_epochs:
        raise ValueError("onset_sec length does not match stages")
    return onsets


def _find_onset(codes: np.ndarray, onsets: np.ndarray, step: float) -> int | None:
    """Find the sleep-onset epoch index.

    Args:
        codes: Stage codes, NaN where unscored.
        onsets: Epoch start times in seconds.
        step: Epoch length in seconds. Used only to set the minimum count.

    Returns:
        Index of sleep onset, or ``None`` when no epoch qualifies.
    """
    min_count = round(FOLLOW_SEC / step)
    sleep = np.isin(codes, list(SLEEP_CODES))
    for idx in range(codes.size):
        if not sleep[idx]:
            continue
        window = (onsets > onsets[idx]) & (onsets <= onsets[idx] + FOLLOW_SEC)
        if int(window.sum()) < min_count:
            continue
        if float(sleep[window].mean()) >= 0.5:
            return idx
    return None


def _find_final(codes: np.ndarray, onsets: np.ndarray, step: float) -> int | None:
    """Find the final-awakening epoch index.

    Args:
        codes: Stage codes, NaN where unscored.
        onsets: Epoch start times in seconds.
        step: Epoch length in seconds.

    Returns:
        Index of the final awakening epoch, or ``None``.
    """
    min_count = round(FOLLOW_SEC / step)
    sleep = np.isin(codes, list(SLEEP_CODES))
    for idx in range(codes.size - 1, -1, -1):
        if not sleep[idx]:
            continue
        window = (onsets >= onsets[idx] - FOLLOW_SEC) & (onsets < onsets[idx])
        if int(window.sum()) < min_count:
            continue
        if float(sleep[window].mean()) >= 0.5:
            return idx
    return None


def _transitions(codes: np.ndarray, onsets: np.ndarray, step: float) -> int:
    """Count clock-adjacent stage changes inside the sleep period.

    An unscored epoch or a gap breaks the chain and is not itself a transition.

    Args:
        codes: Stage codes of epochs inside the sleep period.
        onsets: Their onset times.
        step: Epoch length in seconds.

    Returns:
        Number of stage transitions.
    """
    count = 0
    for left in range(codes.size - 1):
        gap = onsets[left + 1] - onsets[left]
        if abs(gap - step) > 0.5:
            continue
        a = codes[left]
        b = codes[left + 1]
        if a not in SCORED_CODES or b not in SCORED_CODES:
            continue
        if a != b:
            count += 1
    return count


def _wake_bouts(codes: np.ndarray, onsets: np.ndarray, step: float) -> int:
    """Count wake bouts of at least one epoch inside the sleep period.

    Args:
        codes: Stage codes of epochs inside the sleep period.
        onsets: Their onset times.
        step: Epoch length in seconds.

    Returns:
        Number of wake bouts.
    """
    bouts = 0
    in_bout = False
    previous_onset = None
    for code, onset in zip(codes, onsets):
        adjacent = previous_onset is not None and abs((onset - previous_onset) - step) <= 0.5
        if code == 0.0 and (in_bout and adjacent):
            pass
        elif code == 0.0:
            bouts += 1
            in_bout = True
        else:
            in_bout = False
        previous_onset = float(onset)
    return bouts


def _rem_latency(
    codes: np.ndarray,
    onsets: np.ndarray,
    spt_start: float,
) -> float | None:
    """Minutes from sleep onset to the first REM epoch in the sleep period.

    Args:
        codes: Stage codes inside the sleep period.
        onsets: Their onset times.
        spt_start: Sleep-onset time in seconds.

    Returns:
        REM latency in minutes, or ``None`` when the period contains no REM.
    """
    rem = np.where(codes == 4.0)[0]
    if rem.size == 0:
        return None
    return float((onsets[rem[0]] - spt_start) / 60.0)


def _versioned(
    record: dict[str, object],
    index_version: str,
    cycle_in_c6: bool,
) -> dict[str, object]:
    """Stamp the index version and, for 1.1.0, the empty cycle fields.

    Args:
        record: A score record, scored or refused.
        index_version: ``sqi-1.0.0`` or ``sqi-1.1.0``.
        cycle_in_c6: Whether this 1.1.0 record would fold cycles into C6.

    Returns:
        The same record.
    """
    record["index_version"] = index_version
    if index_version == INDEX_VERSION_11:
        for name in (
            "n_cycles",
            "cycle_duration_cv",
            "n3_front_fraction",
            "c_cycle_count",
            "c_cycle_regularity",
            "c_n3_front",
            "c_cycles",
        ):
            record[name] = None
        record["cycle_in_architecture"] = cycle_in_c6
    return record


def _cycle_payload(
    codes: np.ndarray,
    onsets: np.ndarray,
    step: float,
    spt_start: float,
    spt_end: float,
) -> dict[str, object]:
    """Cycle count, regularity, and N3 front-loading for one sleep period.

    A completed cycle is an NREM period of at least 15 minutes followed by a
    REM period of at least 5 minutes. NREM is N1, N2, or N3. A REM stretch
    shorter than 5 minutes does not end the NREM period. NREM shorter than
    15 minutes between REM stretches does not end the REM period; those
    minutes stay in the NREM total, which is Feinberg and Floyd's
    intercurrent-NREM rule. Wake or unscored time shorter than 15 minutes
    is skipped and is not part of the duration. A break at least that long
    ends the period. Applying the 15-minute break to wake, and requiring 5
    minutes of the first REM period, are project choices. Cycle duration is
    NREM minutes plus REM minutes.

    Args:
        codes: Stage codes inside the sleep period.
        onsets: Their onset times.
        step: Epoch length in seconds.
        spt_start: Sleep-onset time in seconds.
        spt_end: End of the final-awakening epoch, in seconds.

    Returns:
        Raw cycle metrics and the three mapped terms. Regularity is ``None``
        with fewer than two cycles. Front-loading is ``None`` when N3 is 0.
        ``c_cycles`` is the mean of the terms that were defined.
    """
    durations = _completed_cycle_durations(_stage_runs(codes, step))
    cv = _duration_cv(durations)
    front = _n3_front_fraction(codes, onsets, spt_start, spt_end)
    count_score = score_cycle_count(len(durations))
    regularity = score_cycle_regularity(cv)
    front_score = score_n3_front(front)
    terms = [count_score]
    if regularity is not None:
        terms.append(regularity)
    if front_score is not None:
        terms.append(front_score)
    return {
        "n_cycles": len(durations),
        "cycle_duration_cv": cv,
        "n3_front_fraction": front,
        "c_cycle_count": count_score,
        "c_cycle_regularity": regularity,
        "c_n3_front": front_score,
        "c_cycles": float(np.mean(terms)),
    }


def _stage_runs(codes: np.ndarray, step: float) -> list[tuple[str, float]]:
    """Collapse a sleep period into NREM, REM, and other runs.

    Args:
        codes: Stage codes. Non-finite values are other.
        step: Epoch length in seconds.

    Returns:
        ``(label, minutes)`` runs in order. Labels are ``nrem``, ``rem``,
        and ``other``.
    """
    runs: list[tuple[str, float]] = []
    kind: str | None = None
    count = 0
    for code in codes:
        label = _cycle_label(float(code))
        if label != kind:
            if kind is not None:
                runs.append((kind, count * step / 60.0))
            kind = label
            count = 0
        count += 1
    if kind is not None:
        runs.append((kind, count * step / 60.0))
    return runs


def _cycle_label(code: float) -> str:
    """Classify one epoch for cycle detection.

    Args:
        code: Stage code. Non-finite values are other.

    Returns:
        ``nrem``, ``rem``, or ``other``.
    """
    if code in (1.0, 2.0, 3.0):
        return "nrem"
    if code == 4.0:
        return "rem"
    return "other"


def _completed_cycle_durations(runs: list[tuple[str, float]]) -> list[float]:
    """Durations, in minutes, of NREM periods followed by REM periods.

    Args:
        runs: Output of :func:`_stage_runs`.

    Returns:
        One duration per completed cycle. Wake minutes are not included.
    """
    durations: list[float] = []
    index = 0
    limit = len(runs)
    while index < limit:
        if runs[index][0] != "nrem":
            index += 1
            continue
        nrem_min, cursor = _accumulate_nrem(runs, index)
        if cursor >= limit or runs[cursor][0] != "rem":
            index += 1
            continue
        nrem_min, rem_min, cursor = _accumulate_rem(runs, cursor, nrem_min)
        if nrem_min >= MIN_NREM_PERIOD_MIN and rem_min >= MIN_REM_PERIOD_MIN:
            durations.append(nrem_min + rem_min)
            index = cursor
        else:
            index += 1
    return durations


def _accumulate_nrem(runs: list[tuple[str, float]], start: int) -> tuple[float, int]:
    """Sum an NREM period until a qualifying REM period or a hard break.

    Args:
        runs: Stage runs.
        start: Index of the first NREM run.

    Returns:
        NREM minutes, and the index where the period stopped. That index
        points at a REM run when one qualified.
    """
    nrem_min = 0.0
    cursor = start
    while cursor < len(runs):
        label, minutes = runs[cursor]
        if label == "nrem":
            nrem_min += minutes
            cursor += 1
        elif (label == "rem" and minutes < MIN_REM_PERIOD_MIN) or (
            label == "other" and minutes < PERIOD_BREAK_MIN
        ):
            cursor += 1
        else:
            break
    return nrem_min, cursor


def _accumulate_rem(
    runs: list[tuple[str, float]],
    start: int,
    nrem_min: float,
) -> tuple[float, float, int]:
    """Sum a REM period, folding in NREM interruptions shorter than 15 min.

    Args:
        runs: Stage runs.
        start: Index of the first REM run.
        nrem_min: NREM minutes already accumulated for this cycle.

    Returns:
        Updated NREM minutes, REM minutes, and the index just after the period.
    """
    rem_min = 0.0
    cursor = start
    while cursor < len(runs):
        label, minutes = runs[cursor]
        if label == "rem":
            rem_min += minutes
            cursor += 1
        elif label == "nrem" and minutes < MIN_NREM_PERIOD_MIN:
            nrem_min += minutes
            cursor += 1
        elif label == "other" and minutes < PERIOD_BREAK_MIN:
            cursor += 1
        else:
            break
    return nrem_min, rem_min, cursor


def _duration_cv(durations: list[float]) -> float | None:
    """Sample coefficient of variation of cycle durations.

    Args:
        durations: Completed-cycle durations in minutes.

    Returns:
        Standard deviation divided by the mean, or ``None`` with fewer than
        two cycles.
    """
    if len(durations) < 2:
        return None
    values = np.asarray(durations, dtype=float)
    mean = float(values.mean())
    if mean <= 0.0:
        return None
    return float(values.std(ddof=1) / mean)


def _n3_front_fraction(
    codes: np.ndarray,
    onsets: np.ndarray,
    spt_start: float,
    spt_end: float,
) -> float | None:
    """Share of N3 epochs whose onset falls in the first half of SPT.

    Args:
        codes: Stage codes inside the sleep period.
        onsets: Their onset times.
        spt_start: Sleep-onset time in seconds.
        spt_end: End of the final-awakening epoch, in seconds.

    Returns:
        The fraction, or ``None`` when the period contains no N3. An epoch
        is in the first half when its onset is before the midpoint.
    """
    n3 = codes == 3.0
    total = int(n3.sum())
    if total == 0:
        return None
    midpoint = spt_start + 0.5 * (spt_end - spt_start)
    return float((n3 & (onsets < midpoint)).sum() / total)


def _blank(
    recording_id: str | None,
    provenance: str,
    age: float | None,
    reason: str | None,
    step: float,
) -> dict[str, object]:
    """Return a record with every score field empty.

    Args:
        recording_id: Optional identifier.
        provenance: Source of the stages.
        age: Age in years, copied through and not used for v1.
        reason: Machine-readable unavailability code, or ``None``.
        step: Epoch length in seconds. Stored only so callers can audit it.

    Returns:
        Flat record. Numeric score fields are ``None``.
    """
    fields = [
        "sqi_arithmetic",
        "sqi_geometric",
        "sqi_arithmetic_age_adj",
        "sqi_geometric_age_adj",
        "c_duration",
        "c_efficiency",
        "c_latency",
        "c_continuity",
        "c_fragmentation",
        "c_architecture",
        "c_architecture_age_adj",
        "c_n3",
        "c_rem",
        "c_rem_latency",
        "c_n3_age_adj",
        "tst_min",
        "se",
        "sol_min",
        "waso_min",
        "sfi_per_h",
        "awakening_index",
        "spt_min",
        "tib_min",
        "pct_n1",
        "pct_n2",
        "pct_n3",
        "pct_rem",
        "rem_latency_min",
        "n_stage_transitions",
        "n_wake_bouts",
        "coverage_fraction",
        "unknown_fraction",
        "window_min",
        "n_epochs",
        "n_unscored",
    ]
    record: dict[str, object] = {name: None for name in fields}
    record["recording_id"] = recording_id
    record["provenance"] = provenance
    record["index_version"] = INDEX_VERSION
    record["unavailable_reason"] = reason
    record["age"] = age
    record["epoch_sec"] = step
    record["sol_assumes_lights_off"] = True
    record["tib_is_window_proxy"] = True
    return record


def _rows_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Read stages and onsets from one recording's CSV rows.

    Args:
        rows: Dict rows sharing one recording id.

    Returns:
        Stage codes and onset times.
    """
    stages = []
    onsets = []
    for row in rows:
        onsets.append(float(row["onset_sec"]))
        stage_text = _first_present(row, ("pred", "label", "stage"))
        if stage_text is not None and stage_text.strip() != "":
            stages.append(_parse_name(stage_text))
            continue
        probs = []
        blank = False
        for column in PROB_COLUMNS:
            cell = row.get(column, "")
            if cell is None or str(cell).strip() == "":
                blank = True
                break
            probs.append(float(cell))
        valid = (
            not blank
            and len(probs) == 5
            and np.isfinite(probs).all()
            and np.all(np.asarray(probs) >= 0.0)
            and np.all(np.asarray(probs) <= 1.0 + PROBABILITY_TOLERANCE)
            and np.isclose(
                np.sum(probs),
                1.0,
                rtol=0.0,
                atol=PROBABILITY_TOLERANCE,
            )
        )
        if not valid:
            stages.append(float("nan"))
        else:
            stages.append(float(int(np.argmax(probs))))
    return np.array(stages, dtype=float), np.array(onsets, dtype=float)


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    """Return the first named cell that exists in ``row``.

    Args:
        row: CSV row.
        names: Column names in priority order.

    Returns:
        The cell text, or ``None`` when none of the columns exist.
    """
    for name in names:
        if name in row:
            return row[name]
    return None


def main(argv: list[str] | None = None) -> None:
    """Score a prediction CSV and write one row per recording.

    Args:
        argv: ``[csv path, optional output path]``. Defaults to ``sys.argv``.

    Raises:
        SystemExit: No CSV path was given.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: sqi.py <predictions.csv> [out.csv]")
    records = score_prediction_csv(args[0])
    if len(args) > 1:
        _write_records(Path(args[1]), records)
    for record in records:
        reason = record["unavailable_reason"] or "ok"
        total = record["sqi_arithmetic"]
        shown = "NA" if total is None else f"{total:.2f}"
        print(
            f"{record['recording_id']} arithmetic={shown} "
            f"geometric={record['sqi_geometric']} reason={reason}",
            flush=True,
        )


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    """Write records to CSV.

    Args:
        path: Destination path.
        records: Records returned by :func:`score_stages`.
    """
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


if __name__ == "__main__":
    main()
