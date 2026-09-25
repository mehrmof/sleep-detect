#!/usr/bin/env python3
"""Build dashboard JSON by running predict.py and sqi.py.

The script never imports a model. Inference is whatever command
``config/model.json`` names. Every prediction column is read through
``csv_columns`` in that file. A null entry means the predictor did not
write that field, and this script fills a default. Swapping models is an
edit to that file and a re-run of this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

ROOT = Path(__file__).resolve().parents[1]
EPOCH_SECONDS = 30
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
HIGH_AMPLITUDE_UV = 500.0
ANN_TO_STAGE = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
WEIGHTS = [
    ("efficiency", "Efficiency", 0.25),
    ("continuity", "Continuity", 0.20),
    ("duration", "Duration", 0.20),
    ("fragmentation", "Fragmentation", 0.15),
    ("latency", "Latency", 0.10),
    ("architecture", "Architecture", 0.10),
]
RECORDING_RE = re.compile(r"^(SC|ST)\d(\d{2})(\d)")
CSV_FIELDS = (
    "epoch_index",
    "onset",
    "stage",
    "p_wake",
    "p_n1",
    "p_n2",
    "p_n3",
    "p_rem",
    "quality_flags",
    "model_version",
    "duration",
)
PROB_FIELDS = ("p_wake", "p_n1", "p_n2", "p_n3", "p_rem")
SQI_COLUMNS = (
    "epoch_idx",
    "onset_sec",
    "prob_W",
    "prob_N1",
    "prob_N2",
    "prob_N3",
    "prob_REM",
    "pred",
)


def load_config(path: Path) -> dict[str, object]:
    """Read the model adapter config.

    Args:
        path: Path to ``model.json``.

    Returns:
        The config object. ``csv_columns`` maps canonical fields to CSV
        headers. A null header means the field is not in the file.

    Raises:
        ValueError: A required key or CSV field is missing.
        TypeError: ``csv_columns`` is not an object.
    """
    payload = json.loads(path.read_text())
    required = (
        "name",
        "version",
        "predict_command",
        "bundle_path",
        "sha256",
        "csv_columns",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"model config missing {missing}")
    columns = payload["csv_columns"]
    if not isinstance(columns, dict):
        raise TypeError("csv_columns must be an object")
    absent = [key for key in CSV_FIELDS if key not in columns]
    if absent:
        raise ValueError(f"csv_columns missing {absent}")
    return payload


def column_map(config: dict[str, object]) -> dict[str, str | None]:
    """Return the canonical field to CSV header map.

    Args:
        config: Model adapter config.

    Returns:
        One entry per canonical field. ``None`` means synthesize a default.

    Raises:
        TypeError: ``csv_columns`` is not an object.
    """
    raw = config["csv_columns"]
    if not isinstance(raw, dict):
        raise TypeError("csv_columns must be an object")
    mapped: dict[str, str | None] = {}
    for key in CSV_FIELDS:
        value = raw[key]
        if value is None or str(value).strip() == "":
            mapped[key] = None
        else:
            mapped[key] = str(value)
    return mapped


def file_sha256(path: Path) -> str:
    """Hash a file with SHA-256.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _num(value: str | None) -> float | None:
    """Parse a CSV cell into a float.

    Args:
        value: Cell text.

    Returns:
        The number, or ``None`` when the cell is blank or not finite.
    """
    if value is None or str(value).strip() == "":
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _round(value: float | None, digits: int = 4) -> float | None:
    """Round a float for JSON, preserving nulls.

    Args:
        value: Number or ``None``.
        digits: Decimal places.

    Returns:
        Rounded value, or ``None``.
    """
    if value is None:
        return None
    return round(float(value), digits)


def _band(total: float | None) -> str | None:
    """Map an arithmetic SQI onto a presentation band.

    Args:
        total: Arithmetic index on 0–100, or ``None``.

    Returns:
        ``excellent``, ``good``, ``fair``, ``poor``, or ``None``.
    """
    if total is None:
        return None
    if total >= 85.0:
        return "excellent"
    if total >= 70.0:
        return "good"
    if total >= 50.0:
        return "fair"
    return "poor"


def _fmt_hours(minutes: float) -> str:
    """Format minutes as ``Xh Ym``.

    Args:
        minutes: Duration in minutes.

    Returns:
        A short clock string.
    """
    whole = round(minutes)
    return f"{whole // 60}h {whole % 60:02d}m"


def recording_identity(stem: str) -> tuple[str, str, int | None]:
    """Parse a Sleep-EDF stem into subject, cohort, and night.

    Args:
        stem: File stem such as ``SC4021E0`` or an uploaded name.

    Returns:
        Subject id, cohort, and night. Unknown names keep the stem and
        use cohort ``NA``.
    """
    match = RECORDING_RE.match(stem)
    if not match:
        return stem, "NA", None
    cohort, subject_num, night = match.group(1), match.group(2), match.group(3)
    return f"{cohort}{subject_num}", cohort, int(night)


def _run(command: list[str]) -> tuple[str, str, float]:
    """Run a command with the parent environment and return its output.

    Args:
        command: Argv, already expanded.

    Returns:
        Stdout, stderr, and wall seconds.

    Raises:
        RuntimeError: The process exited non-zero.
    """
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()
        raise RuntimeError(detail)
    return completed.stdout, completed.stderr, elapsed


def predict_command(config: dict[str, object], edf: Path, csv_path: Path) -> list[str]:
    """Expand the config command for one EDF.

    Args:
        config: Model config.
        edf: PSG path.
        csv_path: Where predictions should be written.

    Returns:
        Argv for the subprocess.

    Raises:
        TypeError: ``predict_command`` is not a string.
    """
    raw = config["predict_command"]
    if not isinstance(raw, str):
        raise TypeError("predict_command must be a string")
    parts = shlex.split(raw)
    mapping = {"edf": str(edf), "csv": str(csv_path)}
    argv = []
    for part in parts:
        text = part.format(**mapping)
        if text == "python":
            text = sys.executable
        argv.append(text)
    return argv


def run_predict(config: dict[str, object], edf: Path, csv_path: Path) -> float:
    """Run the configured predictor.

    Args:
        config: Model config.
        edf: PSG path.
        csv_path: Prediction CSV destination.

    Returns:
        Wall-clock seconds spent in the predictor.
    """
    _stdout, _stderr, elapsed = _run(predict_command(config, edf, csv_path))
    return elapsed


def run_sqi(csv_path: Path, out_csv: Path) -> dict[str, str]:
    """Run ``sqi.py`` on a prediction CSV.

    Args:
        csv_path: Prediction CSV from ``predict.py``.
        out_csv: One-row SQI CSV destination.

    Returns:
        The first SQI record as strings.
    """
    _run([sys.executable, str(ROOT / "sqi.py"), str(csv_path), str(out_csv)])
    with out_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"sqi.py wrote no rows for {csv_path.name}")
    return rows[0]


def _read_predictions(csv_path: Path) -> list[dict[str, str]]:
    """Load prediction rows.

    Args:
        csv_path: CSV written by ``predict.py``.

    Returns:
        Row dicts in file order.
    """
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _cell(row: dict[str, str], header: str | None) -> str | None:
    """Read one mapped CSV cell.

    Args:
        row: Prediction row.
        header: CSV header from ``csv_columns``, or ``None`` when absent.

    Returns:
        Stripped text, or ``None`` when the column was not provided or blank.
    """
    if header is None or header not in row:
        return None
    text = row[header]
    if text is None or str(text).strip() == "":
        return None
    return str(text).strip()


def _stage_name(code: str | None, probs: list[float] | None) -> str:
    """Resolve a stage name from a cell or from probabilities.

    Args:
        code: Stage cell. Names (``W``, ``N1``, ``N2``, ``N3``, ``REM``) and
            integer codes are both accepted. ``None`` means the column was
            not provided.
        probs: Five class probabilities, or ``None`` when those columns were
            not provided.

    Returns:
        One of ``W``, ``N1``, ``N2``, ``N3``, ``REM``. A missing stage uses
        ``argmax``. A missing stage and missing probabilities default to ``W``.
    """
    if code is not None:
        upper = code.upper()
        if upper == "WAKE":
            return "W"
        if upper in STAGE_ORDER:
            return upper
        return STAGE_ORDER[int(float(code))]
    if probs is not None:
        return STAGE_ORDER[int(np.argmax(probs))]
    return "W"


def _probability_row(
    row: dict[str, str],
    columns: dict[str, str | None],
) -> list[float] | None:
    """Read the five class probabilities through the column map.

    Args:
        row: Prediction row.
        columns: Canonical field to CSV header map.

    Returns:
        Five probabilities, or ``None`` when any probability column is null
        or blank.
    """
    headers = [columns[key] for key in PROB_FIELDS]
    if any(header is None for header in headers):
        return None
    values: list[float] = []
    for header in headers:
        cell = _cell(row, header)
        if cell is None:
            return None
        values.append(float(cell))
    return values


def _one_hot(stage: str) -> list[float]:
    """Build a default probability vector for one stage.

    Args:
        stage: Stage name.

    Returns:
        Five probabilities with ``1`` on ``stage`` and ``0`` elsewhere.
    """
    return [1.0 if name == stage else 0.0 for name in STAGE_ORDER]


def map_prediction_rows(
    rows: list[dict[str, str]],
    columns: dict[str, str | None],
) -> list[dict[str, object]]:
    """Translate predictor rows into canonical epoch records.

    Null map entries are filled here. Epoch index defaults to the row number.
    Onset defaults to ``epoch_index * duration``. Duration defaults to 30 s.
    A missing probability block becomes a one-hot of the stage. A missing
    stage becomes ``argmax`` of the probabilities, or ``W`` when both are
    absent. Quality flags and model version stay ``None`` so the caller can
    apply the EDF quality scan and the config version.

    Args:
        rows: CSV rows in file order.
        columns: Canonical field to CSV header map.

    Returns:
        One record per epoch, with keys ``epoch``, ``onset_sec``, ``stage``,
        ``p``, ``quality_flag``, ``model_version``, and ``duration_sec``.
    """
    mapped: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        duration_cell = _cell(row, columns["duration"])
        duration = EPOCH_SECONDS if duration_cell is None else float(duration_cell)
        probs = _probability_row(row, columns)
        stage = _stage_name(_cell(row, columns["stage"]), probs)
        if probs is None:
            probs = _one_hot(stage)
        epoch_cell = _cell(row, columns["epoch_index"])
        epoch = index if epoch_cell is None else int(float(epoch_cell))
        onset_cell = _cell(row, columns["onset"])
        onset = float(epoch) * duration if onset_cell is None else float(onset_cell)
        mapped.append(
            {
                "epoch": epoch,
                "onset_sec": onset,
                "stage": stage,
                "p": probs,
                "quality_flag": _cell(row, columns["quality_flags"]),
                "model_version": _cell(row, columns["model_version"]),
                "duration_sec": duration,
            }
        )
    return mapped


def write_sqi_csv(path: Path, epochs: list[dict[str, object]]) -> None:
    """Write the canonical CSV ``sqi.py`` scores.

    ``sqi.py`` does not know about model column names. This file always uses
    ``onset_sec``, ``pred``, and ``prob_W`` through ``prob_REM``.

    Args:
        path: Destination CSV.
        epochs: Canonical epoch records from :func:`map_prediction_rows`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SQI_COLUMNS))
        writer.writeheader()
        for epoch in epochs:
            stage = str(epoch["stage"])
            probs = epoch["p"]
            assert isinstance(probs, list)
            writer.writerow(
                {
                    "epoch_idx": epoch["epoch"],
                    "onset_sec": epoch["onset_sec"],
                    "prob_W": probs[0],
                    "prob_N1": probs[1],
                    "prob_N2": probs[2],
                    "prob_N3": probs[3],
                    "prob_REM": probs[4],
                    "pred": STAGE_ORDER.index(stage),
                }
            )


def _bundle_digest(config: dict[str, object]) -> str:
    """Return the configured bundle digest.

    An empty ``bundle_path`` means this checkout ships no weights. The hash
    check runs only when both the path and the expected digest are set.

    Args:
        config: Model adapter config.

    Returns:
        The digest stored on the recording.

    Raises:
        RuntimeError: The file hash does not match ``sha256``.
    """
    bundle = str(config.get("bundle_path") or "").strip()
    expected = str(config.get("sha256") or "").strip()
    if not bundle or not expected:
        return expected
    digest = file_sha256((ROOT / bundle).resolve())
    if digest != expected:
        raise RuntimeError(f"bundle sha256 {digest} does not match config/model.json")
    return digest


def _components(row: dict[str, str]) -> list[dict[str, object]]:
    """Build the six component rows from an SQI CSV record.

    Args:
        row: One ``sqi.py`` record.

    Returns:
        Component dicts in weight order. Empty when the index was refused.
    """
    if (row.get("unavailable_reason") or "").strip():
        return []
    se = _num(row.get("se"))
    waso = _num(row.get("waso_min"))
    tst = _num(row.get("tst_min"))
    sfi = _num(row.get("sfi_per_h"))
    sol = _num(row.get("sol_min"))
    architecture = _num(row.get("c_architecture"))
    raw_values = {
        "efficiency": se,
        "continuity": waso,
        "duration": tst,
        "fragmentation": sfi,
        "latency": sol,
        "architecture": architecture,
    }
    displays = {
        "efficiency": None if se is None else f"{se * 100:.0f}%",
        "continuity": None if waso is None else f"{waso:.0f} min",
        "duration": None if tst is None else _fmt_hours(tst),
        "fragmentation": None if sfi is None else f"{sfi:.1f}/h",
        "latency": None if sol is None else f"{sol:.0f} min",
        "architecture": None if architecture is None else f"{architecture:.2f}",
    }
    built = []
    for key, label, weight in WEIGHTS:
        score = _num(row.get(f"c_{key}"))
        contribution = None if score is None else weight * score
        built.append(
            {
                "name": label,
                "key": key,
                "weight": weight,
                "raw": _round(raw_values[key], 4),
                "raw_display": displays[key],
                "score": _round(score, 4),
                "contribution": _round(contribution, 4),
            }
        )
    return built


def _sleep_stats(row: dict[str, str]) -> dict[str, float | None]:
    """Copy sleep statistics from the SQI record.

    Args:
        row: One ``sqi.py`` record.

    Returns:
        The dashboard sleep-stat object. Values stay null when SQI refused.
    """
    se = _num(row.get("se"))
    return {
        "tst_min": _round(_num(row.get("tst_min")), 2),
        "spt_min": _round(_num(row.get("spt_min")), 2),
        "tib_min": _round(_num(row.get("tib_min")), 2),
        "waso_min": _round(_num(row.get("waso_min")), 2),
        "sol_min": _round(_num(row.get("sol_min")), 2),
        "se_pct": None if se is None else _round(se * 100.0, 2),
        "sfi_per_hour": _round(_num(row.get("sfi_per_h")), 3),
        "awakening_index": _round(_num(row.get("awakening_index")), 3),
        "rem_latency_min": _round(_num(row.get("rem_latency_min")), 2),
        "n3_pct": _round(_num(row.get("pct_n3")), 2),
        "rem_pct": _round(_num(row.get("pct_rem")), 2),
    }


def _analysis_window(stats: dict[str, float | None]) -> dict[str, float] | None:
    """Derive the SQI window edges from SOL and SPT.

    Args:
        stats: Sleep statistics copied from ``sqi.py``.

    Returns:
        Start and end in seconds from the file start, or ``None``.
    """
    sol = stats.get("sol_min")
    spt = stats.get("spt_min")
    if sol is None or spt is None:
        return None
    start = float(sol) * 60.0
    return {"start_sec": round(start, 3), "end_sec": round(start + float(spt) * 60.0, 3)}


def _sqi_block(row: dict[str, str]) -> dict[str, object]:
    """Map one SQI CSV row onto the dashboard SQI object.

    Args:
        row: One ``sqi.py`` record.

    Returns:
        The ``sqi`` object in the recording schema.
    """
    reason = (row.get("unavailable_reason") or "").strip() or None
    total = _num(row.get("sqi_arithmetic"))
    geometric = _num(row.get("sqi_geometric"))
    provenance = row.get("provenance") or "prediction"
    if provenance == "prediction":
        provenance = "predicted"
    return {
        "version": row.get("index_version") or "sqi-1.0.0",
        "total_arithmetic": _round(total, 2),
        "total_geometric": _round(geometric, 2),
        "band": _band(total),
        "components": _components(row),
        "coverage": _round(_num(row.get("coverage_fraction")), 4),
        "provenance": provenance,
        "tib_is_window_proxy": str(row.get("tib_is_window_proxy")).lower() == "true",
        "sol_assumes_lights_off": str(row.get("sol_assumes_lights_off")).lower() == "true",
        "unavailable_reason": reason,
    }


def _stage_summary(stages: list[str]) -> dict[str, dict[str, float]]:
    """Count predicted stages into minutes and percent of the file.

    Args:
        stages: Predicted stage name per epoch.

    Returns:
        One entry per stage, in clinical order.
    """
    total = max(len(stages), 1)
    summary = {}
    for name in STAGE_ORDER:
        count = sum(stage == name for stage in stages)
        minutes = count * EPOCH_SECONDS / 60.0
        summary[name] = {
            "minutes": round(minutes, 2),
            "pct": round(100.0 * count / total, 2),
        }
    return summary


def _quality_flags(edf: Path, n_epochs: int) -> tuple[list[str], dict[str, float]]:
    """Flag each epoch from the Fpz-Cz trace.

    A flat epoch matches the frozen ``is_constant_eeg`` rule (minimum equals
    maximum). High amplitude is a peak absolute value above 500 µV. This is
    display-only signal QC. It is not a model feature and it does not change
    the stages ``predict.py`` wrote.

    Args:
        edf: PSG path.
        n_epochs: Complete 30 s epochs already predicted.

    Returns:
        Per-epoch flags and the four quality percentages.
    """
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="error")
    channel = "EEG Fpz-Cz"
    if channel not in raw.ch_names:
        flags = ["missing"] * n_epochs
        return flags, {
            "coverage_pct": 0.0,
            "missing_pct": 100.0,
            "constant_eeg_pct": 0.0,
            "high_amplitude_pct": 0.0,
        }
    samples = int(EPOCH_SECONDS * float(raw.info["sfreq"]))
    data = np.asarray(raw.get_data(picks=[channel]), dtype=np.float64)[0] * 1e6
    flags: list[str] = []
    for index in range(n_epochs):
        epoch = data[index * samples : (index + 1) * samples]
        if epoch.size == 0 or not np.isfinite(epoch).all():
            flags.append("missing")
        elif float(np.min(epoch)) == float(np.max(epoch)):
            flags.append("constant")
        elif float(np.max(np.abs(epoch))) > HIGH_AMPLITUDE_UV:
            flags.append("high_amplitude")
        else:
            flags.append("good")
    count = max(n_epochs, 1)
    missing = flags.count("missing")
    return flags, {
        "coverage_pct": round(100.0 * (n_epochs - missing) / count, 2),
        "missing_pct": round(100.0 * missing / count, 2),
        "constant_eeg_pct": round(100.0 * flags.count("constant") / count, 2),
        "high_amplitude_pct": round(100.0 * flags.count("high_amplitude") / count, 2),
    }


def _hypnogram_path(edf: Path) -> Path | None:
    """Find a sibling hypnogram, if the directory has one.

    Args:
        edf: PSG path.

    Returns:
        Hypnogram path, or ``None`` when this file has no expert staging.
    """
    stem = edf.name.split("-")[0]
    if stem.lower().endswith(".edf"):
        stem = stem[:-4]
    prefix = stem[:6]
    matches = sorted(edf.parent.glob(f"{prefix}*Hypnogram.edf"))
    matches = [path for path in matches if path.resolve() != edf.resolve()]
    return matches[0] if matches else None


def _reference_stages(edf: Path, onsets: list[float]) -> list[str | None] | None:
    """Align expert annotations onto the predicted epoch grid.

    Args:
        edf: PSG path. A sibling ``*Hypnogram.edf`` is optional.
        onsets: Epoch start times in seconds from the PSG start.

    Returns:
        Stage names or nulls, one per epoch, or ``None`` when no hypnogram
        file exists.
    """
    path = _hypnogram_path(edf)
    if path is None:
        return None
    annotations = mne.read_annotations(str(path))
    events = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        strict=True,
    ):
        stage = ANN_TO_STAGE.get(str(description))
        events.append((float(onset), float(onset) + float(duration), stage))
    labels: list[str | None] = []
    for onset in onsets:
        midpoint = float(onset) + EPOCH_SECONDS / 2.0
        stage = None
        for start, stop, name in events:
            if start <= midpoint < stop:
                stage = name
                break
        labels.append(stage)
    return labels


def _scores(truth: list[str], pred: list[str]) -> dict[str, object]:
    """Compute agreement on epochs both sides scored.

    Args:
        truth: Reference stages.
        pred: Predicted stages, same length, already filtered.

    Returns:
        Macro-F1, Cohen's kappa, accuracy, per-class F1, and a 5×5 matrix.
        Rows are reference, columns are predicted, order W N1 N2 N3 REM.
    """
    index = {name: pos for pos, name in enumerate(STAGE_ORDER)}
    matrix = [[0 for _ in STAGE_ORDER] for _ in STAGE_ORDER]
    for true_stage, pred_stage in zip(truth, pred, strict=True):
        matrix[index[true_stage]][index[pred_stage]] += 1
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(5))
    accuracy = correct / total if total else 0.0
    f1_values = {}
    for name, row_i in index.items():
        support = sum(matrix[row_i])
        predicted = sum(matrix[r][row_i] for r in range(5))
        tp = matrix[row_i][row_i]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        if precision + recall == 0.0:
            f1_values[name] = 0.0
        else:
            f1_values[name] = 2.0 * precision * recall / (precision + recall)
    macro = sum(f1_values.values()) / 5.0
    observed = accuracy
    expected = 0.0
    if total:
        for pos in range(5):
            row_sum = sum(matrix[pos])
            col_sum = sum(matrix[r][pos] for r in range(5))
            expected += (row_sum / total) * (col_sum / total)
    kappa = 0.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "macro_f1": round(macro, 4),
        "kappa": round(kappa, 4),
        "accuracy": round(accuracy, 4),
        "per_class_f1": {name: round(f1_values[name], 4) for name in STAGE_ORDER},
        "confusion_matrix": matrix,
    }


def _header(edf: Path) -> tuple[str | None, float, list[str]]:
    """Read start time, sampling rate, and channel names.

    Args:
        edf: PSG path.

    Returns:
        ISO start time, sampling rate, and channel names.
    """
    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose="error")
    started = raw.info["meas_date"]
    text = started.isoformat() if started is not None else None
    return text, float(raw.info["sfreq"]), list(raw.ch_names)


def build_recording(
    edf: Path,
    out_dir: Path,
    config: dict[str, object],
) -> dict[str, object]:
    """Score one EDF and write its dashboard JSON.

    Args:
        edf: PSG path.
        out_dir: Dashboard data directory.
        config: Model adapter config.

    Returns:
        The recording object that was written.
    """
    edf = edf.resolve()
    stem = edf.name.replace("-PSG.edf", "").replace(".edf", "")
    subject_id, cohort, night = recording_identity(stem)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    pred_csv = work / f"{stem}_predictions.csv"
    sqi_csv = work / f"{stem}_sqi.csv"
    inference_seconds = run_predict(config, edf, pred_csv)
    columns = column_map(config)
    epochs = map_prediction_rows(_read_predictions(pred_csv), columns)
    canonical_csv = work / f"{stem}_canonical.csv"
    write_sqi_csv(canonical_csv, epochs)
    sqi_row = run_sqi(canonical_csv, sqi_csv)
    start_dt, sfreq, channels = _header(edf)
    flags, quality = _quality_flags(edf, len(epochs))
    use_csv_flags = columns["quality_flags"] is not None
    hypnogram = []
    stages = []
    onsets = []
    durations = []
    versions = []
    for epoch, flag in zip(epochs, flags, strict=True):
        stage = str(epoch["stage"])
        onset = float(epoch["onset_sec"])
        probs = epoch["p"]
        assert isinstance(probs, list)
        rounded = [_round(float(value), 4) for value in probs]
        csv_flag = epoch["quality_flag"]
        chosen = str(csv_flag) if use_csv_flags and csv_flag else flag
        stages.append(stage)
        onsets.append(onset)
        durations.append(float(epoch["duration_sec"]))
        if epoch["model_version"]:
            versions.append(str(epoch["model_version"]))
        hypnogram.append(
            {
                "epoch": int(epoch["epoch"]),
                "onset_sec": onset,
                "stage": stage,
                "p": rounded,
                "quality_flag": chosen,
            }
        )
    if use_csv_flags:
        chosen_flags = [str(item["quality_flag"]) for item in hypnogram]
        count = max(len(chosen_flags), 1)
        missing = chosen_flags.count("missing")
        quality = {
            "coverage_pct": round(100.0 * (len(chosen_flags) - missing) / count, 2),
            "missing_pct": round(100.0 * missing / count, 2),
            "constant_eeg_pct": round(100.0 * chosen_flags.count("constant") / count, 2),
            "high_amplitude_pct": round(
                100.0 * chosen_flags.count("high_amplitude") / count,
                2,
            ),
        }
    stats = _sleep_stats(sqi_row)
    reference_labels = _reference_stages(edf, onsets)
    reference: dict[str, object] | None = None
    if reference_labels is not None:
        paired_t = []
        paired_p = []
        for label, stage in zip(reference_labels, stages, strict=True):
            if label in STAGE_ORDER:
                paired_t.append(label)
                paired_p.append(stage)
        reference = _scores(paired_t, paired_p)
        reference["reference_hypnogram"] = reference_labels
        reference["n_compared"] = len(paired_t)
    duration_hours = sum(durations) / 3600.0
    digest = _bundle_digest(config)
    model_version = versions[0] if versions else str(config["version"])
    record: dict[str, object] = {
        "recording_id": stem,
        "subject_id": subject_id,
        "cohort": cohort,
        "night": night,
        "start_datetime": start_dt,
        "duration_hours": round(duration_hours, 3),
        "n_epochs": len(epochs),
        "epoch_seconds": EPOCH_SECONDS,
        "sampling_rate": sfreq,
        "channels": channels,
        "model": {
            "name": config["name"],
            "version": model_version,
            "sha256": digest,
            "inference_seconds": round(inference_seconds, 3),
            "device": "external_classifier_subprocess",
        },
        "hypnogram": hypnogram,
        "stage_summary": _stage_summary(stages),
        "sleep_stats": stats,
        "analysis_window": _analysis_window(stats),
        "sqi": _sqi_block(sqi_row),
        "quality": quality,
        "reference": reference,
    }
    dest = out_dir / "recordings" / f"{stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record))
    pred_csv.unlink(missing_ok=True)
    canonical_csv.unlink(missing_ok=True)
    sqi_csv.unlink(missing_ok=True)
    return record


def summary_row(record: dict[str, object]) -> dict[str, object]:
    """Keep the fields the recordings table needs.

    Args:
        record: Full recording object.

    Returns:
        A small index row.
    """
    sqi = record["sqi"]
    stats = record["sleep_stats"]
    assert isinstance(sqi, dict)
    assert isinstance(stats, dict)
    model = record["model"]
    assert isinstance(model, dict)
    return {
        "recording_id": record["recording_id"],
        "subject_id": record["subject_id"],
        "cohort": record["cohort"],
        "night": record["night"],
        "start_datetime": record["start_datetime"],
        "duration_hours": record["duration_hours"],
        "n_epochs": record["n_epochs"],
        "sqi": sqi.get("total_arithmetic"),
        "band": sqi.get("band"),
        "unavailable_reason": sqi.get("unavailable_reason"),
        "se_pct": stats.get("se_pct"),
        "has_reference": record["reference"] is not None,
        "inference_seconds": model.get("inference_seconds"),
        "status": "completed",
    }


def write_index(out_dir: Path, config: dict[str, object]) -> None:
    """Rewrite ``index.json`` from every recording file.

    Args:
        out_dir: Dashboard data directory.
        config: Model adapter config, copied into the index header.
    """
    folder = out_dir / "recordings"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text())
        rows.append(summary_row(record))
    payload = {
        "model": {
            "name": config["name"],
            "version": config["version"],
            "sha256": config["sha256"],
            "device": "external_classifier_subprocess",
        },
        "recordings": rows,
    }
    (out_dir / "index.json").write_text(json.dumps(payload, indent=2))


def iter_edfs(edf_dir: Path | None, explicit: list[str]) -> list[Path]:
    """Collect PSG paths, skipping hypnogram files.

    Args:
        edf_dir: Optional directory of EDFs.
        explicit: Extra file paths from ``--edf``.

    Returns:
        Unique PSG paths.
    """
    found: list[Path] = []
    if edf_dir is not None:
        found.extend(sorted(edf_dir.glob("*.edf")))
    found.extend(Path(item) for item in explicit)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        if "Hypnogram" in path.name:
            continue
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def main(argv: list[str] | None = None) -> None:
    """Score each EDF and refresh the recording index.

    Args:
        argv: Optional argument vector, excluding the program name.
    """
    parser = argparse.ArgumentParser(description="Build dashboard recording JSON.")
    parser.add_argument("--edf-dir", type=Path, default=None)
    parser.add_argument("--edf", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model.json")
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    edfs = iter_edfs(args.edf_dir, args.edf)
    if not edfs:
        raise SystemExit("no EDF files found")
    for edf in edfs:
        print(f"scoring {edf.name}", flush=True)
        record = build_recording(edf, out_dir, config)
        sqi = record["sqi"]
        model = record["model"]
        assert isinstance(sqi, dict)
        assert isinstance(model, dict)
        print(
            f"  {record['recording_id']} sqi={sqi.get('total_arithmetic')} "
            f"band={sqi.get('band')} reference={record['reference'] is not None} "
            f"inference_s={model.get('inference_seconds')}",
            flush=True,
        )
    write_index(out_dir, config)
    print(f"wrote {out_dir / 'index.json'}", flush=True)


if __name__ == "__main__":
    main()
