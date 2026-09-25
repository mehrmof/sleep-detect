"""Translate frozen RC1 prediction CSVs to the sqi-1.0.0 CLI schema.

This utility performs schema translation only. It does not alter classifier
predictions, probabilities, stages, onset values, or row order.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


SOURCE_FIELDS = (
    "recording_id",
    "epoch_index",
    "onset",
    "duration",
    "predicted_stage",
    "quality_flags",
    "model_version",
    "P(Wake)",
    "P(N1)",
    "P(N2)",
    "P(N3)",
    "P(REM)",
)
TARGET_FIELDS = (
    "recording_id",
    "epoch_idx",
    "onset_sec",
    "prob_W",
    "prob_N1",
    "prob_N2",
    "prob_N3",
    "prob_REM",
    "pred",
)
STAGES = {"Wake", "N1", "N2", "N3", "REM"}


def adapt_csv(source: str | Path, destination: str | Path) -> int:
    """Write an SQI-compatible copy of a canonical RC1 prediction CSV."""
    source_path = Path(source)
    destination_path = Path(destination)
    with source_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        missing = [name for name in SOURCE_FIELDS if name not in headers]
        if missing:
            raise ValueError(f"missing required RC1 columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("RC1 prediction CSV contains no rows")

    converted: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=2):
        onset_text = row["onset"]
        try:
            onset_value = float(onset_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {number}: onset is not numeric") from exc
        if not math.isfinite(onset_value):
            raise ValueError(f"row {number}: onset is not finite")
        stage = row["predicted_stage"]
        if stage not in STAGES:
            raise ValueError(f"row {number}: invalid predicted_stage {stage!r}")
        converted.append(
            {
                "recording_id": row["recording_id"],
                "epoch_idx": row["epoch_index"],
                "onset_sec": onset_text,
                "prob_W": row["P(Wake)"],
                "prob_N1": row["P(N1)"],
                "prob_N2": row["P(N2)"],
                "prob_N3": row["P(N3)"],
                "prob_REM": row["P(REM)"],
                "pred": stage,
            }
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(converted)
    return len(converted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate canonical FINAL_CLASSIFIER_RC1 CSV to sqi-1.0.0 input schema."
    )
    parser.add_argument("--input", required=True, help="Canonical RC1 prediction CSV")
    parser.add_argument("--output", required=True, help="SQI-compatible CSV")
    args = parser.parse_args()
    count = adapt_csv(args.input, args.output)
    print(f"adapted_rows={count}")


if __name__ == "__main__":
    main()
