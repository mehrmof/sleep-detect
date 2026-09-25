#!/usr/bin/env python3
"""Hand-checkable hypnograms for the Sleep Quality Index.

Each night below is built from counted 30 s epochs so the component values
can be recomputed with a calculator. Anchor checks sit beside the eight
hypnograms required for the deliverable.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sqi


def night(*blocks: tuple[float, int]) -> np.ndarray:
    """Repeat ``(code, count)`` blocks into one stage array.

    Args:
        blocks: Pairs of stage code and epoch count. NaN is unscored.

    Returns:
        Float stage array.
    """
    parts = [np.full(count, code, dtype=float) for code, count in blocks]
    return np.concatenate(parts)


class AnchorTests(unittest.TestCase):
    """Piecewise maps at the published and assumed knots."""

    def test_duration_knots(self) -> None:
        """TST anchors: 0 at 5 h, 1 on 7–9 h, 0.7 from 10 h."""
        self.assertEqual(sqi.score_duration(4.0), 0.0)
        self.assertEqual(sqi.score_duration(5.0), 0.0)
        self.assertAlmostEqual(sqi.score_duration(6.0), 0.5)
        self.assertEqual(sqi.score_duration(7.0), 1.0)
        self.assertEqual(sqi.score_duration(8.0), 1.0)
        self.assertEqual(sqi.score_duration(9.0), 1.0)
        self.assertAlmostEqual(sqi.score_duration(9.5), 0.85)
        self.assertAlmostEqual(sqi.score_duration(10.0), 0.7)
        self.assertAlmostEqual(sqi.score_duration(12.0), 0.7)

    def test_efficiency_latency_waso_sfi_knots(self) -> None:
        """SE, SOL, WASO, and SFI at the knots used by the index."""
        self.assertEqual(sqi.score_efficiency(0.65), 0.0)
        self.assertAlmostEqual(sqi.score_efficiency(0.75), 0.5)
        self.assertEqual(sqi.score_efficiency(0.85), 1.0)
        self.assertEqual(sqi.score_latency(15.0), 1.0)
        self.assertAlmostEqual(sqi.score_latency(30.0), 2.0 / 3.0)
        self.assertEqual(sqi.score_latency(60.0), 0.0)
        self.assertEqual(sqi.score_continuity(20.0), 1.0)
        self.assertAlmostEqual(sqi.score_continuity(31.0), 29.0 / 40.0)
        self.assertEqual(sqi.score_continuity(60.0), 0.0)
        self.assertEqual(sqi.score_fragmentation(10.0), 1.0)
        self.assertAlmostEqual(sqi.score_fragmentation(20.0), 0.5)
        self.assertEqual(sqi.score_fragmentation(30.0), 0.0)


class HypnogramTests(unittest.TestCase):
    """Eight hypnograms with values fixed before the assertions."""

    def test_perfect_night(self) -> None:
        """15 min wake, then 7.5 h of sleep inside every full-credit band.

        Sleep epochs: 120 N2, 198 REM (22%), 162 N3 (18%), 420 N2.
        REM starts after 60 min. Three stage changes in 7.5 h.
        """
        stages = night((0, 30), (2, 120), (4, 198), (3, 162), (2, 420))
        record = sqi.score_stages(stages, provenance="synthetic")
        self.assertIsNone(record["unavailable_reason"])
        self.assertAlmostEqual(record["sol_min"], 15.0)
        self.assertAlmostEqual(record["tst_min"], 450.0)
        self.assertAlmostEqual(record["waso_min"], 0.0)
        self.assertAlmostEqual(record["pct_n3"], 18.0)
        self.assertAlmostEqual(record["pct_rem"], 22.0)
        self.assertAlmostEqual(record["rem_latency_min"], 60.0)
        self.assertEqual(record["n_stage_transitions"], 3)
        for key in (
            "c_duration",
            "c_efficiency",
            "c_latency",
            "c_continuity",
            "c_fragmentation",
            "c_architecture",
        ):
            self.assertEqual(record[key], 1.0, key)
        self.assertAlmostEqual(record["sqi_arithmetic"], 100.0)
        self.assertAlmostEqual(record["sqi_geometric"], 100.0)

    def test_all_wake(self) -> None:
        """Eight hours of wake has no sleep period and no score."""
        record = sqi.score_stages(night((0, 16 * 60)))
        self.assertEqual(record["unavailable_reason"], "no_sleep")
        self.assertIsNone(record["sqi_arithmetic"])
        self.assertIsNone(record["tst_min"])

    def test_fragmented_night(self) -> None:
        """Thirty minutes of wake, then four hours of N2/wake alternation.

        Onset is epoch 60. Final awakening is the last N2 (epoch 538).
        TST is 120 min, WASO is 119.5 min, SE is 120/270, and SFI is
        478 * 120 / 479, which is above 30 per hour. Only latency scores.
        """
        body = np.tile(np.array([2.0, 0.0]), 240)
        stages = np.concatenate([np.zeros(60), body])
        record = sqi.score_stages(stages)
        self.assertIsNone(record["unavailable_reason"])
        self.assertAlmostEqual(record["tst_min"], 120.0)
        self.assertAlmostEqual(record["waso_min"], 119.5)
        self.assertAlmostEqual(record["sol_min"], 30.0)
        self.assertAlmostEqual(record["se"], 120.0 / 270.0)
        self.assertAlmostEqual(record["sfi_per_h"], 478.0 * 120.0 / 479.0)
        self.assertEqual(record["c_duration"], 0.0)
        self.assertEqual(record["c_efficiency"], 0.0)
        self.assertEqual(record["c_continuity"], 0.0)
        self.assertEqual(record["c_fragmentation"], 0.0)
        self.assertEqual(record["c_architecture"], 0.0)
        self.assertAlmostEqual(record["c_latency"], 2.0 / 3.0)
        self.assertAlmostEqual(record["sqi_arithmetic"], 100.0 * 0.10 * (2.0 / 3.0))
        self.assertEqual(record["sqi_geometric"], 0.0)

    def test_short_night(self) -> None:
        """Thirty minutes of wake and three hours of N2.

        The window is 3.5 h, so the night is computable. TST is below 5 h.
        SE is 180/210, which clears 85%. There is no N3 and no REM.
        """
        record = sqi.score_stages(night((0, 60), (2, 360)))
        self.assertIsNone(record["unavailable_reason"])
        self.assertAlmostEqual(record["tst_min"], 180.0)
        self.assertAlmostEqual(record["tib_min"], 210.0)
        self.assertEqual(record["c_duration"], 0.0)
        self.assertEqual(record["c_efficiency"], 1.0)
        self.assertAlmostEqual(record["c_latency"], 2.0 / 3.0)
        self.assertEqual(record["c_continuity"], 1.0)
        self.assertEqual(record["c_fragmentation"], 1.0)
        self.assertEqual(record["c_architecture"], 0.0)
        expected = 100.0 * (0.25 + 0.20 + 0.15 + 0.10 * (2.0 / 3.0))
        self.assertAlmostEqual(record["sqi_arithmetic"], expected)
        self.assertEqual(record["sqi_geometric"], 0.0)

    def test_no_rem(self) -> None:
        """N2 and N3 only. REM percent and REM latency both score zero.

        N3 is 80/480 of sleep, inside 13–23%, so architecture is one third.
        """
        record = sqi.score_stages(night((0, 60), (2, 200), (3, 80), (2, 200)))
        self.assertIsNone(record["unavailable_reason"])
        self.assertEqual(record["pct_rem"], 0.0)
        self.assertIsNone(record["rem_latency_min"])
        self.assertEqual(record["c_rem"], 0.0)
        self.assertEqual(record["c_rem_latency"], 0.0)
        self.assertEqual(record["c_n3"], 1.0)
        self.assertAlmostEqual(record["c_architecture"], 1.0 / 3.0)

    def test_no_n3(self) -> None:
        """REM at 20% after a 60 min latency, and no N3.

        Architecture is the mean of 0, 1, and 1.
        """
        record = sqi.score_stages(night((0, 60), (2, 120), (4, 80), (2, 200)))
        self.assertIsNone(record["unavailable_reason"])
        self.assertEqual(record["pct_n3"], 0.0)
        self.assertAlmostEqual(record["pct_rem"], 20.0)
        self.assertAlmostEqual(record["rem_latency_min"], 60.0)
        self.assertEqual(record["c_n3"], 0.0)
        self.assertEqual(record["c_rem"], 1.0)
        self.assertEqual(record["c_rem_latency"], 1.0)
        self.assertAlmostEqual(record["c_architecture"], 2.0 / 3.0)

    def test_single_epoch_sleep(self) -> None:
        """One N2 epoch does not satisfy the 10-minute majority rule."""
        stages = np.zeros(400)
        stages[200] = 2.0
        record = sqi.score_stages(stages)
        self.assertEqual(record["unavailable_reason"], "no_sleep")
        self.assertIsNone(record["sqi_arithmetic"])

    def test_missing_labels_mid_night(self) -> None:
        """Thirty unscored epochs sit inside the sleep period.

        They are not TST and not WASO, so TST + WASO is 15 min short of SPT.
        N3 is 50/410 of sleep, REM is 60/410, REM latency is 165 min.
        """
        stages = night(
            (0, 60),
            (2, 200),
            (np.nan, 30),
            (2, 100),
            (4, 60),
            (3, 50),
        )
        record = sqi.score_stages(stages)
        self.assertIsNone(record["unavailable_reason"])
        self.assertAlmostEqual(record["tst_min"], 205.0)
        self.assertAlmostEqual(record["waso_min"], 0.0)
        self.assertAlmostEqual(record["spt_min"], 220.0)
        self.assertLess(record["tst_min"] + record["waso_min"], record["spt_min"])
        self.assertAlmostEqual(record["coverage_fraction"], 410.0 / 440.0)
        self.assertAlmostEqual(record["se"], 205.0 / 250.0)
        self.assertAlmostEqual(record["c_efficiency"], 0.85)
        self.assertEqual(record["c_duration"], 0.0)
        self.assertEqual(record["c_continuity"], 1.0)
        self.assertEqual(record["c_fragmentation"], 1.0)
        self.assertAlmostEqual(record["c_latency"], 2.0 / 3.0)
        self.assertAlmostEqual(record["c_n3"], (50.0 / 410.0 * 100.0) / 13.0)
        self.assertAlmostEqual(record["c_rem"], ((60.0 / 410.0 * 100.0) - 5.0) / 15.0)
        self.assertAlmostEqual(record["c_rem_latency"], 0.625)
        architecture = np.mean(
            [record["c_n3"], record["c_rem"], record["c_rem_latency"]]
        )
        self.assertAlmostEqual(record["c_architecture"], architecture)
        arithmetic = 100.0 * (
            0.25 * 0.85
            + 0.20 * 1.0
            + 0.15 * 1.0
            + 0.10 * (2.0 / 3.0)
            + 0.10 * architecture
        )
        self.assertAlmostEqual(record["sqi_arithmetic"], arithmetic)

    def test_window_too_short_precedes_sleep(self) -> None:
        """A two-hour block of sleep is refused before a score is built."""
        record = sqi.score_stages(night((2, 240)))
        self.assertEqual(record["unavailable_reason"], "window_too_short")
        self.assertIsNone(record["sqi_arithmetic"])

    def test_unknown_fraction_precedes_a_score(self) -> None:
        """More than a quarter unscored epochs is refused."""
        record = sqi.score_stages(night((np.nan, 150), (2, 250)))
        self.assertEqual(record["unavailable_reason"], "unknown_fraction")
        self.assertIsNone(record["tst_min"])

    def test_age_adjustment_does_not_change_v1(self) -> None:
        """Five percent N3 is partial at age 25 and full credit at age 80.

        v1 is identical at both ages. Only the secondary score moves.
        """
        stages = night((0, 30), (2, 180), (4, 198), (3, 45), (2, 477))
        young = sqi.score_stages(stages, age=25)
        old = sqi.score_stages(stages, age=80)
        self.assertAlmostEqual(young["pct_n3"], 5.0)
        self.assertAlmostEqual(young["sqi_arithmetic"], old["sqi_arithmetic"])
        self.assertAlmostEqual(young["c_architecture"], old["c_architecture"])
        self.assertAlmostEqual(young["c_n3"], 5.0 / 13.0)
        self.assertEqual(old["c_n3_age_adj"], 1.0)
        self.assertGreater(old["sqi_arithmetic_age_adj"], old["sqi_arithmetic"])
        self.assertAlmostEqual(
            young["sqi_arithmetic_age_adj"],
            young["sqi_arithmetic"],
        )

    def test_prediction_csv_blank_probability_is_unscored(self) -> None:
        """A blank probability row is not assigned a stage by argmax."""
        stages = night((0, 60), (2, 400))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.csv"
            lines = ["onset_sec,pred,prob_W,prob_N1,prob_N2,prob_N3,prob_REM"]
            for idx, code in enumerate(stages):
                if idx == 100:
                    lines.append(f"{idx * 30},,,,,,")
                else:
                    lines.append(f"{idx * 30},{int(code)},,,,,")
            path.write_text("\n".join(lines) + "\n")
            record = sqi.score_prediction_csv(path)[0]
        self.assertEqual(record["provenance"], "prediction")
        self.assertEqual(record["index_version"], "sqi-1.0.0")
        self.assertGreater(record["n_unscored"], 0)
        self.assertIsNone(record["unavailable_reason"])


class CycleTests(unittest.TestCase):
    """Cycle knots and the sqi-1.1.0 path. sqi-1.0.0 stays the default."""

    def test_cycle_knots(self) -> None:
        """Count, regularity, and N3 front-loading at the stated knots."""
        self.assertEqual(sqi.score_cycle_count(0), 0.0)
        self.assertEqual(sqi.score_cycle_count(1), 0.0)
        self.assertAlmostEqual(sqi.score_cycle_count(2), 1.0 / 3.0)
        self.assertEqual(sqi.score_cycle_count(4), 1.0)
        self.assertEqual(sqi.score_cycle_count(6), 1.0)
        self.assertEqual(sqi.score_cycle_count(7), 1.0)
        self.assertAlmostEqual(sqi.score_cycle_count(8.5), 0.5)
        self.assertEqual(sqi.score_cycle_count(10), 0.0)
        self.assertEqual(sqi.score_n3_front(0.35), 0.0)
        self.assertAlmostEqual(sqi.score_n3_front(0.475), 0.5)
        self.assertEqual(sqi.score_n3_front(0.60), 1.0)
        self.assertIsNone(sqi.score_n3_front(None))
        self.assertIsNone(sqi.score_cycle_regularity(None))
        self.assertEqual(sqi.score_cycle_regularity(0.0), 1.0)
        self.assertEqual(sqi.score_cycle_regularity(1.5), 0.0)

    def test_four_equal_cycles_omit_missing_n3(self) -> None:
        """Four 50-minute cycles. No N3, so front-loading is left out of the mean.

        Each cycle is 40 min of N2 and 10 min of REM. Durations match, so
        regularity is 1. The sub-score is the mean of count and regularity.
        """
        body = np.tile(np.concatenate([np.full(80, 2.0), np.full(20, 4.0)]), 4)
        stages = np.concatenate([np.zeros(60), body])
        frozen = sqi.score_stages(stages)
        descriptive = sqi.score_stages(stages, index_version="sqi-1.1.0", include_cycle=False)
        included = sqi.score_stages(stages, index_version="sqi-1.1.0", include_cycle=True)
        self.assertNotIn("c_cycles", frozen)
        self.assertEqual(frozen["index_version"], "sqi-1.0.0")
        self.assertAlmostEqual(frozen["c_architecture"], 5.0 / 9.0)
        self.assertEqual(descriptive["n_cycles"], 4)
        self.assertIsNone(descriptive["c_n3_front"])
        self.assertEqual(descriptive["c_cycle_regularity"], 1.0)
        self.assertEqual(descriptive["c_cycles"], 1.0)
        self.assertAlmostEqual(descriptive["c_architecture"], frozen["c_architecture"])
        self.assertAlmostEqual(descriptive["sqi_arithmetic"], frozen["sqi_arithmetic"])
        self.assertAlmostEqual(included["c_architecture"], 2.0 / 3.0)
        self.assertAlmostEqual(
            included["sqi_arithmetic"] - frozen["sqi_arithmetic"],
            10.0 / 9.0,
        )

    def test_one_cycle_omits_regularity(self) -> None:
        """One cycle and all of the N3 in the first half. Regularity is not a zero.

        The sub-score is the mean of a count score of 0 and a front-loading
        score of 1, which is 0.5. Scoring the missing regularity as 0 would
        give one third.
        """
        stages = night((0, 60), (3, 60), (2, 60), (4, 40), (2, 200))
        record = sqi.score_stages(stages, index_version="sqi-1.1.0", include_cycle=False)
        self.assertEqual(record["n_cycles"], 1)
        self.assertIsNone(record["c_cycle_regularity"])
        self.assertEqual(record["n3_front_fraction"], 1.0)
        self.assertEqual(record["c_n3_front"], 1.0)
        self.assertAlmostEqual(record["c_cycles"], 0.5)

    def test_short_rem_does_not_split_the_nrem_period(self) -> None:
        """Three minutes of REM inside NREM is not its own REM period."""
        stages = night((0, 60), (2, 80), (4, 6), (2, 20), (4, 20), (2, 200))
        record = sqi.score_stages(stages, index_version="sqi-1.1.0")
        self.assertEqual(record["n_cycles"], 1)

    def test_long_wake_ends_the_period(self) -> None:
        """Twenty minutes of wake between NREM and REM prevents a cycle."""
        stages = night((0, 60), (2, 80), (0, 40), (4, 20), (2, 200))
        record = sqi.score_stages(stages, index_version="sqi-1.1.0")
        self.assertEqual(record["n_cycles"], 0)
        self.assertIsNone(record["c_cycle_regularity"])
        self.assertEqual(record["c_cycles"], 0.0)


class InputValidationTests(unittest.TestCase):
    """Strict clock and probability checks added without changing valid scores."""

    def test_duplicate_onset_is_rejected(self) -> None:
        stages = night((0, 60), (2, 400))
        onsets = np.arange(stages.size, dtype=float) * 30
        onsets[100] = onsets[99]
        self.assertEqual(sqi.score_stages(stages, onsets)["unavailable_reason"], "invalid_epoch_onsets")

    def test_reversed_pair_is_rejected(self) -> None:
        stages = night((0, 60), (2, 400))
        onsets = np.arange(stages.size, dtype=float) * 30
        onsets[100], onsets[101] = onsets[101], onsets[100]
        self.assertEqual(sqi.score_stages(stages, onsets)["unavailable_reason"], "invalid_epoch_onsets")

    def test_fully_reordered_is_rejected(self) -> None:
        stages = night((0, 60), (2, 400))
        onsets = (np.arange(stages.size, dtype=float) * 30)[::-1]
        self.assertEqual(sqi.score_stages(stages, onsets)["unavailable_reason"], "invalid_epoch_onsets")

    def test_monotonic_onsets_preserve_score(self) -> None:
        stages = night((0, 60), (2, 400))
        baseline = sqi.score_stages(stages)
        explicit = sqi.score_stages(stages, np.arange(stages.size) * 30.0)
        self.assertIsNone(explicit["unavailable_reason"])
        self.assertEqual(explicit["sqi_arithmetic"], baseline["sqi_arithmetic"])

    def _prob_stage(self, cells: list[object]) -> float:
        row = {"onset_sec": "0"}
        row.update(dict(zip(sqi.PROB_COLUMNS, map(str, cells), strict=True)))
        return float(sqi._rows_to_arrays([row])[0][0])

    def test_valid_probability_row(self) -> None:
        self.assertEqual(self._prob_stage([0.1, 0.2, 0.5, 0.1, 0.1]), 2.0)

    def test_negative_probability_is_unscored(self) -> None:
        self.assertTrue(np.isnan(self._prob_stage([-0.1, 0.2, 0.7, 0.1, 0.1])))

    def test_nan_probability_is_unscored(self) -> None:
        self.assertTrue(np.isnan(self._prob_stage([0.1, 0.2, "nan", 0.3, 0.4])))

    def test_inf_probability_is_unscored(self) -> None:
        self.assertTrue(np.isnan(self._prob_stage([0.1, 0.2, "inf", 0.3, 0.4])))

    def test_sum_ten_is_unscored(self) -> None:
        self.assertTrue(np.isnan(self._prob_stage([2, 2, 2, 2, 2])))

    def test_missing_probability_is_unscored(self) -> None:
        self.assertTrue(np.isnan(self._prob_stage([0.1, 0.2, "", 0.3, 0.4])))

    def test_tiny_sum_deviation_is_accepted(self) -> None:
        self.assertEqual(self._prob_stage([0.1, 0.2, 0.5, 0.1, 0.1000005]), 2.0)


if __name__ == "__main__":
    unittest.main()
