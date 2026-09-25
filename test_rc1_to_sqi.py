import csv
import tempfile
import unittest
from pathlib import Path

from rc1_to_sqi import SOURCE_FIELDS, adapt_csv


def row(index=0, onset="0", stage="Wake"):
    return {
        "recording_id": "T",
        "epoch_index": str(index),
        "onset": onset,
        "duration": "30",
        "predicted_stage": stage,
        "quality_flags": "",
        "model_version": "FINAL_CLASSIFIER_RC1",
        "P(Wake)": "0.7",
        "P(N1)": "0.1",
        "P(N2)": "0.1",
        "P(N3)": "0.05",
        "P(REM)": "0.05",
    }


class AdapterTests(unittest.TestCase):
    def convert(self, rows, fields=SOURCE_FIELDS):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.csv"
            target = Path(folder) / "target.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            count = adapt_csv(source, target)
            with target.open(newline="", encoding="utf-8") as handle:
                output = list(csv.DictReader(handle))
            return count, output

    def test_valid_canonical_csv(self):
        count, output = self.convert([row()])
        self.assertEqual(count, 1)
        self.assertEqual(output[0]["onset_sec"], "0")
        self.assertEqual(output[0]["pred"], "Wake")

    def test_missing_onset(self):
        fields = tuple(name for name in SOURCE_FIELDS if name != "onset")
        with self.assertRaisesRegex(ValueError, "onset"):
            self.convert([{key: value for key, value in row().items() if key != "onset"}], fields)

    def test_non_numeric_onset(self):
        with self.assertRaisesRegex(ValueError, "not numeric"):
            self.convert([row(onset="later")])

    def test_missing_predicted_stage(self):
        fields = tuple(name for name in SOURCE_FIELDS if name != "predicted_stage")
        with self.assertRaisesRegex(ValueError, "predicted_stage"):
            self.convert(
                [{key: value for key, value in row().items() if key != "predicted_stage"}],
                fields,
            )

    def test_invalid_stage(self):
        with self.assertRaisesRegex(ValueError, "invalid predicted_stage"):
            self.convert([row(stage="N4")])

    def test_row_order_preserved(self):
        _, output = self.convert([row(2, "60", "N2"), row(0, "0", "Wake")])
        self.assertEqual([item["epoch_idx"] for item in output], ["2", "0"])

    def test_epoch_count_preserved(self):
        source = [row(index, str(index * 30), "N2") for index in range(4)]
        count, output = self.convert(source)
        self.assertEqual(count, len(source))
        self.assertEqual(len(output), len(source))

    def test_numeric_onset_preserved_exactly(self):
        _, output = self.convert([row(onset="30.000000")])
        self.assertEqual(output[0]["onset_sec"], "30.000000")


if __name__ == "__main__":
    unittest.main()
