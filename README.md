# AI for Better Sleep — judge package

This package contains the frozen `FINAL_CLASSIFIER_RC1` clinical-PSG sleep-stage classifier, its six frozen weights, the deployed `sqi-1.0.0` implementation, an explicit non-scientific RC1-to-SQI schema adapter, the audited technical report, the future hardware concept, and release-verification artifacts.

## Included

- Frozen RC1 inference code and model weights.
- `rc1_to_sqi.py`, which translates RC1 column names to the existing SQI CLI schema without changing values.
- `sqi-1.0.0`, tests, and machine-readable contracts.
- `MASTER_REPORT.md`, hardware concept documents, and hash manifests.

## Not included

Sleep-EDF recordings, hypnograms, sealed labels, subject-level sealed dumps, private development caches, and a physical wearable are not included. The hardware is concept and simulation only; no wearable was built or tested.

## Validated environment

Windows; Python 3.12.14; PyTorch 2.6.0+cu126; CUDA 12.6; NVIDIA GeForce RTX 3050 Laptop GPU. Other retained versions are pinned in `verification/requirements_release.txt`. Install the CUDA-compatible PyTorch build through the appropriate official PyTorch index, then install the remaining pinned packages. The validated release path requires CUDA. CPU inference is unsupported and not validated.

## Input contract

Input is EDF with exact `EEG Fpz-Cz`, `EEG Pz-Oz`, horizontal EOG, and submental EMG channels. EEG/EOG must be 100 Hz and signals are expected in µV. Validated EMG handling includes ST-style 100 Hz raw EMG and SC-style 1 Hz envelope EMG. Epochs are 30 seconds; a final partial epoch is omitted. See `verification/classifier_input_contract.json`.

## Classifier inference

Run these commands from the extracted package root.

Single EDF:

```bash
python predict_final.py --input <recording.edf> --output <predictions.csv>
```

Directory of EDFs:

```bash
python predict_final.py --input <edf_directory> --output <output_directory>
```

The canonical RC1 output contains `onset`, `predicted_stage`, and `P(Wake)` through `P(REM)`, plus recording/epoch keys, duration, quality flags, and model version. See `verification/classifier_output_contract.json`.

## SQI

The frozen SQI CLI uses its established `onset_sec`, `pred`, and `prob_*` schema. Convert the canonical RC1 CSV, then run SQI:

```bash
python rc1_to_sqi.py --input <predictions.csv> --output <sqi_input.csv>
python integration/sleep-dashboard-handoff/sqi.py <sqi_input.csv> [sqi_output.csv]
```

The adapter performs column mapping only: `onset → onset_sec`, `predicted_stage → pred`, `epoch_index → epoch_idx`, and the five probability-column renames. It preserves rows and values and performs no scientific transformation. The SQI is an interpretable objective hypnogram-derived project composite. It is not PSQI, clinically validated, or diagnostic.

## Tests

```bash
python -m unittest -v test_rc1_to_sqi.py
python integration/sleep-dashboard-handoff/test_sqi.py
```

## Performance and runtime

The deployment-faithful full-EDF diagnostic Macro-F1 was **0.7812** on the retained sealed recordings. This is a post-freeze diagnostic on the same recordings as the historical controlled evaluation, not an independent second test. Historical label-window-dependent values are documented with their protocol qualifiers in `MASTER_REPORT.md`.

Validated runtime was 7.27 seconds for a 9.97-hour EDF and 10.25 seconds for a 22.08-hour EDF on the validated CUDA laptop GPU.

## Limitations

- CUDA is the validated release path; CPU inference is unsupported/not validated.
- Channel names, montage, units, and rates are strict.
- Historical controlled scores used label-derived evaluation windows and are not guaranteed full-EDF performance.
- `sqi-1.0.0` is nonclinical and has no uncertainty interval.
- The hardware is concept-only and cannot run frozen RC1; it requires a different reduced-input model.
- Training reproduction is partial and requires separately obtained Sleep-EDF data and annotations.

## Verification

`verification/package_manifest.csv` inventories the archive. `verification/package_sha256.txt` gives key hashes, and `verification/RELEASE_IDENTITY.md` records the scientific freeze and adapter boundary. Sleep-EDF is not redistributed; obtain it from PhysioNet under its applicable terms.
