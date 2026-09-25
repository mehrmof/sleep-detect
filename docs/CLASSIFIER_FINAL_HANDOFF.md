# Classifier Final Handoff

## Final decision

**Status: `CLASSIFICATION_TRACK_CLOSED`**

The final deployable classifier is `FINAL_CLASSIFIER_RC1`:

```text
0.60 × N06
+ 0.20 × C01 stride-20
+ 0.20 × C01 overlap stride-10
→ P(N3) × 0.60
→ probability renormalization
→ argmax
```

RC1 freeze SHA-256: `c6e1f71f310999be5a629c37884544cfb2f643f8fa1e09231544a5baa213a010`. RC2 calibration is not part of competition inference.

## Performance evidence by context

| Context | Macro-F1 | Interpretation |
|---|---:|---|
| Strict label-free fresh full-EDF, MSP scoring | **0.781188** | Conservative deployment-style evidence |
| Historical sealed MSP-60 | **0.796358** | Historical label-derived window preprocessing |
| Historical sealed Naive | **0.805174** | Historical label-derived window preprocessing |
| Five-person approximately 9-hour diagnostic | **0.8009** | Secondary diagnostic; never used for tuning |

Historical window metrics are valid for their original protocol but are not guaranteed full-EDF deployment performance.

Development milestones are retained as traceable evidence, not directly interchangeable estimates:

| Milestone | Approx. Macro-F1 |
|---|---:|
| N02 | 0.7097 |
| N03_raw_plus_normalized | 0.7356 |
| N04_wider_context | 0.7539 |
| N05_class_weights | 0.7582 |
| N06 | 0.7590 |
| FINAL_CLASSIFIER_RC1_development_MSP | 0.7753 |

## Closed research branches

- **R03 BiGRU-40 — REJECTED:** modest N1 improvement did not offset N2/N3 and pooled Macro-F1 degradation.
- **R04 focal loss — REJECTED:** substantial Macro-F1 degradation and collapsed N1 precision.
- **R07 transition prior — REJECTED:** λ=0.10/0.20/0.30 all reduced Macro-F1; smoothing did not improve accuracy. No HMM, Viterbi, duration rules, or transition smoothing is present.
- **RC2 calibration — REJECTED FOR DEPLOYMENT:** strong window-based development gains failed to replicate in one-shot complete-EDF inference (`RC2 0.762864` versus fresh upstream RC1 `0.781188`).

## Preprocessing mismatch limitation

The audit found 41,711 exact shared historical/fresh MSP epoch keys and zero historical-only keys. Raw N06 engineered features were identical, while final RC1 probabilities differed by mean `≈0.0373`, maximum `≈0.6281`, and argmax on `≈9.52%` of epochs. Historical preprocessing reproduced its stored artifact to maximum error `≈1.11e-16` with zero argmax changes.

Causes:

1. Historical N06 used `label-derived window → normalization → context`; fresh inference uses `complete EDF → normalization → context`.
2. Historical CNN used `Naive window → waveform normalization → sequence construction`; fresh inference starts normalization and sequence construction from the complete EDF.
3. Cropping changes rolling/context support and edge replication.

Label-derived boundaries cannot be used during organizer inference. Historical trimmed metrics must remain identified as window-dependent.

## Competition-day input contract

- Input is the organizer-supplied PSG EDF processed exactly as supplied.
- No hypnogram or label is required or permitted for inference.
- Required exact channel names: `EEG Fpz-Cz`, `EEG Pz-Oz`, `EOG horizontal`, `EMG submental`.
- EEG/EOG must be 100 Hz. The validated EMG path supports the dataset's 1 Hz envelope and 100 Hz raw representations.
- Do not use MSP/Naive cropping, first/last-sleep labels, RC2 calibration, HMM, or Viterbi.

Output columns:

```text
recording_id, epoch_index, onset, duration, predicted_stage,
quality_flags, model_version,
P(Wake), P(N1), P(N2), P(N3), P(REM)
```

Probabilities must be finite, nonnegative, and sum to one. `predicted_stage` must equal argmax. `model_version` must be `FINAL_CLASSIFIER_RC1`.

## Competition-day commands

From `D:\Hackathon5`:

One EDF:

```powershell
& 'C:\Users\rs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'predict_final.py' --input 'D:\path\recording.edf' --output 'D:\path\predictions.csv'
```

Directory of EDFs:

```powershell
& 'C:\Users\rs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'predict_final.py' --input 'D:\path\edf_folder' --output 'D:\path\prediction_folder'
```

## Competition-day smoke procedure

1. Verify Python, CUDA, and the RTX 3050 are available.
2. Recompute hashes and compare them with `classifier_final_manifest.json`.
3. Inspect EDF channel labels and sampling rates.
4. Run frozen `predict_final.py` without labels.
5. Verify output rows equal `floor(recording_duration_seconds / 30)`.
6. Verify finite, nonnegative probabilities summing to one.
7. Verify `predicted_stage` equals argmax.
8. Save predictions and the batch report under a new output directory.
9. Do not alter the classifier based on prediction appearance or organizer results.

## Environment

- Validated Python: `C:\Users\rs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- PyTorch `2.6.0+cu126`, CUDA runtime `12.6`
- GPU: `NVIDIA GeForce RTX 3050 Laptop GPU`
- Observed peak RSS: approximately `1162 MB`
- CPU fallback exists through device selection but was not benchmarked.

## Known limitations

- Strict montage and exact channel-name matching; equivalent differently named channels can be rejected.
- Submental EMG is mandatory in the current pipeline.
- EEG/EOG must be 100 Hz.
- Robust normalization is sensitive to the supplied recording duration/population.
- Historical evaluation used label-derived windows unavailable at hidden inference.
- Subject performance varies, and N1 remains the hardest class.
- CPU inference runtime is unknown.

## Frozen component hashes

Manifest SHA-256: `12a272c53a5f35b963555774bbef9ae6fe1fa2801ef8a0ff755565c99a381ccc`

| Component | SHA-256 |
|---|---|
| `predict_final.py` | `2ba652f2c7d02e547603b3e4378c8578af96871eb11d1076f2971491739746cd` |
| `artifacts/N06/model.joblib` | `66af8808a9fca42c6046e83a09e02bdcd10d912cc5d013ae02ba52837c7fa262` |
| `artifacts/FINAL_CLASSIFIER_RC1/freeze.json` | `c6e1f71f310999be5a629c37884544cfb2f643f8fa1e09231544a5baa213a010` |
| `artifacts/FINAL_CLASSIFIER_RC1/release_manifest.json` | `d514f6ee241e5de2e48ed5027b4a393bc3a7f945f4b1cb957b0d7aa2ae2faa75` |
| `artifacts/CNN_C01/fold_1/best.pt` | `da828fefa7bb452ac5c2a31c49bdadf8ea6f209a6c9f3a84253f589afc56a28b` |
| `artifacts/CNN_C01/fold_2/best.pt` | `e894001d17d2015b5a048bd04421b0fd07e17394be98db43170a747eeb87618c` |
| `artifacts/CNN_C01/fold_3/best.pt` | `932aac9d7ee8bdb70d1652b12e84fddc240d337fc426593c9b5bc9c68d238a74` |
| `artifacts/CNN_C01/fold_4/best.pt` | `d7a1ffcb3c29eacb752425077ba628ae415a46a0b5d3e3ee411d5cce55843024` |
| `artifacts/CNN_C01/fold_5/best.pt` | `43cc48f8720fce07b572ad395e7f7e6d966904c6daf93e32f1d69181a41ab601` |

## Closure

- Final classifier: `FINAL_CLASSIFIER_RC1`
- RC2 calibration: rejected
- HMM/transition decoding: rejected
- Further classifier tuning: stopped
- Hidden/organizer results must not trigger tuning
- Reopen only for a genuine implementation failure that prevents inference

**CLASSIFICATION_TRACK_CLOSED**
