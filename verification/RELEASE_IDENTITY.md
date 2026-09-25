# Release identity

Scientific release: `FINAL_CLASSIFIER_RC1`

Scientific freeze SHA-256: `c6e1f71f310999be5a629c37884544cfb2f643f8fa1e09231544a5baa213a010`

SQI: `sqi-1.0.0`

Reproducibility status: `READY_WITH_LIMITATIONS`

The six model files are frozen. The portability patch changed only path/configuration handling and did not alter predictions; `predict_final.py` remained byte-identical. The validated release path uses CUDA. CPU execution is unsupported and not validated.

## RC1-to-SQI release interface

Canonical RC1 output is translated to the `sqi-1.0.0` CLI schema by the non-scientific `rc1_to_sqi.py` schema adapter. The adapter renames established fields and preserves row order, row count, onset values, stage values, and probability values. It does not alter classifier predictions or SQI methodology.
