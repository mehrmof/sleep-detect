# Portability patch diff

## `integration/sleep-dashboard-handoff/config/model.json`

- **Old:** `bundle_path` named `D:/Hackathon5/artifacts/FINAL_CLASSIFIER_RC1/freeze.json`.
- **New:** `bundle_path` is `../../artifacts/FINAL_CLASSIFIER_RC1/freeze.json`, resolved by the existing builder against its own repository root.
- **Old:** `predict_command` embedded the workstation Python executable and `D:/Hackathon5/predict_final.py`.
- **New:** `predict_command` is `python ../../predict_final.py --input {edf} --output {csv}`. The existing `predict_command()` function replaces the exact `python` token with the active interpreter and `_run()` retains its established repository-root working directory.
- **Reason:** remove three workstation-specific runtime paths while retaining the established adapter interface.
- **Scientific behavior affected:** **NO**. No classifier or SQI source, model byte, probability operation, preprocessing rule, channel rule, or refusal rule changed.

## Unchanged files

`predict_final.py` was not edited; its original and portable SHA-256 are both `2ba652f2c7d02e547603b3e4378c8578af96871eb11d1076f2971491739746cd`. All six weight hashes match the historical release manifest and staged copies. `sqi.py` and the dashboard builder were not edited.
