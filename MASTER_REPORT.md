# AI for Better Sleep

## An Audited End-to-End System for Five-Stage Sleep Classification, Objective Sleep Quality Scoring, and Wearable Translation

## Executive summary

This project implements and audits an end-to-end software path from a clinical polysomnography (PSG) EDF to 30-second sleep-stage probabilities, an interpretable objective Sleep Quality Index (SQI), and dashboard-ready data:

```text
Clinical PSG EDF
→ FINAL_CLASSIFIER_RC1
→ five-stage probabilities and hypnogram
→ sqi-1.0.0
→ dashboard
```

The final classifier, `FINAL_CLASSIFIER_RC1`, is frozen. It combines a 775-feature LightGBM model (N06) with two inference views of the same five-fold CNN-LSTM checkpoints, then applies a fixed N3 probability correction. On the **historical label-window-dependent sealed evaluation**, pooled five-class Macro-F1 was **0.796358** for the project-defined MSP-60 population and **0.805174** for the Naive population. A later **deployment-faithful full-EDF diagnostic on the same sealed recordings, not an independent second test set**, produced Macro-F1 **0.781188**. This lower, conservative figure reflects how the frozen pipeline operates without label-derived inference windows. A secondary five-subject diagnostic produced pooled Macro-F1 **0.8009**; it was not selection evidence.

Release validation established label-free, deterministic inference in the validated CUDA environment. Two fresh runs had maximum probability difference 0.0. Six unlabeled rehearsal EDFs completed without failure. Runtime was 7.27 seconds for a 9.97-hour recording and 10.25 seconds for a 22.08-hour recording, with peak process RSS of roughly 1.1 GB. The release rejects missing channels, unsupported rates, corrupt files, and zero-complete-epoch inputs rather than fabricating data.

The deployed `sqi-1.0.0` is an objective, hypnogram-derived composite of efficiency, continuity, duration, fragmentation, latency, and architecture. It is not PSQI, a diagnosis, or a clinically validated universal measure of sleep quality. In the primary propagation analysis, frozen RC1 ran label-free over full development EDFs; expert labels were aligned only afterward. Among 156 paired recordings, RC1-derived SQI had MAE **5.353**, bias **+1.743**, Spearman **0.881**, and Bland–Altman limits **−12.603 to +16.089** points. These results quantify both useful rank preservation and material per-recording disagreement.

The hardware work is **concept only; no physical prototype was built**. The proposed future path uses Fpz-to-mastoid EEG, horizontal outer-canthi EOG, an ADS1299-4 at 250 samples/s, a proposed 64 Hz classifier representation, nRF52840 BLE streaming, and phone-side inference. A clinical-PSG software ablation using Fpz-Cz plus EOG reached OOF Macro-F1 **0.7537** at 100 Hz (0.7543 after retrospective 64 Hz resampling). This is not wearable validation. Frozen RC1 requires Fpz-Cz, Pz-Oz, horizontal EOG, and submental EMG, so it cannot run directly on the proposed montage; a future reduced-input model is required.

The release portability gate is **READY_WITH_LIMITATIONS**. A path-only integration patch removed three workstation-specific runtime paths while leaving `predict_final.py`, all six model files, the scientific freeze, and predictions unchanged. External staging reproduced T02 exactly (`max ΔP = 0`, zero stage differences), processed all six rehearsal EDFs, and reproduced the SQI value exactly. CPU inference remains unsupported as a release path, training reproduction is partial, and protected sealed labels are intentionally excluded.

[Internal evidence: Steps 4, 6, 7, and 8B; `final_classifier_results.json`, `sqi_final_contract.json`, `hardware_final_contract.json`, `portable_release_manifest.json`]

## Contributions

1. A subject-disjoint five-stage sleep-classification protocol with explicit train, development, and sealed partitions.
2. A frozen three-stream ensemble combining complementary engineered-feature and waveform representations.
3. A direct audit of historical label-window-dependent preprocessing versus deployment-faithful full-EDF inference.
4. A transparent six-component objective SQI with refusal behavior and disclosed construct limitations.
5. Deployment-faithful RC1-to-SQI propagation evidence on 156 development recordings.
6. A portable, deterministic CUDA release with explicit input, output, failure, and hash contracts.
7. An evidence-tiered wearable concept derived from clinical-PSG software ablations and engineering calculations.
8. Traceable negative-result reporting that records when added complexity failed.

# Part A — Data and protocol

## A1. Dataset and local inventory

The local corpus is **Sleep-EDF Database Expanded v1.0.0** [1]. Official dataset documentation and local project accounting are kept separate; the latter defines the actual experimental population.

| Cohort | Subjects | PSG recordings | Matched hypnograms | Signal hours | Median duration | Range |
|---|---:|---:|---:|---:|---:|---:|
| Sleep Cassette (SC) | 78 | 153 | 153 | 3,470.200 | 22.967 h | 17.000–24.000 h |
| Sleep Telemetry (ST) | 22 | 44 | 44 | 378.836 | 8.561 h | 7.494–10.697 h |
| **Total** | **100** | **197** | **197** | **3,849.036** | **22.683 h** | **7.494–24.000 h** |

SC contains long home recordings; ST contains approximately nine-hour hospital recordings with placebo/temazepam nights. [Internal evidence: Step 2 / `data_protocol_contract.json`]

## A2. Signals, annotations, and harmonization

| Signal | Exact local label | SC acquisition | ST acquisition | RC1 use |
|---|---|---|---|---|
| EEG | `EEG Fpz-Cz` | 100 Hz, µV | 100 Hz, µV | N06 and CNN |
| EEG | `EEG Pz-Oz` | 100 Hz, µV | 100 Hz, µV | N06 and CNN |
| EOG | `EOG horizontal` | 100 Hz, µV | 100 Hz, µV | N06 and CNN |
| EMG | `EMG submental` | 1 Hz processed envelope | 100 Hz raw | N06; required |

ST EMG is high-pass filtered at 10 Hz and converted to a one-second RMS envelope; SC uses its native one-Hz envelope. Native cohort acquisition is therefore not identical.

Sleep-EDF annotations follow the Rechtschaffen–Kales lineage. The project maps them into five classes:

| Original | Target |
|---|---|
| W | Wake |
| Stage 1 | N1 |
| Stage 2 | N2 |
| Stages 3 and 4 | N3 |
| R | REM |
| Movement / unknown | excluded |

This is a project conversion to a modern-style N3 category; it is not a claim that the source annotations used modern AASM scoring [2].

## A3. Epoch accounting and imbalance

The 197 signals contain **461,870** complete 30-second epochs. Retained audit metadata reconcile **457,652** usable five-class epochs and **4,218** uniquely invalid or unlabelled epochs. Invalid subtype overlaps were not retained and are not reconstructed.

| Population | Wake | N1 | N2 | N3 | REM | Total |
|---|---:|---:|---:|---:|---:|---:|
| Full usable | 289,856 (63.34%) | 25,175 (5.50%) | 88,983 (19.44%) | 19,454 (4.25%) | 34,184 (7.47%) | 457,652 |
| Training Naive | 41,343 (29.41%) | 14,244 (10.13%) | 52,386 (37.26%) | 12,870 (9.15%) | 19,739 (14.04%) | 140,582 |
| Development MSP-60 | 8,096 (19.14%) | 5,668 (13.40%) | 18,435 (43.59%) | 2,685 (6.35%) | 7,412 (17.52%) | 42,296 |
| Historical sealed MSP-60 | 8,614 (20.65%) | 4,727 (11.33%) | 17,509 (41.98%) | 3,894 (9.34%) | 6,967 (16.70%) | 41,711 |

The full corpus is dominated by Wake, while N1 and N3 are minority stages. This motivates an equal-class metric rather than headline accuracy.

## A4. Windowing contract

**Untrimmed** covers every complete signal epoch; invalid labels remain in the interval but are excluded from supervised rows.

**Naive** starts 60 epochs before the first labelled sleep epoch and ends 60 epochs after the final labelled sleep epoch, clipped to the recording. Internal Wake gaps remain.

**Project-defined MSP-60** splits ordered sleep epochs at continuous Wake bouts strictly longer than 120 epochs, selects the episode containing the most sleep epochs (earliest start breaks ties), adds ±60 epochs, and clips to the recording. This is not asserted to be a clinical MSP standard.

| Training window | Training epochs | MSP-60 validation F1 | Naive validation F1 | Decision |
|---|---:|---:|---:|---|
| Untrimmed | 271,465 | 0.673626 | 0.686920 | Reject |
| Naive | 140,582 | 0.682111 | 0.693738 | Retain |
| MSP-60 | 119,272 | 0.678462 | 0.689524 | Reject |

Training uses Naive. Historical primary development evaluation uses MSP-60; Naive is robustness evaluation. Deployment processes each supplied EDF as-is. No retained organizer artifact proves that the provider performs an equivalent label-derived crop.

## A5. Subject split and leakage control

The fixed split contains 60 training subjects (117 recordings), 20 validation subjects (40 recordings), and 20 sealed-test subjects (40 recordings). It is subject-disjoint and keeps repeated nights together. Split SHA-256: `fcb6bca7c84af0ed3151eaeaf00840ee4b3f3b7553d34c9dfbd0f497fac3e0e9`.

C01 additionally uses five deterministic subject-held-out folds over all 80 non-test subjects to produce out-of-fold CNN probabilities. This does not replace the original split. Fold-manifest SHA-256: `07dd621cef1290e106a73c6a0fc4b1509b3f3fd92e42d7b8c420ef7258cc1b71`.

| Process | Development labels | Sealed labels before freeze | Deployment labels |
|---|---|---|---|
| Model fitting/class weights | training/fold-training only | No | No |
| CNN monitoring | fold-training subjects only | No | No |
| Ensemble/N3 scaling | Yes, development selection | No | Fixed |
| Historical windows | Yes | Used after freeze for historical evaluation | Unavailable |
| Final inference | No | Predictions generated first | No |

## A6. Metric contract

The headline metric is **pooled five-class Macro-F1** in class order Wake, N1, N2, N3, REM. The implementation pools epochs, calls scikit-learn `f1_score` with all five labels, `average="macro"`, and `zero_division=0`, and averages class F1 equally. Cohen’s kappa, cohort results, and subject summaries are secondary. Nineteen Naive recordings contain zero N3, so per-recording five-class averages can be sensitive to absent-class and zero-division conventions.

# Part B — Classifier development and release

## B1. Controlled progression

| Experiment | Controlled change | MSP-60 F1 | Naive F1 | Decision |
|---|---|---:|---:|---|
| N02 | ±1 epoch LightGBM context | 0.7097 | 0.7204 | Baseline |
| N03 normalized-only | Normalized features only | 0.7297 | 0.7371 | Reject in favor of stronger representation |
| N03 raw+normalized | Preserve raw and robust-normalized features | 0.7356 | 0.7424 | Retain |
| N04 | ±2 context and ±15 rolling summaries | 0.7539 | 0.7608 | Retain |
| N05 | Balanced class weights | 0.7582 | 0.7644 | Retain |
| N06 | Harmonized submental EMG summaries | 0.7590 | 0.7651 | Retain |
| N07 | Viterbi smoothing | 0.7508 | 0.7596 | Reject |
| N08 | Age/sex metadata | 0.7546 | 0.7623 | Reject |
| C01 | Raw-waveform CNN-LSTM | 0.7221 | 0.7323 | Retain as complement |
| E13 | C01 overlap inference, no retraining | 0.7280 | 0.7383 | Retain as complement |
| RC1 | Three-view ensemble and N3 scaling | 0.7753 | 0.7845 | Freeze |

[Internal evidence: Step 3 / `classifier_experiment_ledger.csv`]

## B2. N06 engineered representation

Each EEG/EOG channel contributes six log absolute band powers, six relative powers, six time-domain features, skewness, and kurtosis: 20 features per channel, 60 total. Raw values and per-record median/IQR normalized copies (clipped to ±20) give 120. EMG adds median, IQR, and 90th percentile in raw and normalized form. Five-position context (center and ±2 epochs), ±15-epoch rolling mean/SD for band features, and a truncation flag produce **775 features**. Context never crosses invalid gaps.

The saved balanced LightGBM uses 250 estimators, learning rate 0.06, 31 leaves, minimum 100 child samples, max-bin 63, colsample 0.9, L2=1, deterministic column-wise training, and seed 20260923.

## B3. C01 waveform model

C01 accepts `(3,3000)` raw µV epochs from Fpz-Cz, Pz-Oz, and horizontal EOG. Each recording/channel is normalized by median/IQR and clipped to ±10. A convolutional encoder produces a 2,048-dimensional embedding; a one-layer 128-unit unidirectional LSTM classifies sequences of 20 epochs. Runs split at recording and invalid-gap boundaries, hidden state resets per sequence, and padded final targets are masked.

Training uses Adam, weighted cross-entropy, batch size 16, gradient clipping 5, at most 18 epochs, patience 4, and seed 42. Five-fold OOF Macro-F1 was **0.743750**; per-class F1 was 0.8887/0.4859/0.8148/0.7288/0.8006. Exact original-validation MSP-60/Naive scores were 0.722056/0.732277. C02 gated attention scored lower (OOF 0.740743) and was rejected.

E13 retained the same checkpoints and sequence length but used stride 10 and averaged repeated-epoch probability vectors. OOF Macro-F1 rose to 0.749492. The two RC1 CNN views are therefore inference views of the same trained models, not independent model families.

## B4. Frozen RC1 specification

> **FINAL_CLASSIFIER_RC1**  
> `0.60 × N06 + 0.20 × C01 stride-20 + 0.20 × C01 overlap stride-10`  
> then `P(N3) × 0.60`, five-class renormalization, and argmax.  
> Class order: Wake, N1, N2, N3, REM.  
> Scientific freeze SHA-256: `c6e1f71f310999be5a629c37884544cfb2f643f8fa1e09231544a5baa213a010`.

## B5. Development results

| Historical label-window-dependent evaluation | Macro-F1 | Kappa | Wake | N1 | N2 | N3 | REM | SC / ST |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development MSP-60 | 0.775264 | 0.742268 | 0.8846 | 0.5447 | 0.8518 | 0.7225 | 0.8727 | 0.7720 / 0.8030 |
| Development Naive | 0.784542 | 0.768642 | 0.9216 | 0.5470 | 0.8514 | 0.7276 | 0.8752 | 0.7824 / 0.8030 |

MSP subject Macro-F1 median was 0.7481, range 0.6020–0.9078; Naive had the same median and range. Between-subject variation is large relative to many incremental model gains.

## B6. Historical sealed evaluation

RC1 was frozen before this evaluation. It covered 20 subjects and 40 recordings, and its result did not modify RC1.

| Historical label-window-dependent sealed evaluation | Epochs | Macro-F1 | Kappa | Wake | N1 | N2 | N3 | REM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MSP-60 | 41,711 | **0.7963577556** | 0.761246 | 0.9005 | 0.5690 | 0.8539 | 0.7985 | 0.8599 |
| Naive | 49,317 | **0.8051740701** | 0.791642 | 0.9409 | 0.5724 | 0.8542 | 0.7987 | 0.8597 |

MSP support was Wake 8,614; N1 4,727; N2 17,509; N3 3,894; REM 6,967. N3 was 9.34% of scored epochs. SC/ST MSP Macro-F1 was 0.7819/0.8315. Subject median was 0.7835, range 0.6145–0.8687. [Internal evidence: Step 4 / `final_classifier_results.json`]

## B7. Historical versus deployment-faithful preprocessing

Historical N06 used label-window normalization and context; fresh inference normalizes and creates context over the full EDF. Historical CNN inputs were normalized and sequenced within the Naive window; deployment normalizes and sequences the full EDF. The mismatch audit found:

| Quantity | Result |
|---|---:|
| Historical MSP rows | 41,711 |
| Fresh full-EDF rows | 92,500 |
| Exact shared MSP keys | 41,711 |
| Final RC1 mean absolute ΔP | 0.0373 |
| Final RC1 P95 absolute ΔP | 0.1895 |
| Final RC1 maximum absolute ΔP | 0.6281 |
| Shared-epoch argmax changed | 9.52% |

Raw N06 engineered features matched, but normalization populations, feature-context boundaries, CNN waveform normalization, and sequence alignment differed. Reconstructing historical conditions reproduced the stored probabilities to about `1.11e-16` maximum difference with zero argmax changes. The weights were intact.

## B8. Deployment-faithful diagnostic

Fresh full-EDF RC1 inference, aligned afterward to the same 41,711 MSP keys, produced pooled Macro-F1 **0.7811876404**. This is a **deployment-faithful full-EDF diagnostic on the same sealed recordings**, not a second independent test. It is the conservative point estimate for executing the frozen inference contract without labels.

## B9. Test-access transparency and RC2 rejection

| Confirmed local sealed-label event | Purpose | Production consequence |
|---|---|---|
| RC1 one-shot evaluation | Formal frozen estimate | RC1 unchanged |
| Five-subject diagnostic | Secondary judging-style check | No tuning |
| RC2 confirmatory evaluation | Test frozen calibration candidate | RC2 rejected |
| Mismatch audit | Diagnose fresh/historical probability discrepancy | Documentation clarified; RC1 unchanged |

Cross-fitted development calibration initially improved MSP from 0.775264 to 0.787567 and Naive from 0.784542 to 0.795369. Its frozen one-shot full-EDF RC2 result was 0.762864, versus fresh upstream RC1 0.781188 in the same run. RC2 was rejected; factors were not retuned and no later experiment entered production.

## B10. Deployment validation

The release ran after isolation from hypnograms, window files, split metadata, and development/test probability artifacts. Determinism across fresh processes was exact in the validated environment. Six unlabeled EDFs completed, and all eight failure cases behaved as specified: missing Pz-Oz, missing EMG, wrong rate, empty EDF, corrupt EDF, 20-second file, 65-second partial-tail behavior, and reordered channels.

| Recording | Duration | Epochs | Wall time | Peak RSS |
|---|---:|---:|---:|---:|
| ST7011J0 | 9.97 h | 1,196 | 7.27 s | 1,117.7 MB |
| SC4001E0 | 22.08 h | 2,650 | 10.25 s | 1,152.6 MB |

The canonical CSV contains recording ID, epoch index, onset, duration, predicted stage, quality flags, model version, and five probabilities. Probabilities are finite, nonnegative, normalized, and consistent with argmax.

## B11. Human inter-scorer context

Expert staging is standardized but not perfectly reproducible. Selected studies report overall agreement around 80.6–86.4% and overall kappa around 0.714–0.798, with major differences in reference construction [3–7]. N1 is consistently least or among least reproducible: Arnal reports N1 F1 49.7%; Rosenberg reports 63.0% agreement; Danker-Hopfe reports the lowest stage kappa for N1 under both R&K and AASM; Nikkonen reports N1 Fleiss κ 0.406; and Lee’s meta-analysis reports pooled N1 κ 0.24.

| Stage | RC1 historical sealed F1 | Human-literature context |
|---|---:|---|
| Wake | 0.9005 | Generally highly reproducible |
| N1 | 0.5690 | Consistently weakest or among weakest |
| N2 | 0.8539 | More reproducible than N1 |
| N3 | 0.7985 | More reproducible than N1; scoring-standard mapping matters |
| REM | 0.8599 | Generally highly reproducible |

These studies provide context, not a directly comparable benchmark or universal ceiling for Sleep-EDF Macro-F1. Accuracy, F1, Cohen κ, Fleiss κ, pairwise agreement, and scorer-versus-consensus agreement are not interchangeable. RC1 is not claimed to be human-level.

# Part C — Sleep Quality Index

## C1. Definition and temporal contract

`sqi-1.0.0` is an interpretable objective composite derived from a hypnogram. It was created because the dataset contains expert stage labels but no direct sleep-quality target. It is not PSQI, subjective assessment, diagnosis, or a clinically validated universal score.

TIB is the supplied recording/window span. Qualifying onset is the first sleep epoch whose following ten-minute horizon is at least 50% sleep; final awakening uses the mirrored preceding rule. SPT spans qualifying onset through final qualifying sleep. TST is N1+N2+N3+REM inside SPT; WASO is Wake inside SPT; SOL is time from file start to qualifying onset. Stage percentages use TST; REM latency starts at qualifying onset; SFI counts adjacent stage changes per SPT hour. Thus SOL is conventionally interpretable only when file start approximates lights-off.

## C2. Formula

\[
SQI=100(0.25C_{eff}+0.20C_{cont}+0.20C_{dur}+0.15C_{frag}+0.10C_{lat}+0.10C_{arch})
\]

where `C_arch = mean(C_N3%, C_REM%, C_REM-latency)`.

| Component | Weight | Raw quantity |
|---|---:|---|
| Efficiency | 25% | TST / supplied-span TIB proxy |
| Continuity | 20% | WASO |
| Duration | 20% | TST hours |
| Fragmentation | 15% | stage transitions per SPT hour |
| Latency | 10% | SOL |
| Architecture | 10% | mean of N3%, REM%, REM latency maps |

## C3. Component maps

| Map | Exact knots `(raw → score)` | Evidence status summary |
|---|---|---|
| Duration, h | 0→0, 5→0, 7→1, 9→1, 10→0.7, ≥10→0.7 | mixed cited/assumption |
| Efficiency | 0→0, 65%→0, 85%→1, above→1 | literature-informed/project |
| Latency, min | 0→1, 15→1, 60→0, above→0 | mixed |
| Continuity/WASO, min | 0→1, 20→1, 60→0, above→0 | project-defined |
| Fragmentation, /h | 0→1, 10→1, 30→0, above→0 | project-defined |
| N3, % TST | 0→0, 13→1, 23→1, 40→0.5, above→0.5 | mixed |
| REM, % TST | 0→0, 5→0, 20→1, 25→1, 40→0.5, above→0.5 | mixed |
| REM latency, min | 0→0, 60→1, 120→1, 240→0, above→0 | project-defined |

Across the 22 nontrivial thresholds, 7 are directly cited, 4 are literature-informed project assumptions, and 11 are project assumptions. They are not all clinical thresholds. Weights were frozen before validation; two map knots were moved before validation. Arithmetic aggregation was retained because geometric aggregation collapsed 89/197 reference nights to zero whenever any component was zero.

## C4. Refusal branch and descriptive bands

Five conditions produce four reason codes: window under 180 minutes (`window_too_short`); unscored fraction above 25% (`unknown_fraction`); no qualifying onset/final awakening (`no_sleep`); SPT under 60 minutes or coverage below 70% (`spt_coverage_low`); and TST≤0 (`no_sleep`). One deployment-faithful reference recording was refused. T07 contained 960 Wake predictions and safely returned `no_sleep` with null score/components.

Dashboard bands—poor <50, fair 50–<70, good 70–<85, excellent ≥85—are descriptive and nonclinical.

## C5. Validation programme

| Study | Population | Retained result | Interpretation |
|---|---|---|---|
| V1 temazepam | 22 paired ST subjects | total +10.05; CI 4.38–16.52; p=.005 | primary direction passed; component directions mixed |
| V2 age | 78 SC subjects | SQI rho −0.691 | total direction passed; TST/duration did not |
| V3 weight sensitivity | 197 nights, 1,000 draws | median rho .998; p5 .994; min .986 | ranking stable under tested perturbations |
| V4 TIB proxy | 197 nights | MAE 8.13; bias −6.87 | file-span proxy materially affects score |
| V5 old stage propagation | 158 nights | MAE 7.54; bias −0.92; rho .883 | historical evidence, superseded for shipping |

Eight traceable V1/V2 directional rows were unconfirmed. “Unconfirmed” does not automatically mean statistically disproven. Transition entropy was rejected because it was redundant with SFI (rho 0.978) and reproduced poorly from predicted stages (rho 0.541). N3 consolidation captured distinct information but reproduced at rho 0.504, below its 0.60 rule. Respiration was excluded because retained availability/calibration evidence was insufficient and no validated SpO₂ path existed.

## C6. Synthetic and sensitivity findings

| Eight-hour synthetic sequence | SQI |
|---|---:|
| All N2 | 90.00 |
| All N3 | 91.67 |
| All REM | 91.67 |
| Alternating Wake/N1 | 10.00 |

These cases expose a construct limitation: architecture contributes only 10%, while homogeneous sleep scores strongly on several other components. They are not physiological validation. Weight sensitivity shows rank stability, not construct validity.

## C7. Deployment-faithful RC1-to-SQI propagation

Frozen RC1 ran label-free on full EDFs for 80 development subjects and 157 recordings, with zero sealed overlap. Labels were read only afterward; one reference refusal left 156 pairs.

| Metric | Result |
|---|---:|
| MAE | **5.353** |
| Median absolute error | 3.333 |
| RMSE | 7.501 |
| Mean bias, RC1−reference | **+1.743** |
| Spearman | **0.881** |
| Pearson | 0.942 |
| Bland–Altman limits | **−12.603 to +16.089** |
| Maximum absolute error | 22.234 |

Correlation is not agreement. Component propagation was strongest for efficiency and latency, while continuity, duration, fragmentation, and architecture were more vulnerable. The largest disagreements were driven mainly by Wake boundaries, SPT, WASO, and onset; architecture was secondary.

![Bland–Altman plot for RC1-derived versus reference SQI](artifacts/RC1_SQI_PROPAGATION/rc1_sqi_bland_altman.png)

*Figure 1. Measured software analysis: Bland–Altman differences for 156 paired development recordings. Source: retained RC1 SQI propagation artifact.*

![Reference versus RC1-derived SQI](artifacts/RC1_SQI_PROPAGATION/rc1_sqi_scatter.png)

*Figure 2. Measured software analysis: reference versus RC1-derived SQI with identity line. Correlation does not imply agreement.*

## C8. SQI limitations

There is no direct quality target. Weights and many knots are project choices; bands are nonclinical. TIB equals file span and SOL assumes recording start near lights-off; efficiency plus latency carry 35% of the score. Hard-stage uncertainty is ignored, no confidence interval is produced, components are correlated, homogeneous-stage sequences can score highly, subjective quality and respiratory/SpO₂ data are absent, and classifier boundary errors propagate. Clinical validation has not been performed.

# Part D — Hardware translation

## D1. Status and architecture

**Concept only. No physical prototype was built.** The objective is to translate lessons from clinical PSG toward a lower-burden future system:

| Stage | Proposed element | Evidence maturity |
|---|---|---|
| Sensing | Fpz-to-mastoid EEG; outer-canthi horizontal EOG; opposite-mastoid bias | future concept |
| Acquisition | ADS1299-4, 24 bit, 250 SPS | datasheet-supported design |
| Processing | anti-aliasing and rational resampling to 64 Hz | software evidence; physical implementation open |
| Transport | nRF52840 BLE, two 24-bit channels | datasheet/calculation |
| Inference | future reduced-input model on phone | research prototype/concept |
| Output | hypnogram → `sqi-1.0.0` | software path established; wearable path unvalidated |

## D2. RC1 incompatibility and reduced-input evidence

| Signal | Frozen RC1 | Proposed wearable | Compatible? |
|---|---|---|---|
| EEG 1 | exact Fpz-Cz | Fpz-mastoid | No |
| EEG 2 | exact Pz-Oz | absent | No |
| EOG | exact horizontal EOG | outer-canthi concept | Related but unvalidated |
| EMG | submental EMG | absent | No |

**RC1 cannot be deployed directly on the proposed wearable.** A separately trained and validated reduced-input model is required.

Five-fold **software ablations on clinical PSG** found R0 full inputs 0.7726 OOF Macro-F1; R1 Fpz-Cz+EOG 0.7537; Fpz-Cz alone 0.7028; and Fpz-Cz+EMG 0.7223. Retrospective resampling of R1 gave 0.7543 at 64 Hz and 0.7442 at 32 Hz. R1 is a **RESEARCH_PROTOTYPE**, not frozen, integrated, trained on Fpz-mastoid, or validated on wearable recordings.

## D3. Engineering estimates and maturity

Calculated average current is **7.56 mA**. With an assumed 400 mAh pack, `400/7.56 = 52.9 h`, reported as approximately **53 h**. Battery life was not measured. Raw payload is 384 B/s; BLE throughput was not measured. The full R1 model is 9.74 MiB and does not fit the nominal 1 MiB MCU flash. Phone inference is proposed; it was not demonstrated. No priced BOM total is verified.

| Evidence tier | Example |
|---|---|
| Measured software | montage and sampling ablations on clinical PSG |
| Datasheet | AFE/MCU capabilities |
| Calculated | power, battery, payload rate |
| Future concept | comfort, wearable signal quality, phone inference |

Physical prototype: no. Human wearable recordings: 0. Measured battery life: no. Target BLE test: no. On-device inference: no. Electrical safety, EMC, biocompatibility, clinical validation, and regulatory review have not occurred.

# Part E — Reproducibility and release

## E1. Portability gate and integrity

Final gate: **READY_WITH_LIMITATIONS**. The original integration config embedded three workstation paths. A path-only patch changed `integration/sleep-dashboard-handoff/config/model.json` to relative references. `predict_final.py` was unchanged; its original and portable SHA-256 are both `2ba652f2c7d02e547603b3e4378c8578af96871eb11d1076f2971491739746cd`. The scientific freeze and all six model hashes remained unchanged.

| Verification | Result |
|---|---|
| Runtime-critical absolute project paths | 3 before; 0 after |
| Outside-root imports and model loads | PASS |
| Staged T02 inference | PASS, 960 epochs |
| Pre/post prediction equivalence | max ΔP 0.0; 0 stage differences |
| Directory mode | PASS, 6/6 unlabeled EDFs |
| SQI outside root | PASS; exact T02 value 34.23283614244784 |
| Secrets | 0 |
| EDFs in whitelist | 0 |

[Internal evidence: Step 8B / `portable_release_manifest.json`]

## E2. Validated environment and commands

Validated environment: Windows, Python 3.12.14, PyTorch 2.6.0+cu126, CUDA 12.6, NVIDIA GeForce RTX 3050 Laptop GPU, NumPy 2.5.3, pandas 3.0.1, SciPy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0, pyEDFlib 0.1.42, MNE 1.13.2, and joblib 1.6.0. CPU is **not validated and is an unsupported release path**. CUDA is the validated path, not a conceptual statement that the models inherently require only GPU execution.

Single EDF:

```bash
python predict_final.py --input <recording.edf> --output <predictions.csv>
```

Directory:

```bash
python predict_final.py --input <edf_directory> --output <output_directory>
```

SQI CLI, after the integration adapter maps RC1 headers to the SQI schema:

```bash
python sqi.py <predictions.csv> [out.csv]
```

## E3. Reproducibility levels

| Item | Status | Main limitation |
|---|---|---|
| Artifact integrity | Reproducible | trusted serialized models |
| RC1 inference | Reproducible in validated CUDA environment | strict input contract; CPU unsupported |
| SQI | Reproducible | requires conforming mapped CSV |
| Dashboard ingestion | Reproducible in validated environment | CUDA environment required |
| N06 training | Partially scripted | dataset/annotations and clean replay external |
| C01 training | Partially scripted | dataset/cache/environment and clean replay external |
| Historical development evaluation | Partially scripted | labels and historical windows required |
| Historical sealed evaluation | Artifact-only/package-unsafe | protected labels deliberately excluded |
| Physical wearable | Not reproducible | no prototype exists |

Seven gaps remain: CPU support; partial N06/C01 training reproduction; protected sealed evaluation; final archive extraction testing; unretained minimum CUDA-driver contract; absence of a physical wearable; and dependency installation outside the staged tree.

# Part F — Negative results and decision discipline

The following table includes measured, traceable rejections. It demonstrates that complexity was not retained merely because it produced a plausible narrative.

| Domain | Experiment | Hypothesis / measured result | Decision and reason |
|---|---|---|---|
| Classifier | Untrimmed training | MSP 0.6736 vs Naive 0.6821 | Reject; weaker primary and robustness results |
| Classifier | MSP-60 training | MSP 0.6785 vs Naive-trained 0.6821 | Reject; removed sleep without gain |
| Classifier | Normalized-only | MSP 0.7297 vs raw+normalized 0.7356 | Reject; discarded useful absolute information |
| Classifier | Viterbi N07 | 0.7508 vs N06 0.7590 | Reject; smoothing reduced accuracy |
| Classifier | Age/sex N08 | 0.7546 vs N06 0.7590 | Reject; no gain |
| Classifier | C02 gated attention | OOF 0.7407 vs C01 0.7437 | Reject; added complexity without improvement |
| Classifier | Focal loss R04 | OOF 0.7119; N1 F1 0.4399 | Reject; substantial degradation |
| Classifier | BiGRU-40 R03 | OOF 0.7265 | Reject; below C01 |
| Classifier | N1 scaling E12 | Did not clear retained complexity threshold | Reject/neutral |
| Classifier | Transition prior R07 | Every nonzero predeclared λ reduced Macro-F1 | Reject; smoother was not more accurate |
| Classifier | RC2 calibration | Development gain; one-shot full-EDF 0.7629 vs RC1 0.7812 | Reject for deployment; no refit |
| SQI | Geometric aggregation | 89/197 reference nights collapsed to zero | Reject; brittle zero behavior |
| SQI | Transition entropy | rho .978 with SFI; propagation rho .541 | Reject; redundant and unstable |
| SQI | N3 consolidation | propagation rho .504 below .60 rule | Reject |
| SQI | Respiration | incomplete availability/calibration; no SpO₂ | Exclude from shipping index |
| SQI | Confidence component | confounded with fragmentation; reference comparison unavailable | Reject |
| SQI | Latent subjective index | no subjective target in Sleep-EDF | Not estimable |
| SQI | Restoration-heavy weights | no validated target; conflicts with frozen rationale | Reject |
| SQI | Cycle extension as headline | cycle rho .640; regularity rho .330 | Keep non-shipping |
| SQI | Temazepam architecture/N3 directions | intervals included zero | Direction unconfirmed |
| SQI | Age TST decline | rho −.044, p=.70 | Direction unconfirmed |
| Hardware | Fpz-Cz only | 0.7028; −0.0698 vs R0 | Reject as preferred reduced input |
| Hardware | Fpz-Cz+EMG | 0.7223; 0.031 below R1 | Reject |
| Hardware | 32 Hz | 0.7442; −0.0095 and lost high-band family | Reject |
| Hardware | Naive decimation with 50 Hz interference | severe Macro-F1 loss | Require anti-aliasing |
| Hardware | Full R1 on MCU | 9.74 MiB vs 1 MiB flash | Reject MCU deployment |
| Hardware | Compact MCU variant | 1.06 MB, F1 .7172 | Reject; still too large and less accurate |
| Hardware | Respiration/SpO₂ | evidence incomplete; no validated path | Exclude |

# System-level limitations

**Data.** Sleep-EDF is historical, cohort-specific, and R&K-derived. SC/ST acquisition differs, especially EMG. External populations and modern acquisition systems may behave differently.

**Classifier.** Historical scores depend on label-window preprocessing and must not be treated as guaranteed full-EDF performance. N1 remains weak. Input names, montage, rate, and units are strict; arbitrary equivalent montages are unsupported. CPU is not a supported release path.

**SQI.** The score is a nonclinical project composite. File boundaries affect TIB, SOL, efficiency, and continuity. Classifier probabilities are not propagated into an uncertainty interval.

**Hardware.** The design is conceptual. There are no wearable recordings, physical tests, measured battery/BLE results, or safety evidence. A future classifier is required.

> **Claims explicitly not made**
>
> - RC1 is human-level or reaches a universal human ceiling.
> - `sqi-1.0.0` is clinically validated, diagnostic, or a PSQI replacement.
> - RC1 runs on the proposed wearable.
> - The calculated 53-hour battery estimate was measured.
> - The wearable was validated on humans.
> - The release supports CPU inference.
> - The organizer performs the same label-derived MSP crop.

# Conclusion

The completed system comprises a frozen five-stage classifier, deterministic CUDA inference, an objective and auditable SQI, quantified full-EDF classifier-to-SQI propagation, dashboard integration, and a clearly separated hardware concept. The strongest deployment claim is the label-free full-EDF pipeline and its 0.7812 diagnostic Macro-F1 on the retained sealed recordings; historical controlled window scores remain useful when properly qualified.

Future work requires a target-montage reduced classifier, physical prototype, prospective wearable data, safety and usability testing, broader external validation, clinical validation of any quality construct, and uncertainty-aware SQI reporting.

# References

1. Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*. 2000;101:e215–e220. Sleep-EDF Expanded v1.0.0, DOI: 10.13026/C2X676.
2. American Academy of Sleep Medicine. Summary of Updates in the AASM Manual for the Scoring of Sleep and Associated Events, version 2.1.
3. Arnal PJ, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43(11):zsaa097. DOI: 10.1093/sleep/zsaa097.
4. Rosenberg RS, Van Hout S. The AASM inter-scorer reliability program: sleep stage scoring. *J Clin Sleep Med*. 2013;9:81–87. DOI: 10.5664/jcsm.2350.
5. Danker-Hopfe H, et al. Interrater reliability for sleep scoring according to R&K and the new AASM standard. *J Sleep Res*. 2009;18:74–84. DOI: 10.1111/j.1365-2869.2008.00700.x.
6. Nikkonen S, et al. Multicentre sleep-stage scoring agreement in the Sleep Revolution project. *J Sleep Res*. 2024;33:e13956. DOI: 10.1111/jsr.13956.
7. Lee YJ, et al. Interrater reliability of sleep stage scoring: a meta-analysis. *J Clin Sleep Med*. 2022;18:193–202. DOI: 10.5664/jcsm.9538.
8. Texas Instruments. ADS1299 family low-noise 24-bit biopotential ADC datasheet. Used only for concept-level hardware specifications.
9. Nordic Semiconductor. nRF52840 product specification. Used only for concept-level MCU/BLE capabilities.

# Appendices

## Appendix A — Classifier experiment evidence

The authoritative 30-entry timeline, evidence classification, parent relationships, populations, and source paths are in `artifacts/MASTER_REPORT_BUILD/classifier_experiment_ledger.csv`. Teammate handoff-only values are excluded from headline comparisons.

## Appendix B — Support and confusion evidence

Final support tables are in `final_support_tables.csv`; retained development and sealed confusion matrices are in `final_classifier_results.json`. Rows are true classes and columns are predicted classes.

## Appendix C — Test-access ledger

`final_test_access_ledger.csv` distinguishes four confirmed local sealed-label events from a teammate-declared test whose identity is unverified. Later RC2 research did not enter production.

## Appendix D — SQI formal contract

`sqi_final_contract.json` records input validation, temporal definitions, exact knots, formula, refusal rules, bands, hashes, and limitations. `sqi_negative_results.csv` contains ten retained negative-result rows.

## Appendix E — Hardware traceability

`hardware_final_contract.json`, `hardware_requirements_traceability.csv`, and `hardware_gap_matrix.csv` separate measured software, datasheet, calculated, assumed, and future evidence.

## Appendix F — Release hashes and contracts

`portable_release_manifest.json` identifies the scientific freeze, portable code, unchanged model hashes, path audit, and equivalence test. `classifier_input_contract.json` and `classifier_output_contract.json` define judge-facing I/O invariants. The submission whitelist contains no dataset EDFs or sealed labels.
