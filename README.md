# [2026 IEEE TCDS Special Issue] Distance-Structured Lag-Sparse Graph Convolution with a Lag-Shared Adjacency Residual for Imagined Speech EEG Decoding

<div align="center">

**Seung Won Kim**<sup></sup>, **Dae Hyeon Kim**<sup></sup>, **Young-Seok Choi**<sup>*</sup>

<sup></sup>Department of Electronics and Communications Engineering, Kwangwoon University, Seoul, South Korea

[![Journal](https://img.shields.io/badge/IEEE%20TCDS-Special%20Issue%202026-00629b.svg)](https://cis.ieee.org/publications/t-cognitive-and-developmental-systems)
[![Status](https://img.shields.io/badge/status-under%20review-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📢 News
* **[Aug. 2026]** 📄 Our paper **"Distance-Structured Lag-Sparse Graph Convolution with a Lag-Shared Adjacency Residual for Imagined Speech EEG Decoding"** has been **submitted to the IEEE Transactions on Cognitive and Developmental Systems (TCDS) Special Issue**. The code will be officially released upon acceptance.

---

## 📝 Abstract

Electroencephalographic imagined-speech decoding must represent electrode interactions over time from a limited number of subject-specific trials. Unified spatial–temporal window graphs provide direct lagged electrode interactions, but a τ-frame residual adjacency scales as τ²N² for N electrodes.

We propose a **distance-structured lag-sparse graph convolutional network (DLS-GCN)**. Each branch represents temporal context by selected shifts and an N × N spatial operator instead of a τN × τN window graph. Exact-distance supports through two hops define distance-specific normalization and the support of adaptive residuals. One masked residual per distance component is shared across all temporal lags of a branch. This constraint permits exact factorization of the lag dimension; because the distance components are also summed before nonlinear processing, they can be folded into one effective graph product per branch without changing the output. A parameter-free temporal recalibration module and a depthwise multiscale temporal module form the remaining network.

Under protocol-matched subject-dependent evaluation, DLS-GCN obtains **0.8433 ± 0.0459** accuracy on Track 3 of the 2020 International BCI Competition and **0.4108 ± 0.0260** on the Thinking Out Loud dataset. The trained model has **430,853 parameters** and **166.88 M MACs** in the component-wise implementation; exact graph folding reduces evaluation to **160.39 M MACs**. Controlled analyses support the distance-structured prior and lag-shared residual as an effective accuracy–complexity trade-off.

---

## 📊 Experimental Results

Protocol-matched subject-dependent evaluation.

| Dataset | Accuracy |
| :--- | :---: |
| **2020 International BCI Competition (Track 3)** | **0.8433 ± 0.0459** |
| **Thinking Out Loud** | **0.4108 ± 0.0260** |

### Complexity

| Metric | Value |
| :--- | :---: |
| Parameters | 430,853 |
| MACs (component-wise implementation) | 166.88 M |
| MACs (exact graph folding, evaluation) | 160.39 M |

> **Key Findings:**
> 1. **Lag factorization.** Sharing one masked residual per distance component across all temporal lags of a branch permits exact factorization of the lag dimension, replacing the τN × τN window graph with an N × N spatial operator.
> 2. **Exact graph folding.** Because the distance components are summed before nonlinear processing, they fold into a single effective graph product per branch **without changing the output**, reducing evaluation cost from 166.88 M to 160.39 M MACs.
> 3. **Accuracy–complexity trade-off.** Controlled analyses support the distance-structured prior and the lag-shared residual as an effective trade-off between accuracy and complexity.

---

## 📦 Repository Structure

This repository contains the minimal, self-contained code required to reproduce training and evaluation on Track 3 of the 2020 International BCI Competition.

```
main.py             Full protocol runner (15 subjects x 10 seeds) and result aggregation
train_eval.py       One run (single subject, single seed): preprocessing, training, evaluation
preprocessing.py    .mat loading, signal preprocessing, seed-wise data splitting
model.py            Model definition
```

| File | Role |
|---|---|
| `model.py` | Model definition. EEG channels are treated as graph nodes and inter-channel WPLI phase synchronization as edges. Contains the WPLI graph builder, exact-hop *k*-adjacency, the dual-branch aggregation with lag-shared masked residuals, the parameter-free temporal recalibration module, and the depthwise multi-scale temporal module. Input `(B, 64, 795)` → class logits `(B, 5)`. |
| `preprocessing.py` | Loads the per-subject `.mat` files, concatenates them, and re-splits them per seed into 60/10/10 trials **per class**. Applies common average reference, a 60 Hz low-pass filter, and channel-wise z-scoring whose statistics are fitted on the training split only. |
| `train_eval.py` | One complete run: seeding → preprocessing → graph initialization → training (AdamW, polynomial LR decay, mixed precision, `L_cls + λ·L_sparse`) → best-validation checkpoint restoration → test evaluation, with learning curves, confusion matrix, and classification report written to disk. |
| `main.py` | Runs the full protocol over 15 subjects × 10 seeds and aggregates per-subject and overall accuracy. |

### Data

Place `Data_Sample{1..15}.mat` (containing `epo_train` / `epo_validation` / `epo_test`) in the following directories:

```
Training set/     Data_Sample1.mat ... Data_Sample15.mat
Validation set/   Data_Sample1.mat ... Data_Sample15.mat
Test set/         Data_Sample1.mat ... Data_Sample15.mat
```

### Quick Start

```bash
python main.py
```

Results are written to `results/proposed_final/subject_XX/seed_YYYY/`, with
`ALL_SUBJECTS_SUMMARY.txt` and `RESULT_{mean}_{std}.txt` at the top level.

### Key Hyperparameters

| Argument | Value | Description |
|---|---|---|
| `NUM_EPOCHS` | 200 | Training epochs |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `FS` | 256 | Sampling rate (Hz) |
| `LAMBDA_SPARSE` | 1e-4 | L1 weight on the masked residual adjacency |
| `SEEDS` | 10 seeds | Each seed defines a different 60/10/10 per-class split |
| `SUBJECT_IDS` | 1–15 | Subject-dependent evaluation |

Optimizer: AdamW, learning rate 1e-3 → 1e-6 (polynomial decay, power 2), weight decay 1e-2,
cross-entropy with label smoothing 0.01.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

The paper is currently under review at the IEEE Transactions on Cognitive and Developmental Systems (TCDS) Special Issue. A citation entry will be added upon acceptance.
