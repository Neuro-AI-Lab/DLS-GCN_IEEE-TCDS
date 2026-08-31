# code_summery — 제안 모델(ProposedFinal) 학습 실행 최소 코드

`model_proposed_final.py` 기준으로 학습/평가 실행에 필요한 파일만 추린 self-contained 세트.

실행 흐름: `main.py` → `train_eval.py` → (`preprocessing.py`, `model.py`)

```
main.py           subject 15 × seed 10 루프를 돌리고 결과를 집계
  └ train_eval.py   1회 실행(= subject 1명 × seed 1개)의 전처리→학습→평가 전 과정
      ├ preprocessing.py   .mat 로딩 / 신호 전처리 / 데이터 분할
      └ model.py           제안 모델 정의
```

---

## `model.py` — 제안 모델 정의

EEG 64채널을 **노드**, 채널 간 WPLI 위상 동기화를 **엣지**로 보는 시공간 그래프 신경망.
`X (B, 64, 795)` → 클래스 로짓 `(B, 5)`. 파라미터 430,853개.

| 이름 | 역할 |
|---|---|
| `WPLIAdjacencyBuilder.build` | **그래프 생성.** train set에 Hilbert 변환을 걸어 채널쌍 WPLI(weighted phase-lag index)를 구하고, 채널마다 상위 16개 이웃만 남긴 대칭 binary 인접행렬 반환. 학습 전 1회만 실행 |
| `compute_k_adjacency` | 인접행렬에서 **정확히 k-hop인 엣지만** 뽑아낸 k-adjacency 생성 (k=0이면 단위행렬). hop별로 정보를 분리(disentangle)하는 역할 |
| `normalize_adjacency` | 대칭 정규화 $D^{-1/2}AD^{-1/2}$. hop 수가 커져도 스케일이 폭주하지 않게 함 |
| `_shift_time` | 시퀀스를 시간축으로 `lag`만큼 밀고 빈 곳을 0으로 채움. 시간 이웃(lag) 집계용 |
| `SimAM1d` | **무파라미터 어텐션.** 시간축 분산 기준으로 각 시점의 중요도를 계산해 재가중. 파라미터를 늘리지 않고 주의집중 효과만 얻음 |
| `ProposedAggregation` | **핵심 그래프 집계 블록.** lag들을 먼저 더한 뒤($\sum_\delta X_{t+\delta}$) 0/1/2-hop 각각에 대해 $S_k = \hat{A}_k + M^{(k)} \odot R_k$ 로 메시지 패싱하고 합산 → SimAM → BN → ELU. $R_k$는 학습가능한 residual로 **고정된 WPLI 그래프를 데이터에 맞게 보정**하는 부분이고, 마스크 $M^{(k)}$ 때문에 원래 엣지가 있던 자리만 수정됨 |
| `MultiScaleTemporalConv` | **시간축 특징 추출.** kernel 3·5 두 갈래 depthwise conv로 짧은/긴 시간 패턴을 동시에 보고, 1×1로 합친 뒤 SimAM + residual + 절반 downsample |
| `ProposedFinal` | 전체 모델 조립. 입력 절반 downsample → **dual-branch 집계**(A: lag −1,0,1 = 짧은 시간범위 / B: lag −4,−2,0,2,4 = 넓은 시간범위)를 채널축으로 concat → MS-TCL → DIP(로그 파워) + 시간 어텐션 두 가지 pooling을 concat → Linear 분류 |
| `ProposedFinal.init_adjacency` | 학습 **직전에** train 데이터로 그래프를 만들어 모델 버퍼에 심어주는 함수. 반드시 학습 전에 호출해야 함 |
| `ProposedFinal.sparsity_l1` | residual $R_k$의 L1 노름. 학습 손실에 더해져 **그래프 보정량이 불필요하게 커지지 않도록 억제** |
| `_dip` | 시간축 제곱평균의 로그(log power). EEG 대역 파워를 특징으로 뽑는 pooling |

## `preprocessing.py` — 데이터 로딩·전처리·분할

| 이름 | 역할 |
|---|---|
| `set_global_seed` | random / numpy / torch 시드 일괄 고정 (seed별 재현성 확보) |
| `_mat_to_xy` | MATLAB `epo` 구조체를 `(N, C, T)` 신호 배열 + 정수 라벨 + 클래스 이름으로 변환 (축 전치, one-hot → index) |
| `load_subject_all_data` | 한 subject의 Training/Validation/Test `.mat` 3개를 읽어 **전부 이어붙여** 반환. 원본 분할을 쓰지 않고 아래에서 seed별로 다시 나누기 위함 |
| `stratified_split_60_10_10_per_class` | 합쳐진 데이터를 seed 기반으로 섞어 **클래스마다** train 60 / val 10 / test 10개로 재분할. seed마다 다른 분할이 나오므로 10-seed 반복이 곧 반복 검증이 됨 |
| `apply_car` | Common Average Reference — 전 채널 평균을 빼서 공통 잡음 제거 |
| `apply_lowpass_filter_array` | 60Hz Butterworth 저역통과 (`filtfilt`, 위상 왜곡 없음) |
| `fit_zscore_train` / `apply_zscore` | 채널별 정규화 통계를 **train에서만** 계산해 val/test에 그대로 적용 (data leakage 방지) |

## `train_eval.py` — 1회 실행 단위의 학습·평가

| 이름 | 역할 |
|---|---|
| `run_one_subject_one_seed` | **이 파일의 엔트리.** 시드 고정 → 데이터 로딩 → CAR → 분할 → 저역통과 → z-score → DataLoader 구성 → 모델 생성 및 `init_adjacency` → 학습 → best 가중치 복원 → test 평가까지 한 번에 수행하고 결과 dict 반환 |
| `train_model` | 학습 루프. AdamW(lr 1e-3→1e-6 poly decay, wd 1e-2) + AMP 혼합정밀 + label smoothing 0.01, 손실은 $L_{cls} + \lambda \cdot L_{sparse}$. 매 epoch validation을 돌려 **val 정확도 최고 시점의 가중치를 메모리에 보관**(동률이면 val loss가 낮은 쪽) |
| `evaluate_model` | 주어진 loader로 정확도·평균 손실 계산. `show_classwise=True`면 클래스별 정확도까지 출력 |
| `save_learning_curves` | epoch별 train/val 정확도·손실 곡선을 `learning_curves.png`로 저장 |
| `save_confusion_and_report` | test 예측으로 행 정규화 혼동행렬 heatmap과 sklearn classification report를 파일로 저장 |

## `main.py` — 전체 프로토콜 실행

subject 1~15 × seed 10개 = 150회를 순차 실행하는 최상위 스크립트.
GPU/스레드 고정, 경로·하이퍼파라미터 상수 정의, subject 하나가 실패해도 다음으로 넘어가는 예외 처리,
그리고 subject별 평균 → 전체 평균/표준편차(ddof=1) 집계를 담당한다.

---

## 데이터

`Training set`, `Validation set`, `Test set` → 상위 폴더로 심볼릭 링크.
각 폴더에 `Data_Sample{1..15}.mat` (`epo_train` / `epo_validation` / `epo_test`) 필요.

## 실행

```bash
source ../admin/bin/activate
taskset -c 0-11 python main.py     # GPU는 main.py 상단 CUDA_VISIBLE_DEVICES="1" 로 고정
```

출력: `results/proposed_final/subject_XX/seed_YYYY/` 에 train_log, learning_curves,
confusion_matrix, classification_report. 최상위에 `ALL_SUBJECTS_SUMMARY.txt` 와
`RESULT_{평균}_{표준편차}.txt`.

## 주요 하이퍼파라미터

- epochs 200, batch 64, fs 256, AdamW lr 1e-3 → 1e-6 (poly power=2), weight_decay 1e-2
- CrossEntropy label_smoothing 0.01, `lambda_sparse` 1e-4 (masked residual L1)
- 입력은 자르지 않고 전체 T=795 사용
- 파라미터 430,853 / 보고된 full-protocol test_acc 0.8443 ± 0.0438
