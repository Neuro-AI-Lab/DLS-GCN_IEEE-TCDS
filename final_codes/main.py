import os
import numpy as np
import torch

from train_eval import run_one_subject_one_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 경로 (Data_Sample{subject}.mat 이 들어있는 폴더)
train_dir = os.path.join(BASE_DIR, "Training set")
val_dir   = os.path.join(BASE_DIR, "Validation set")
test_dir  = os.path.join(BASE_DIR, "Test set")

ROOT_SAVE_DIR = os.path.join(BASE_DIR, "results/proposed_final")
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

FS = 256
SEEDS = [42, 420, 777, 820, 2023, 2026, 2032, 3033, 2034, 3036]
SUBJECT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
N_TRAIN_PER_CLASS = 60
N_VAL_PER_CLASS   = 10
N_TEST_PER_CLASS  = 10
NUM_EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True
LAMBDA_SPARSE = 1e-4   # L = L_cls + lambda_sparse * ||masked residual||_1


if __name__ == '__main__':
    print("Using device:", device)
    all_rows = []

    for subject_id in SUBJECT_IDS:
        subj_rows = []

        for seed in SEEDS:
            try:
                print("\n" + "=" * 100)
                print(f"[RUN] subject={subject_id:02d}, seed={seed}")
                print("=" * 100)

                row = run_one_subject_one_seed(
                    subject_id=subject_id,
                    seed=seed,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    root_save_dir=ROOT_SAVE_DIR,
                    train_dir=train_dir,
                    val_dir=val_dir,
                    test_dir=test_dir,
                    fs=FS,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    device=device,
                    n_train_per_class=N_TRAIN_PER_CLASS,
                    n_val_per_class=N_VAL_PER_CLASS,
                    n_test_per_class=N_TEST_PER_CLASS,
                    lambda_sparse=LAMBDA_SPARSE,
                )

                subj_rows.append(row)
                all_rows.append(row)

            except Exception as e:
                print(f"[Fail] subject={subject_id:02d}, seed={seed} -> {repr(e)}")

        if len(subj_rows) > 0:
            accs = np.array([r["test_acc"] for r in subj_rows], dtype=np.float64)
            mean_acc = float(accs.mean())

            subj_dir = os.path.join(ROOT_SAVE_DIR, f"subject_{subject_id:02d}")
            os.makedirs(subj_dir, exist_ok=True)

            subject_summary_path = os.path.join(subj_dir, "SUBJECT_SUMMARY.txt")
            with open(subject_summary_path, "w", encoding="utf-8") as f:
                f.write(f"Sub {subject_id}:\n")
                f.write(f"Avg acc: {mean_acc:.4f}\n")
                for r in subj_rows:
                    f.write(f"seed {r['seed']}: {r['test_acc']:.4f}\n")

            print(f"[Subject {subject_id:02d}] mean_test_acc={mean_acc:.4f}")

    summary_path = os.path.join(ROOT_SAVE_DIR, "ALL_SUBJECTS_SUMMARY.txt")
    subject_means = []
    used_subject_ids = []

    with open(summary_path, "w", encoding="utf-8") as f:
        for subject_id in SUBJECT_IDS:
            subj_rows = [r for r in all_rows if r["subject_id"] == subject_id]
            if len(subj_rows) == 0:
                continue

            accs = np.array([r["test_acc"] for r in subj_rows], dtype=np.float64)
            mean_acc = float(accs.mean())
            subject_means.append(mean_acc)
            used_subject_ids.append(subject_id)

            f.write(f"Sub {subject_id}:\n")
            f.write(f"Avg acc: {mean_acc:.4f}\n")
            for r in subj_rows:
                f.write(f"seed {r['seed']}: {r['test_acc']:.4f}\n")
            f.write("\n")

    print(f"\n전체 요약 저장 완료: {summary_path}")

    # ── 최종 집계: subject별 평균의 평균 / 표준편차(ddof=1) ──
    # seed 10개를 모두 사용하므로 기존 '정리코드.py'의 top-10 seed 집계와 결과가 동일하다.
    if len(subject_means) > 0:
        sm = np.array(subject_means, dtype=np.float64)
        overall_mean = float(sm.mean())
        overall_std = float(sm.std(ddof=1)) if len(sm) > 1 else 0.0
        n_sub = len(sm)

        marker_path = os.path.join(
            ROOT_SAVE_DIR, f"RESULT_{overall_mean:.4f}_{overall_std:.4f}.txt"
        )
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"전체 {n_sub}명 기준\n")
            f.write(f"평균 정확도: {overall_mean:.4f}\n")
            f.write(f"표준편차: {overall_std:.4f}\n\n")
            for sid, m in zip(used_subject_ids, subject_means):
                f.write(f"Sub {sid}: {m:.4f}\n")

        print(f"[정리 완료] 평균 {overall_mean:.4f} / 표준편차 {overall_std:.4f}")
        print(f"[마커 파일] {marker_path}")
