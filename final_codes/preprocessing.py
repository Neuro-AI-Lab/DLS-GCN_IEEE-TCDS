import os
import random
import numpy as np
import scipy.io
import scipy.signal as signal
import torch


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def apply_lowpass_filter_array(X, fs=256, cutoff=60.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyq, btype="low")
    return signal.filtfilt(b, a, X, axis=-1).astype(np.float32)


def fit_zscore_train(X_train, eps=1e-8):
    mean = np.mean(X_train, axis=(0, 2), keepdims=True)
    std  = np.std(X_train, axis=(0, 2), keepdims=True)
    std = np.maximum(std, eps)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_zscore(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def apply_car(X):
    """
    Common Average Reference (CAR) 공간 필터
    X shape: (N, C, T) -> (Trials, Channels, Time)
    """
    common_average = np.mean(X, axis=1, keepdims=True)
    return (X - common_average).astype(np.float32)


def _mat_to_xy(epo_obj):
    x = getattr(epo_obj, "x", None)
    y = getattr(epo_obj, "y", None)
    class_names = getattr(epo_obj, "className", None)
    x = np.array(x)
    X = np.transpose(x, (2, 1, 0))
    y = np.array(y)
    y_idx = np.argmax(y, axis=0).astype(np.int64)
    if isinstance(class_names, np.ndarray):
        class_names = class_names.tolist()
    class_names = [str(c) for c in class_names]
    return X.astype(np.float32), y_idx.astype(np.int64), class_names



def load_subject_all_data(subject_id, train_dir, val_dir, test_dir):
    train_path = os.path.join(train_dir, f"Data_Sample{subject_id}.mat")
    val_path   = os.path.join(val_dir,   f"Data_Sample{subject_id}.mat")
    test_path  = os.path.join(test_dir,  f"Data_Sample{subject_id}.mat")

    dtr = scipy.io.loadmat(train_path, struct_as_record=False, squeeze_me=True)
    dva = scipy.io.loadmat(val_path,   struct_as_record=False, squeeze_me=True)
    dte = scipy.io.loadmat(test_path,  struct_as_record=False, squeeze_me=True)

    epo_train = dtr.get("epo_train", None)
    epo_val   = dva.get("epo_validation", None)
    epo_test  = dte.get("epo_test", None)

    Xtr, ytr, class_names = _mat_to_xy(epo_train)
    Xva, yva, _ = _mat_to_xy(epo_val)
    Xte, yte, _ = _mat_to_xy(epo_test)

    X_all = np.concatenate([Xtr, Xva, Xte], axis=0)
    y_all = np.concatenate([ytr, yva, yte], axis=0)
    return X_all, y_all, class_names


def stratified_split_60_10_10_per_class(X_all, y_all, seed, n_train_per_class=60, n_val_per_class=10, n_test_per_class=10):
    rng = np.random.RandomState(seed)

    classes = np.unique(y_all)

    idx_train_list = []
    idx_val_list = []
    idx_test_list = []

    for c in classes:
        cls_idx = np.where(y_all == c)[0]
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)

        idx_train_c = cls_idx[:n_train_per_class]
        idx_val_c   = cls_idx[n_train_per_class:n_train_per_class + n_val_per_class]
        idx_test_c  = cls_idx[n_train_per_class + n_val_per_class:
                              n_train_per_class + n_val_per_class + n_test_per_class]

        idx_train_list.append(idx_train_c)
        idx_val_list.append(idx_val_c)
        idx_test_list.append(idx_test_c)

    idx_train = np.concatenate(idx_train_list)
    idx_val   = np.concatenate(idx_val_list)
    idx_test  = np.concatenate(idx_test_list)

    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)
    return idx_train, idx_val, idx_test