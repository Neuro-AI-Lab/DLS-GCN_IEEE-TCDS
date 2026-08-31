import os
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

from preprocessing import (
    set_global_seed,
    apply_lowpass_filter_array,
    fit_zscore_train,
    apply_zscore,
    apply_car,
    load_subject_all_data,
    stratified_split_60_10_10_per_class,
)
from model import ProposedFinal


def evaluate_model(model, data_loader, class_names, criterion, device, show_classwise=True):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    class_correct = {name: 0 for name in class_names}
    class_total   = {name: 0 for name in class_names}

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            if show_classwise:
                for i in range(len(y_batch)):
                    lab = int(y_batch[i].item())
                    class_total[class_names[lab]] += 1
                    if int(preds[i].item()) == lab:
                        class_correct[class_names[lab]] += 1

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    if show_classwise:
        print(f"\nOverall Accuracy: {acc:.4f}")
        for name in class_names:
            denom = class_total[name]
            a = (class_correct[name] / denom) if denom > 0 else 0.0
            print(f"  - {name}: {a:.4f} ({class_correct[name]}/{denom})")

    return acc, avg_loss

def save_learning_curves(save_dir, train_acc, val_acc, train_loss, val_loss):
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(train_acc, label="Train Accuracy")
    ax1.plot(val_acc, label="Validation Accuracy")
    ax1.set_title("Training & Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(train_loss, label="Train Loss")
    ax2.plot(val_loss, label="Validation Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "learning_curves.png"), dpi=200)
    plt.close(fig)


def save_confusion_and_report(model, test_loader, class_names, device, save_dir, tag="test"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            outputs = model(X_batch)
            preds = outputs.argmax(1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".3f",
        cmap="Greys",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_title(f"Confusion Matrix ({tag})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"confusion_matrix_{tag}.png"), dpi=250)
    plt.close(fig)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(save_dir, f"classification_report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    return report


def train_model(model, train_loader, val_loader, class_names, device, num_epochs, save_dir,
                lambda_sparse: float = 1e-4):
    """
    lambda_sparse : lag-specific masked residual에 대한 L1 sparsity 가중치 (식 79).
                    L = L_cls + lambda_sparse * L_sparse.
                    0으로 두면 sparsity regularization 비활성화.
    """
    os.makedirs(save_dir, exist_ok=True)

    log_txt_path = os.path.join(save_dir, "train_log.txt")

    initial_lr = 1e-3
    final_lr   = 1e-6
    power      = 2.0
    weight_decay = 1e-2

    def lr_lambda(epoch):
        progress = epoch / num_epochs
        return (final_lr / initial_lr) + (1 - final_lr / initial_lr) * (1 - progress) ** power

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device.type == "cuda")

    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_acc = -1.0
    best_val_loss_at_best = float("inf")
    best_state = None
    best_epoch = -1

    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write("Training log\n")
        f.write(f"lambda_sparse = {lambda_sparse}\n")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    out = model(X_batch)
                    # L = L_cls + lambda_sparse * L_sparse  (식 79)
                    loss = criterion(out, y_batch)
                    if lambda_sparse > 0:
                        loss = loss + lambda_sparse * model.sparsity_l1()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(X_batch)
                loss = criterion(out, y_batch)
                if lambda_sparse > 0:
                    loss = loss + lambda_sparse * model.sparsity_l1()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

        scheduler.step()

        train_acc = correct / total if total > 0 else 0.0
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        val_acc, val_loss = evaluate_model(
            model, val_loader, class_names, criterion, device, show_classwise=False
        )

        train_acc_hist.append(train_acc)
        train_loss_hist.append(train_loss)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

        lr = scheduler.get_last_lr()[0]
        line = (
            f"Epoch [{epoch+1}/{num_epochs}] | LR: {lr:.8f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        print(line)

        with open(log_txt_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss_at_best):
            best_val_acc = val_acc
            best_val_loss_at_best = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    save_learning_curves(save_dir, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist)

    if best_state is None:
        raise RuntimeError("No best model state was selected.")

    return best_state, best_epoch, float(best_val_acc), float(best_val_loss_at_best)

def run_one_subject_one_seed(subject_id, seed, num_epochs, batch_size, 
                             root_save_dir, train_dir, val_dir, test_dir, 
                             fs, num_workers, pin_memory, device,
                             n_train_per_class=60, n_val_per_class=10, n_test_per_class=10,
                             lambda_sparse: float = 1e-4):
    
    save_dir = os.path.join(root_save_dir, f"subject_{subject_id:02d}", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    set_global_seed(seed)
    X_all, y_all, class_names = load_subject_all_data(
        subject_id, train_dir, val_dir, test_dir
    )

    # 시퀀스를 자르지 않고 전체(T=795)를 사용한다.
    X_all = apply_car(X_all)

    idx_train, idx_val, idx_test = stratified_split_60_10_10_per_class(
        X_all, y_all,
        seed=seed,
        n_train_per_class=n_train_per_class,
        n_val_per_class=n_val_per_class,
        n_test_per_class=n_test_per_class
    )
    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val,   y_val   = X_all[idx_val],   y_all[idx_val]
    X_test,  y_test  = X_all[idx_test],  y_all[idx_test]

    with open(os.path.join(save_dir, "split_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"subject={subject_id}, seed={seed}\n")
        f.write(f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}\n\n")
        for c, name in enumerate(class_names):
            f.write(
                f"class {c} ({name}) -> "
                f"train={(y_train == c).sum()}, "
                f"val={(y_val == c).sum()}, "
                f"test={(y_test == c).sum()}\n"
            )

    X_train = apply_lowpass_filter_array(X_train, fs=fs)
    X_val   = apply_lowpass_filter_array(X_val,   fs=fs)
    X_test  = apply_lowpass_filter_array(X_test,  fs=fs)

    mean, std = fit_zscore_train(X_train)
    X_train = apply_zscore(X_train, mean, std)
    X_val   = apply_zscore(X_val,   mean, std)
    X_test  = apply_zscore(X_test,  mean, std)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory 
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    model = ProposedFinal(
        num_nodes=64,
        num_classes=len(class_names),
    ).to(device)

    model.init_adjacency(X_train)

    best_state, best_epoch, best_val_acc, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        lambda_sparse=lambda_sparse,
    )

    # best 가중치는 용량 절약을 위해 파일로 저장하지 않고 메모리에서 바로 복원한다.
    model.load_state_dict(best_state)
    model.to(device)

    test_criterion = nn.CrossEntropyLoss()
    test_acc, test_loss = evaluate_model(
        model,
        test_loader,
        class_names,
        test_criterion,
        device,
        show_classwise=True
    )

    save_confusion_and_report(
        model,
        test_loader,
        class_names,
        device,
        save_dir,
        tag="test"
    )

    return {
        "subject_id": subject_id,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "save_dir": save_dir,
    }
