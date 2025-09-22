import torch
from sklearn import metrics
import numpy as np

LOG_FILE = "logs/validation_per_epoch.txt"
def log_epoch(epoch, training_loss,
              dev_fold_id, dev_mae, dev_rmse, dev_r2, dev_avg_loss,
              test_fold_id, test_mae, test_rmse, test_r2, test_avg_loss,
              file_path):
    with open(file_path, "a") as file:
        file.write(
            f"epoch: {epoch}, "
            f"training loss: {training_loss:.6f}, "
            f"dev fold: {dev_fold_id}, dev loss: {dev_avg_loss:.6f}, "
            f"test fold: {test_fold_id}, test loss: {test_avg_loss:.6f}, "
            f"dev MAE: {dev_mae:.6f}, test MAE: {test_mae:.6f}, "
            f"dev RMSE: {dev_rmse:.6f}, test RMSE: {test_rmse:.6f}, "
            f"dev R2: {dev_r2:.6f}, test R2: {test_r2:.6f}\n"
        )

def epoch_of_training(model, train_set, optimizer, criterion, processor):
    model.train()

    cumulative_loss = 0
    total_samples = 0
    batch_loss = 0

    for batch in train_set:
        optimizer.zero_grad()
        features, targets = batch
        targets = targets.to(torch.float32).to(processor)
        features = features.to(torch.float32).to(processor)
        logits = model(features)
        batch_loss = criterion(logits, targets)
        cumulative_loss += batch_loss.item() * features.size(0)
        total_samples += features.size(0)

        batch_loss.backward()
        optimizer.step()
        #print(cumulative_loss)
        #print(total_samples)
    #exit()

    return cumulative_loss/total_samples
    


def evaluate_model(model, criterion, fold, device, ttp_censor_val=42.0):
    """
    Evaluates a single-head regression model when many labels are the censor value (e.g., 42).

    Returns a dict with:
      - loss: average criterion over all samples
      - mae_all, rmse_all, r2_all: regression metrics on ALL targets (for continuity with past runs)
      - mae_non42, rmse_non42, r2_non42: regression metrics ONLY on non-42 targets (the informative subset)
      - auc_tbplus: ROC-AUC for classifying TB+ (non-42) vs TB- (42) using a simple score from predictions
      - acc_tbplus: accuracy of that classification using a threshold at 42
    """
    model.eval()
    total_loss, n = 0.0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in fold:
            x, y = batch
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            total_loss += loss.item() * x.size(0)
            n += x.size(0)

            all_targets.append(y.detach().cpu().numpy())
            all_preds.append(y_hat.detach().cpu().numpy())

    targets = np.concatenate(all_targets).reshape(-1)
    preds   = np.concatenate(all_preds).reshape(-1)

    # ---- Overall regression (includes 42s; keep for comparability) ----
    mae_all  = metrics.mean_absolute_error(targets, preds)
    rmse_all = np.sqrt(metrics.mean_squared_error(targets, preds))
    # r2 can be undefined if variance is ~0; guard:
    try:
        r2_all = metrics.r2_score(targets, preds)
    except Exception:
        r2_all = np.nan

    # ---- Regression only on informative (non-42) targets ----
    mask_non42 = (targets != ttp_censor_val)
    if mask_non42.any():
        mae_non42  = metrics.mean_absolute_error(targets[mask_non42], preds[mask_non42])
        rmse_non42 = np.sqrt(metrics.mean_squared_error(targets[mask_non42], preds[mask_non42]))
        try:
            r2_non42 = metrics.r2_score(targets[mask_non42], preds[mask_non42])
        except Exception:
            r2_non42 = np.nan
    else:
        mae_non42 = rmse_non42 = r2_non42 = np.nan

    # ---- “Censor” classification metrics (is TB+?) ----
    # Define TB+ label as target < 42 (non-censored):
    y_cls_true = (targets != ttp_censor_val).astype(int)
    # Score: lower predicted TTP => more likely TB+ (non-censored).
    # A simple monotonic score is (ttp_censor_val - pred): higher => more TB+.
    y_cls_score = (ttp_censor_val - preds)

    # ROC-AUC (only if both classes present)
    if y_cls_true.sum() > 0 and y_cls_true.sum() < len(y_cls_true):
        auc_tbplus = metrics.roc_auc_score(y_cls_true, y_cls_score)
    else:
        auc_tbplus = np.nan

    # Thresholded accuracy using 42 as the cutoff (predict TB+ if pred < 42)
    y_cls_pred = (preds < ttp_censor_val).astype(int)
    acc_tbplus = metrics.accuracy_score(y_cls_true, y_cls_pred) if not np.all(y_cls_true == y_cls_true[0]) else np.nan

    avg_loss = total_loss / max(1, n)

    return {
        "loss": avg_loss,
        "mae_all": mae_all, "rmse_all": rmse_all, "r2_all": r2_all,
        "mae_non42": mae_non42, "rmse_non42": rmse_non42, "r2_non42": r2_non42,
        "auc_tbplus": auc_tbplus, "acc_tbplus": acc_tbplus,
        "n_total": int(n), "n_non42": int(mask_non42.sum())
    }





def train_and_validate_model_k(model, train_set, dev_set, test_set,dev_fold,test_fold, learning_rate, weight_decay, num_epochs, processor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(0, num_epochs):
        training_loss = epoch_of_training(model, train_set, optimizer, criterion, processor)
        if dev_set is not None:
            dev_mae, dev_rmse, dev_r2, dev_avg_loss = evaluate_model(model, criterion, dev_set, processor)
        if test_set is not None:
            test_mae, test_rmse, test_r2, test_avg_loss = evaluate_model(model, criterion, test_set, processor)
        log_epoch(epoch, training_loss,
            dev_fold, dev_mae, dev_rmse, dev_r2, dev_avg_loss,
            test_fold, test_mae, test_rmse, test_r2, test_avg_loss,
            LOG_FILE)
    
    return dev_mae, dev_rmse, dev_r2, dev_avg_loss, test_mae, test_rmse, test_r2, test_avg_loss

