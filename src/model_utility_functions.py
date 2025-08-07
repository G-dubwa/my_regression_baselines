import torch
from sklearn import metrics
import numpy as np

LOG_FILE = "logs/validation_per_epoch.txt"
def log_epoch(epoch, training_loss, dev_mae, dev_rmse, dev_r2, dev_avg_loss,test_mae, test_rmse, test_r2, test_avg_loss, file_path):
    with open(file_path,"a") as file:
        file.write(
            f"epoch: {epoch}, training loss: {training_loss:.6f}, "
            f"dev loss: {dev_avg_loss:.6f}, test loss: {test_avg_loss:.6f}, "
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
    
def evaluate_model(model, criterion, fold, processor):
    model.eval()
    cumulative_loss = 0
    acc = 0
    auc = 0
    total_samples = 0
    
    targets = []
    logits = []

    with torch.no_grad():
        for batch in fold:
            batch_features, batch_targets = batch
            batch_targets = batch_targets.to(torch.float).to(processor)
            batch_features = batch_features.to(torch.float32).to(processor)
            batch_logits = model(batch_features)
            targets.extend(batch_targets.detach().cpu().numpy())
            logits.extend(batch_logits.detach().cpu().numpy())
            batch_loss = criterion(batch_logits, batch_targets)
            cumulative_loss += batch_loss.item() * batch_features.size(0)
            total_samples += batch_features.size(0)

    targets = np.array(targets)
    logits = np.array(logits)

    mae = metrics.mean_absolute_error(targets, logits)
    rmse = np.sqrt(metrics.mean_squared_error(targets, logits))
    r2 = metrics.r2_score(targets, logits)

    avg_loss = cumulative_loss / total_samples

    return mae, rmse, r2, avg_loss




def train_and_validate_model_k(model, train_set, dev_set, test_set, learning_rate, weight_decay, num_epochs, processor):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    for epoch in range(0, num_epochs):
        training_loss = epoch_of_training(model, train_set, optimizer, criterion, processor)
        if dev_set is not None:
            dev_mae, dev_rmse, dev_r2, dev_avg_loss = evaluate_model(model, criterion, dev_set, processor)
        if test_set is not None:
            test_mae, test_rmse, test_r2, test_avg_loss = evaluate_model(model, criterion, test_set, processor)
        log_epoch(epoch, training_loss, dev_mae, dev_rmse, dev_r2, dev_avg_loss, test_mae, test_rmse, test_r2, test_avg_loss, LOG_FILE)
    
    return dev_mae, dev_rmse, dev_r2, dev_avg_loss, test_mae, test_rmse, test_r2, test_avg_loss

