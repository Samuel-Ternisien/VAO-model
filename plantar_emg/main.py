import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import wandb
from tqdm import tqdm

from config import CONFIG, TRAIN_SEQS, EVAL_SEQS, TRAIN_SUBJECTS_LIST, TEST_SUBJECTS_LIST
from utils import get_logger
from dataset import PlantarDataset, EMGDataset, SkeletonDataset, IMUDataset, MultimodalDataset
from models import HybridFourStreamModel

logger = get_logger()

def train_epoch(model, loader, crit, opt, device, is_multimodal=False):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    
    for batch_data, y, _ in loader:
        y = y.to(device)
        
        if is_multimodal:
            # Unpack 4 tensors
            X_p, X_e, X_s, X_i = batch_data 
            X_p, X_e = X_p.to(device), X_e.to(device)
            X_s, X_i = X_s.to(device), X_i.to(device)
            out = model(X_p, X_e, X_s, X_i)
        else:
            X = batch_data.to(device)
            out = model(X)
            
        opt.zero_grad()
        loss = crit(out, y)
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * y.size(0)
        total_acc += (out.argmax(1) == y).sum().item()
        n += y.size(0)
        
    return total_loss/n, total_acc/n

def evaluate_epoch(model, loader, crit, device, is_multimodal=False):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, y, _ in loader:
            y = y.to(device)
            
            if is_multimodal:
                X_p, X_e, X_s, X_i = batch_data
                X_p, X_e = X_p.to(device), X_e.to(device)
                X_s, X_i = X_s.to(device), X_i.to(device)
                out = model(X_p, X_e, X_s, X_i)
            else:
                X = batch_data.to(device)
                out = model(X)
                
            loss = crit(out, y)
            total_loss += loss.item() * y.size(0)
            
            # Get predictions
            preds = out.argmax(1)
            total_acc += (preds == y).sum().item()
            n += y.size(0)
            
            # Store for F1 score
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # Compute metrics
    avg_loss = total_loss / n
    avg_acc = total_acc / n
    f1 = f1_score(all_labels, all_preds, average='weighted') # or 'macro'
    
    return avg_loss, avg_acc, f1

def main():
    try:
        # WandB Setup
        wandb_config = {k: v for k, v in CONFIG.items() if k != 'device'}
        wandb.login(key="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", relogin=True)
        wandb.init(project=CONFIG["wandb_project"], name=CONFIG["run_name"], config=wandb_config)

        mode = CONFIG['modality']
        is_multimodal = (mode == "MULTIMODAL")
        
        # Load Datasets
        logger.info(">>> Loading TRAIN...")
        if is_multimodal:
            train_ds = MultimodalDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, TRAIN_SEQS, fit_scaler=True)
            scalers = train_ds.scalers
        else:
            # Fallback for single (Skeleton/IMU logic would need to be added here if you run single mode)
            pass 

        logger.info(">>> Loading EVAL & TEST...")
        if is_multimodal:
            eval_ds = MultimodalDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, EVAL_SEQS, 
                                        label_encoder=train_ds.le, fit_scaler=False, scalers=scalers)
            test_ds = MultimodalDataset(CONFIG['data_root'], TEST_SUBJECTS_LIST, TRAIN_SEQS + EVAL_SEQS, 
                                        label_encoder=train_ds.le, fit_scaler=False, scalers=scalers)

        train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        eval_dl = DataLoader(eval_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        
        # Model Init
        n_classes = len(train_ds.le.classes_)
        if is_multimodal:
            input_dims = {
                'plantar': train_ds.X['plantar'].shape[2],
                'emg': train_ds.X['emg'].shape[2],
                'skeleton': train_ds.X['skeleton'].shape[2],
                'imu': train_ds.X['imu'].shape[2]
            }
            logger.info(f"Dims: {input_dims} | Classes: {n_classes}")
            model = HybridFourStreamModel(input_dims, num_classes=n_classes)
        
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(CONFIG['device'])

        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        crit = nn.CrossEntropyLoss()
        
        # Loop
        best_val_acc = 0.0
        best_test_acc = 0.0
        save_path_test = os.path.join(CONFIG["output_dir"], f"best_model_test_{mode}.pth")

        for ep in range(CONFIG['epochs']):
            # Train loop doesn't strictly need F1 calc every batch, but good to have
            # For speed, we might keep train_epoch simple, but let's update it if you want train_f1 too.
            # For now, let's assume train_epoch returns (loss, acc) as before to save time, 
            # or update it similarly if needed. Let's stick to VAL/TEST F1 which is most important.
            
            tl, ta = train_epoch(model, train_dl, crit, opt, CONFIG['device'], is_multimodal)
            
            # Update evaluate call
            vl, va, vf1 = evaluate_epoch(model, eval_dl, crit, CONFIG['device'], is_multimodal)
            
            logger.info(f"Ep {ep+1:02d} | Train: {ta:.3f} | Val: {va:.3f} | Val F1: {vf1:.3f}")
            
            wandb.log({
                "train_loss": tl, "train_acc": ta, 
                "val_loss": vl, "val_acc": va, "val_f1": vf1
            })

            # Save Best Val
            if va > best_val_acc:
                best_val_acc = va
                # ... (save logic) ...

            # Periodic Test
            if (ep + 1) % 5 == 0:
                tl_test, ta_test, tf1_test = evaluate_epoch(model, test_dl, crit, CONFIG['device'], is_multimodal)
                
                logger.info(f"   [TEST] Acc: {ta_test:.3f} | F1: {tf1_test:.3f}")
                wandb.log({"test_acc": ta_test, "test_f1": tf1_test})
                
                if ta_test > best_test_acc:
                    best_test_acc = ta_test
                    state = {
                        'model': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                        'le': train_ds.le,
                        'scalers': scalers
                    }
                    torch.save(state, save_path_test)
                    logger.info("   --> New Best Test Model Saved")

        wandb.finish()

    except Exception as e:
        logger.critical(f"Crash: {e}", exc_info=True)
        wandb.finish()

if __name__ == "__main__":
    main()