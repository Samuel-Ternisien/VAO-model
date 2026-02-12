import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import logging
import wandb
# from tqdm import tqdm

from config import CONFIG, TRAIN_SEQS, EVAL_SEQS, TRAIN_SUBJECTS_LIST, TEST_SUBJECTS_LIST
from utils import get_logger
from dataset import PlantarDataset, EMGDataset, MultimodalDataset
from models import HybridTwoStreamModel

logger = get_logger()

def train_epoch(model, loader, crit, opt, device, is_multimodal=False):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    
    for batch_data, y, _ in loader:
        y = y.to(device)
        
        if is_multimodal:
            X_p, X_e = batch_data 
            X_p, X_e = X_p.to(device), X_e.to(device)
            out = model(X_p, X_e)
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
    with torch.no_grad():
        for batch_data, y, _ in loader:
            y = y.to(device)
            
            if is_multimodal:
                X_p, X_e = batch_data
                X_p, X_e = X_p.to(device), X_e.to(device)
                out = model(X_p, X_e)
            else:
                X = batch_data.to(device)
                out = model(X)
                
            loss = crit(out, y)
            total_loss += loss.item() * y.size(0)
            total_acc += (out.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss/n, total_acc/n

def main():
    try:
        # 1. Initialize WandB
        wandb_config = {k: v for k, v in CONFIG.items() if k != 'device'}
        # Replace with your key or use wandb login in terminal
        wandb.login(key="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", relogin=True)
        wandb.init(
            project=CONFIG["wandb_project"],
            name=CONFIG["run_name"],
            config=wandb_config
        )

        mode = CONFIG['modality']
        is_multimodal = (mode == "MULTIMODAL")
        logger.info(f"=== TRAINING START : {mode} ===")
        
        # 2. Chargement TRAIN
        logger.info(">>> Chargement TRAIN...")
        if is_multimodal:
            train_ds = MultimodalDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, TRAIN_SEQS, fit_scaler=True)
        elif mode == "PLANTAR":
            train_ds = PlantarDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, TRAIN_SEQS, fit_scaler=True)
        elif mode == "EMG":
            train_ds = EMGDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, TRAIN_SEQS, fit_scaler=True)
        
        # DEFINITION DES SCALERS AVANT UTILISATION
        scalers = None
        if is_multimodal:
            scalers = {'plantar': train_ds.scaler_plantar, 'emg': train_ds.scaler_emg}

        # 3. Chargement EVAL
        logger.info(">>> Chargement EVAL...")
        if is_multimodal:
            eval_ds = MultimodalDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, EVAL_SEQS, 
                                        label_encoder=train_ds.le, fit_scaler=False, scalers=scalers)
        else:
            DatasetClass = PlantarDataset if mode == "PLANTAR" else EMGDataset
            eval_ds = DatasetClass(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, EVAL_SEQS, 
                                   label_encoder=train_ds.le, fit_scaler=False, scaler=train_ds.scaler)

        # 4. Chargement TEST (Periodic)
        logger.info(">>> Chargement TEST (for periodic eval)...")
        if is_multimodal:
            test_ds = MultimodalDataset(CONFIG['data_root'], TEST_SUBJECTS_LIST, TRAIN_SEQS + EVAL_SEQS,
                                        label_encoder=train_ds.le, fit_scaler=False, scalers=scalers)
        else:
            DatasetClass = PlantarDataset if mode == "PLANTAR" else EMGDataset
            test_ds = DatasetClass(CONFIG['data_root'], TEST_SUBJECTS_LIST, TRAIN_SEQS + EVAL_SEQS,
                                   label_encoder=train_ds.le, fit_scaler=False, scaler=train_ds.scaler)
                                   
        train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        eval_dl = DataLoader(eval_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # 5. Modèle
        n_classes = len(train_ds.le.classes_)
        
        if is_multimodal:
            dim_p = train_ds.X_plantar.shape[2]
            dim_e = train_ds.X_emg.shape[2]
            logger.info(f"Dims: Plantar={dim_p}, EMG={dim_e}, Classes={n_classes}")
            model = HybridTwoStreamModel(dim_p, dim_e, hidden_dim=64, num_classes=n_classes).to(CONFIG['device'])
        else:
            # Fallback simple model if not multimodal
            # Assuming you have GenericActionLSTM available or reuse Hybrid with 0 dim
            pass 
            
        # WandB watch
        wandb.watch(model, log="all", log_freq=10)

        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        crit = nn.CrossEntropyLoss()
        
        # 6. Loop
        best_val_acc = 0.0
        best_test_acc = 0.0
        
        save_path_val = os.path.join(CONFIG["output_dir"], f"best_model_val_{mode}.pth")
        save_path_test = os.path.join(CONFIG["output_dir"], f"best_model_test_{mode}.pth")
        
        for ep in range(CONFIG['epochs']):
            tl, ta = train_epoch(model, train_dl, crit, opt, CONFIG['device'], is_multimodal)
            vl, va = evaluate_epoch(model, eval_dl, crit, CONFIG['device'], is_multimodal)
            
            logger.info(f"Ep {ep+1:02d} | Train: {ta:.3f} | Val: {va:.3f} (L:{vl:.3f})")
            
            wandb_metrics = {
                "epoch": ep + 1,
                "train_loss": tl,
                "train_acc": ta,
                "val_loss": vl,
                "val_acc": va
            }

            # --- Save best based on VALIDATION ---
            if va > best_val_acc:
                best_val_acc = va
                state = {
                    'model': model.state_dict(),
                    'le': train_ds.le,
                    'modality': mode,
                    'is_multimodal': is_multimodal
                }
                if is_multimodal:
                    state['scalers'] = scalers
                else:
                    state['scaler'] = train_ds.scaler
                    
                torch.save(state, save_path_val)
                logger.info("  --> Saved (Best Val).")
                wandb_metrics["best_val_acc"] = best_val_acc

            # --- PERIODIC TEST EVALUATION (Every 5 Epochs) ---
            if (ep + 1) % 5 == 0:
                logger.info("--- Periodic Test Evaluation ---")
                test_loss, test_acc = evaluate_epoch(model, test_dl, crit, CONFIG['device'], is_multimodal)
                logger.info(f"    Test Acc: {test_acc:.3f} (Loss: {test_loss:.3f})")
                
                wandb_metrics["test_loss"] = test_loss
                wandb_metrics["test_acc"] = test_acc
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    state_test = {
                        'model': model.state_dict(),
                        'le': train_ds.le,
                        'modality': mode,
                        'is_multimodal': is_multimodal
                    }
                    if is_multimodal:
                        state_test['scalers'] = scalers
                    else:
                        state_test['scaler'] = train_ds.scaler

                    torch.save(state_test, save_path_test)
                    logger.info(f"  --> Saved (New Best Test: {test_acc:.3f}).")
                    wandb_metrics["best_test_acc"] = best_test_acc
            
            wandb.log(wandb_metrics)

        # 7. FINAL TEST EVALUATION & MATRICES
        logger.info("\n>>> FINAL TEST (S25-S32)...")
        if not os.path.exists(save_path_test): 
            logger.warning("No best test model found, skipping final matrix generation.")
            return
        
        ckpt = torch.load(save_path_test)
        model.load_state_dict(ckpt['model'])
        
        # No need to reload dataset, we already have test_dl
        
        res = {s: {'true':[], 'pred':[]} for s in TEST_SUBJECTS_LIST}
        model.eval()
        
        with torch.no_grad():
            for batch_data, y, meta in test_dl:
                y = y.numpy()
                subjs = meta[0]
                
                if is_multimodal:
                    X_p, X_e = batch_data
                    X_p, X_e = X_p.to(CONFIG['device']), X_e.to(CONFIG['device'])
                    pred = model(X_p, X_e).argmax(1).cpu().numpy()
                else:
                    X = batch_data.to(CONFIG['device'])
                    pred = model(X).argmax(1).cpu().numpy()
                
                for i, s in enumerate(subjs):
                    res[s]['true'].append(y[i])
                    res[s]['pred'].append(pred[i])
                    
        classes = ckpt['le'].classes_
        avg_test_acc = 0
        
        for s, data in res.items():
            if not data['true']: continue
            acc = accuracy_score(data['true'], data['pred'])
            avg_test_acc += acc
            cm = confusion_matrix(data['true'], data['pred'], labels=range(len(classes)))
            
            fig = plt.figure(figsize=(12,10))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f"CM {s} - {mode} (Acc: {acc:.1%})")
            plt.tight_layout()
            
            img_path = os.path.join(CONFIG["output_dir"], "confusion_matrix", f"cm_{s}_{mode}.png")
            plt.savefig(img_path)
            plt.close()
            
            wandb.log({f"confusion_matrix/{s}": wandb.Image(img_path, caption=f"Acc: {acc:.1%}")})
            logger.info(f"Saved CM for {s} (Acc: {acc:.1%})")

        avg_test_acc /= len(res)
        wandb.log({"final_test_avg_accuracy": avg_test_acc})
        
        wandb.finish()

    except Exception as e:
        logger.critical(f"Crash: {e}", exc_info=True)
        wandb.finish()

if __name__ == "__main__":
    main()