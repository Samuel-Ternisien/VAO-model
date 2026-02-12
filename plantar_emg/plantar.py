import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import sys

# ==========================================
# 0. LOGGING CONFIGURATION
# ==========================================

# Create a custom logger
logger = logging.getLogger("MindScan_Logger")
logger.setLevel(logging.INFO)

# Create handlers
log_file = "/home/fisa/stockage1/data_pirosa/training.log"
# Ensure directory exists for the log file
os.makedirs(os.path.dirname(log_file), exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)

# Optional: Add StreamHandler to also output to console if needed (commented out for background run)
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)

# ==========================================
# 1. CONFIGURATION
# ==========================================

CONFIG = {
    "data_root": "/home/fisa/stockage1/mindscan",
    "output_dir": "/home/fisa/stockage1/data_pirosa",
    "target_fs": 50,         # Fréquence cible (Insoles 100Hz -> 50Hz)
    "window_size_sec": 2.5,  # Fenêtre de 2.5s
    "step_size_sec": 1.25,   # Overlap 50%
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Ensure output directories exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "confusion_matrix"), exist_ok=True)

# Définition des Splits
TRAIN_SEQS = [f"{i:02d}" for i in range(1, 9)]   # 01 à 08
EVAL_SEQS = ["09", "10"]                         # 09 et 10

TRAIN_SUBJECTS_LIST = [f"S{i:02d}" for i in range(1, 25)] # S01 à S24
TEST_SUBJECTS_LIST = [f"S{i:02d}" for i in range(25, 33)] # S25 à S32

logger.info(f"Device used: {CONFIG['device']}")

# ==========================================
# 2. DATASET & PRE-PROCESSING
# ==========================================

def resample_dataframe(df, target_fs=50, original_fs=100):
    if len(df) == 0: return df
    ratio = original_fs / target_fs
    new_length = int(len(df) / ratio)
    
    # Interpolation des colonnes numériques
    x_old = np.linspace(0, len(df)-1, len(df))
    x_new = np.linspace(0, len(df)-1, new_length)
    
    new_data = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        new_data[col] = np.interp(x_new, x_old, df[col].values)
        
    return pd.DataFrame(new_data)

class PlantarDataset(Dataset):
    def __init__(self, root_dir, subjects, sequences, label_encoder=None, fit_scaler=False, scaler=None):
        self.samples = []
        self.metadata = []
        
        # Listes temporaires
        X_list = []
        y_list = []
        
        self.target_fs = CONFIG['target_fs']
        self.window_len = int(CONFIG['window_size_sec'] * self.target_fs)
        self.step_len = int(CONFIG['step_size_sec'] * self.target_fs)
        
        logger.info(f"Chargement : {len(subjects)} sujets x {len(sequences)} séquences...")
        
        # Removed tqdm for clean log files, or use it only if interactive
        for subj in subjects:
            for seq_id in sequences:
                path_insoles = os.path.join(root_dir, "Plantar_activity", subj, f"Sequence_{seq_id}", "insoles.csv")
                path_labels = os.path.join(root_dir, "Events", subj, f"Sequence_{seq_id}", "classif.csv")
                
                if not os.path.exists(path_insoles): continue
                if not os.path.exists(path_labels): continue
                
                try:
                    df_insoles = pd.read_csv(path_insoles, sep=';') 
                    df_classif = pd.read_csv(path_labels, sep=';')
                except Exception as e:
                    logger.error(f"Erreur lecture {subj}/{seq_id}: {e}")
                    continue
                
                # Nettoyage et Detection Colonnes
                df_classif.columns = df_classif.columns.str.strip()
                
                label_col = next((c for c in df_classif.columns if 'action' in c.lower() or 'label' in c.lower() or 'cat' in c.lower()), None)
                start_cols = [c for c in df_classif.columns if ('start' in c.lower() or 'debut' in c.lower()) and 'time' in c.lower()]
                end_cols = [c for c in df_classif.columns if ('end' in c.lower() or 'fin' in c.lower()) and 'time' in c.lower()]
                
                if not label_col: label_col = df_classif.columns[0]
                start_col = start_cols[0] if start_cols else None
                end_col = end_cols[0] if end_cols else None
                
                if not start_col and len(df_classif.columns) >= 4:
                    start_col = df_classif.columns[3]
                if not end_col and len(df_classif.columns) >= 6:
                    end_col = df_classif.columns[5]

                if not (start_col and end_col and label_col):
                    continue

                # Resampling & Windowing
                df_resampled = resample_dataframe(df_insoles, self.target_fs, original_fs=100)
                data_values = df_resampled.values 
                
                for _, row in df_classif.iterrows():
                    try:
                        action = row[label_col]
                        t_start = float(str(row[start_col]).replace(',', '.'))
                        t_end = float(str(row[end_col]).replace(',', '.'))
                        
                        idx_start = int(t_start * self.target_fs)
                        idx_end = int(t_end * self.target_fs)
                        
                        if idx_start >= len(data_values): continue
                        if idx_end > len(data_values): idx_end = len(data_values)
                        
                        segment = data_values[idx_start:idx_end]
                        
                        if len(segment) >= self.window_len:
                            for i in range(0, len(segment) - self.window_len + 1, self.step_len):
                                window = segment[i : i + self.window_len]
                                X_list.append(window)
                                y_list.append(action)
                                self.metadata.append((subj, seq_id))
                    except ValueError:
                        continue 

        if not X_list:
            logger.critical("Aucune donnée chargée ! Arrêt du script.")
            raise ValueError("Aucune donnée chargée !")

        self.X = np.array(X_list, dtype=np.float32)
        
        # Encodage Labels
        if label_encoder is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(y_list)
        else:
            self.le = label_encoder
            known_mask = np.isin(y_list, self.le.classes_)
            if not np.all(known_mask):
                logger.warning(f"{np.sum(~known_mask)} samples ignorés (classes inconnues dans le test set)")
                self.X = self.X[known_mask]
                y_list = np.array(y_list)[known_mask]
                self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if known_mask[i]]
            
            self.y = self.le.transform(y_list)

        # Scaling
        N, T, F = self.X.shape
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X.reshape(-1, F)).reshape(N, T, F)
        else:
            if scaler:
                self.scaler = scaler
                self.X = self.scaler.transform(self.X.reshape(-1, F)).reshape(N, T, F)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long), self.metadata[idx]

# ==========================================
# 3. MODÈLE
# ==========================================

class PlantarActionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(PlantarActionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# ==========================================
# 4. TRAINING & EVAL LOOP
# ==========================================

def train(model, loader, crit, opt, device):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for X, y, _ in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()*X.size(0)
        total_acc += (out.argmax(1)==y).sum().item()
        n += X.size(0)
    return total_loss/n, total_acc/n

def evaluate(model, loader, crit, device):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = crit(out, y)
            total_loss += loss.item()*X.size(0)
            total_acc += (out.argmax(1)==y).sum().item()
            n += X.size(0)
    return total_loss/n, total_acc/n

# ==========================================
# 5. MAIN
# ==========================================

def main():
    try:
        # --- DATASETS ---
        logger.info(">>> Chargement TRAIN...")
        train_ds = PlantarDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, TRAIN_SEQS, fit_scaler=True)
        
        logger.info(">>> Chargement EVAL...")
        eval_ds = PlantarDataset(CONFIG['data_root'], TRAIN_SUBJECTS_LIST, EVAL_SEQS, 
                                 label_encoder=train_ds.le, scaler=train_ds.scaler, fit_scaler=False)
        
        train_dl = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        eval_dl = DataLoader(eval_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # --- MODEL ---
        input_dim = train_ds.X.shape[2]
        n_classes = len(train_ds.le.classes_)
        logger.info(f"Features: {input_dim}, Classes: {n_classes}")
        
        model = PlantarActionModel(input_dim, 128, n_classes).to(CONFIG['device'])
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        crit = nn.CrossEntropyLoss()
        
        # --- TRAIN ---
        best_acc = 0
        save_path = os.path.join(CONFIG["output_dir"], "best_model.pth")
        
        for ep in range(CONFIG['epochs']):
            tl, ta = train(model, train_dl, crit, opt, CONFIG['device'])
            vl, va = evaluate(model, eval_dl, crit, CONFIG['device'])
            
            logger.info(f"Ep {ep+1:02d} | Train: {ta:.3f} | Val: {va:.3f} (Loss: {vl:.3f})")
            
            if va > best_acc:
                best_acc = va
                torch.save({'model': model.state_dict(), 'scaler': train_ds.scaler, 'le': train_ds.le}, save_path)
                logger.info("  --> Nouveau meilleur modèle sauvegardé.")
                
        # --- TEST ---
        logger.info("\n>>> TEST (Sujets 25-32)...")
        if not os.path.exists(save_path):
             logger.error("Modèle introuvable pour le test.")
             return

        ckpt = torch.load(save_path)
        model.load_state_dict(ckpt['model'])
        
        test_ds = PlantarDataset(CONFIG['data_root'], TEST_SUBJECTS_LIST, TRAIN_SEQS + EVAL_SEQS,
                                 label_encoder=ckpt['le'], scaler=ckpt['scaler'], fit_scaler=False)
        test_dl = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # Collecte resultats
        res = {s: {'true':[], 'pred':[]} for s in TEST_SUBJECTS_LIST}
        model.eval()
        with torch.no_grad():
            for X, y, meta in test_dl:
                X = X.to(CONFIG['device'])
                pred = model(X).argmax(1).cpu().numpy()
                y = y.numpy()
                subjs = meta[0] # Tuple de sujets dans le batch
                
                for i, s in enumerate(subjs):
                    res[s]['true'].append(y[i])
                    res[s]['pred'].append(pred[i])
                    
        # Matrices
        classes = ckpt['le'].classes_
        for s, data in res.items():
            if not data['true']: continue
            acc = accuracy_score(data['true'], data['pred'])
            cm = confusion_matrix(data['true'], data['pred'], labels=range(len(classes)))
            
            plt.figure(figsize=(12,10))
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f"Confusion Matrix {s} (Acc: {acc:.1%})")
            plt.tight_layout()
            
            out_file = os.path.join(CONFIG["output_dir"], "confusion_matrix", f"cm_{s}.png")
            plt.savefig(out_file)
            plt.close()
            logger.info(f"Sauvegardé {out_file} (Acc: {acc:.1%})")
            
    except Exception as e:
        logger.critical(f"Erreur critique dans le main: {e}", exc_info=True)

if __name__ == "__main__":
    main()