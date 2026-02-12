import os
import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.asyncio import tqdm

# --- Configuration & Logging ---
LOG_FILE = "training_log.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOTS = {
    "imu": "/home/fisa/stockage1/mindscan/IMU",
    "plantar": "/home/fisa/stockage1/mindscan/Plantar_activity",
    "events": "/home/fisa/stockage1/mindscan/Events",
}
FILENAMES = {"imu": "imu.csv", "plantar": "insoles.csv"}
L = 256
NUM_CLASSES = 31
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---
def smart_read_csv(path):
    for sep in [";", "\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1: return df
        except Exception: pass
    raise ValueError(f"Impossible de lire {path}")

def find_time_col(df):
    for c in df.columns:
        if "time" in c.lower(): return c
    raise ValueError("Colonne temps non trouvee")

def read_and_slice_by_time(path, t0, t1):
    df = smart_read_csv(path)
    tcol = find_time_col(df)
    df = df[(df[tcol] >= t0) & (df[tcol] <= t1)]
    df = df.drop(columns=[tcol])
    return torch.tensor(df.values, dtype=torch.float32)

def resample_to_L(x, L):
    T, C = x.shape
    if T == 0: return torch.zeros(L, C)
    if T == L: return x
    idx = torch.linspace(0, T - 1, L)
    idx0 = idx.long()
    idx1 = torch.clamp(idx0 + 1, max=T - 1)
    w = idx - idx0
    return (1 - w).unsqueeze(1) * x[idx0] + w.unsqueeze(1) * x[idx1]

# --- Dataset & Model ---
class MultiModalEventDataset(Dataset):
    def __init__(self, segments_df, roots, filenames, L=256):
        self.df = segments_df.reset_index(drop=True)
        self.roots, self.filenames, self.L = roots, filenames, L

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        t0, t1, y = float(r["t0"]), float(r["t1"]), int(r["label"]) - 1
        
        imu = read_and_slice_by_time(os.path.join(self.roots["imu"], r["subject"], r["seq"], self.filenames["imu"]), t0, t1)
        imu = resample_to_L(imu, self.L).transpose(0, 1).contiguous()
        
        plantar = read_and_slice_by_time(os.path.join(self.roots["plantar"], r["subject"], r["seq"], self.filenames["plantar"]), t0, t1)
        plantar = resample_to_L(plantar, self.L).transpose(0, 1).contiguous()
        
        return {"imu": imu, "plantar": plantar, "y": torch.tensor(y, dtype=torch.long)}

class ConvBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, x):
        return self.features(x).mean(dim=-1)

class IMUPlantarMultiCNN(nn.Module):
    def __init__(self, imu_in_channels, plantar_in_channels, num_classes):
        super().__init__()
        self.imu_branch = ConvBranch(imu_in_channels)
        self.plantar_branch = ConvBranch(plantar_in_channels)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
    def forward(self, imu_x, plantar_x):
        fused = torch.cat([self.imu_branch(imu_x), self.plantar_branch(plantar_x)], dim=1)
        return self.classifier(fused)

# --- Async Training Logic ---
async def train_one_epoch(model, dl, optimizer, criterion, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    # tqdm.asyncio allows for non-blocking progress bars
    async for batch in tqdm(dl, desc=f"Epoch {epoch+1}/{total_epochs}"):
        imu_x = batch["imu"].to(DEVICE, non_blocking=True)
        plantar_x = batch["plantar"].to(DEVICE, non_blocking=True)
        y = batch["y"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imu_x, plantar_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imu_x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += imu_x.size(0)
        
        # Yield control to the event loop
        await asyncio.sleep(0) 

    logger.info(f"Epoch {epoch+1} - Loss: {running_loss/total:.4f}, Acc: {correct/total:.3f}")

async def validate(model, dl):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dl:
            imu_x, plantar_x, y = batch["imu"].to(DEVICE), batch["plantar"].to(DEVICE), batch["y"].to(DEVICE)
            logits = model(imu_x, plantar_x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    logger.info(f"Validation Accuracy: {acc:.3f}")
    return acc

async def main():
    logger.info(f"Starting Multimodal Training on {DEVICE}")
    
    # Indexing
    rows = []
    subjects = sorted([d for d in os.listdir(ROOTS["events"]) if os.path.isdir(os.path.join(ROOTS["events"], d))])
    for subject in subjects:
        subj_path = os.path.join(ROOTS["events"], subject)
        for seq in os.listdir(subj_path):
            classif = os.path.join(subj_path, seq, "classif.csv")
            if os.path.exists(classif):
                df = pd.read_csv(classif, sep=";")
                for _, r in df.iterrows():
                    rows.append({"subject": subject, "seq": seq, "label": int(r["Class"]), "t0": float(r["Timestamp Start"]), "t1": float(r["Timestamp End"])})
    
    segments = pd.DataFrame(rows)
    allowed_subjects = {f"S{str(i).zfill(2)}" for i in range(1, 25)}
    segments = segments[segments["subject"].isin(allowed_subjects)].reset_index(drop=True)
    
    # Split
    unique_subjs = segments["subject"].unique()
    np.random.seed(0)
    np.random.shuffle(unique_subjs)
    n_train = int(0.8 * len(unique_subjs))
    train_segments = segments[segments["subject"].isin(unique_subjs[:n_train])]
    val_segments = segments[segments["subject"].isin(unique_subjs[n_train:])]

    # DataLoaders
    train_ds = MultiModalEventDataset(train_segments, ROOTS, FILENAMES, L)
    val_ds = MultiModalEventDataset(val_segments, ROOTS, FILENAMES, L)
    
    # Note: Use num_workers > 0 for faster loading, but keep an eye on memory
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)

    # Initialize Model
    # Determine channels from first sample
    sample = train_ds[0]
    model = IMUPlantarMultiCNN(
        imu_in_channels=sample["imu"].shape[0],
        plantar_in_channels=sample["plantar"].shape[0],
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(20):
        await train_one_epoch(model, train_dl, optimizer, criterion, epoch, 20)
        await validate(model, val_dl)

if __name__ == "__main__":
    asyncio.run(main())