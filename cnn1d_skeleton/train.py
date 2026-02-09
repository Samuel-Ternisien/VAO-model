from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import wandb
import yaml

from .models.cnn1d import CNN1D

class NPZDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    ys, yh = [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * len(y)
        n += len(y)
        pred = logits.argmax(dim=1)
        ys.append(y.cpu().numpy())
        yh.append(pred.cpu().numpy())
    ys = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    yh = np.concatenate(yh) if yh else np.array([], dtype=np.int64)
    acc = float((ys == yh).mean()) if len(ys) else 0.0
    f1m = float(f1_score(ys, yh, average="macro")) if len(ys) else 0.0
    cm = confusion_matrix(ys, yh, labels=list(range(num_classes))) if len(ys) else None
    return {"loss": loss_sum/max(n,1), "acc": acc, "f1_macro": f1m, "cm": cm}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default="artifacts")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(args.data_dir)
    train_ds = NPZDataset(str(data_dir/"train.npz"))
    val_ds   = NPZDataset(str(data_dir/"val.npz"))
    test_ds  = NPZDataset(str(data_dir/"test.npz"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    num_classes = int(cfg["num_classes"])
    in_features = train_ds.X.shape[-1]

    model = CNN1D(in_features=in_features, num_classes=num_classes, dropout=float(cfg["dropout"])).to(device)

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    ce = nn.CrossEntropyLoss()

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config={**cfg, "device": device, "in_features": in_features,
                "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds)}
    )

    best_f1 = -1.0
    best_path = Path("artifacts/best_model.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(y)
            n += len(y)

        val_metrics = evaluate(model, val_loader, device, num_classes)
        wandb.log({
            "epoch": epoch,
            "train/loss": loss_sum/max(n,1),
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "val/f1_macro": val_metrics["f1_macro"],
        })

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_path)

    # test final
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, num_classes)

    wandb.summary["test_loss"] = test_metrics["loss"]
    wandb.summary["test_acc"] = test_metrics["acc"]
    wandb.summary["test_f1_macro"] = test_metrics["f1_macro"]
    if test_metrics["cm"] is not None:
        wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=None,
            preds=None,
            class_names=[str(i+1) for i in range(num_classes)],
            matrix=test_metrics["cm"]
        )})
    wandb.finish()

if __name__ == "__main__":
    main()