from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from windowing import frame_to_label_array, make_windows
from .label_reader import read_events_segments
from .skeleton_reader import read_skeleton_csv

def subj_id_to_name(i: int) -> str:
    return f"S{i:02d}"

def build_split(data_root: str, subjects: list[int], out_path: str,
                win: int, stride: int, purity: float):
    root = Path(data_root)
    all_X = []
    all_y = []
    meta = []  # (subject, sequence)

    for sid in tqdm(subjects, desc=f"Building {Path(out_path).stem}"):
        sname = subj_id_to_name(sid)
        skel_dir = root / "Skeleton" / sname
        evt_dir  = root / "Events" / sname

        # suppose fichiers: Sequence_01, Sequence_02, ... (même nom)
        skel_files = sorted(skel_dir.glob("Sequence_*"))
        for skel_path in skel_files:
            seq_name = skel_path.name  # ex: Sequence_01
            evt_path = evt_dir / seq_name
            if not evt_path.exists():
                # parfois extension .csv / .txt -> tenter des variantes
                alt = list(evt_dir.glob(seq_name + ".*"))
                if alt:
                    evt_path = alt[0]
                else:
                    continue

            times, X, feat_names = read_skeleton_csv(skel_path)
            segments = read_events_segments(evt_path)
            y_frame = frame_to_label_array(X.shape[0], segments)

            Xw, yw = make_windows(X, y_frame, win=win, stride=stride, purity=purity)
            if len(yw) == 0:
                continue

            all_X.append(Xw)
            all_y.append(yw)
            meta.extend([(sid, seq_name)] * len(yw))

    X_final = np.concatenate(all_X, axis=0) if all_X else np.empty((0, win, 0), dtype=np.float32)
    y_final = np.concatenate(all_y, axis=0) if all_y else np.empty((0,), dtype=np.int32)

    # labels sont 1..31 → on convertit en 0..30 pour PyTorch
    y_final = y_final - 1

    np.savez_compressed(out_path,
                        X=X_final,
                        y=y_final,
                        meta=np.array(meta, dtype=object),
                        win=win,
                        stride=stride,
                        purity=purity)
    return X_final.shape, y_final.shape

def main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = cfg["fps"]
    win = int(cfg["window_sec"] * fps)
    stride = int(cfg["stride_sec"] * fps)
    purity = float(cfg["purity"])

    data_root = cfg["data_root"]
    train_subj = cfg["train_subjects"]
    val_subj   = cfg["val_subjects"]
    test_subj  = cfg["test_subjects"]

    build_split(data_root, train_subj, str(out_dir/"train.npz"), win, stride, purity)
    build_split(data_root, val_subj,   str(out_dir/"val.npz"),   win, stride, purity)
    build_split(data_root, test_subj,  str(out_dir/"test.npz"),  win, stride, purity)

if __name__ == "__main__":
    main()