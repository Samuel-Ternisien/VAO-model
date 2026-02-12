import torch
import os

CONFIG = {
    "data_root": "/home/fisa/stockage1/mindscan",
    
    # Paramètres de traitement
    "target_fs": 50,         
    "window_size_sec": 3,  
    "step_size_sec": 1.25,   
    
    # Paramètres d'entraînement
    "batch_size": 256,
    "epochs": 100,
    "learning_rate": 1e-4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    
    "modality": "MULTIMODAL",

    # WandB
    "wandb_project": "MindScan_Action_Recognition"
}

# Génération dynamique du nom de run pour WandB
CONFIG["run_name"] = f"{CONFIG['modality']}_LR{CONFIG['learning_rate']}_BS{CONFIG['batch_size']}_Win{CONFIG['window_size_sec']}"
CONFIG["output_dir"] = f"/home/fisa/stockage1/data_pirosa/results/{CONFIG['run_name']}"
# Splits
TRAIN_SEQS = [f"{i:02d}" for i in range(1, 9)]
EVAL_SEQS = ["09", "10"]
TRAIN_SUBJECTS_LIST = [f"S{i:02d}" for i in range(1, 25)]
TEST_SUBJECTS_LIST = [f"S{i:02d}" for i in range(25, 33)]

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(os.path.join(CONFIG["output_dir"], "confusion_matrix"), exist_ok=True)