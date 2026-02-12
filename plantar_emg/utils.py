import logging
import os
import sys
import numpy as np
import pandas as pd
from config import CONFIG

def get_logger(name="MindScan_Logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Évite d'ajouter plusieurs handlers si le logger existe déjà
    if not logger.handlers:
        log_file = os.path.join(CONFIG["output_dir"], f"training_{CONFIG['modality']}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Optionnel : StreamHandler pour voir dans la console (si lancé avec nohup, ça part dans nohup.out)
        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setFormatter(formatter)
        # logger.addHandler(console_handler)
        
    return logger

def resample_dataframe(df, target_fs=50, original_fs=100):
    """
    Ré-échantillonne un DataFrame numérique.
    """
    if len(df) == 0: return df
    
    # Calcul des nouveaux indices
    duration_sec = len(df) / original_fs
    new_length = int(duration_sec * target_fs)
    
    x_old = np.linspace(0, len(df)-1, len(df))
    x_new = np.linspace(0, len(df)-1, new_length)
    
    new_data = {}
    # On ne prend que les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        new_data[col] = np.interp(x_new, x_old, df[col].values)
        
    return pd.DataFrame(new_data)