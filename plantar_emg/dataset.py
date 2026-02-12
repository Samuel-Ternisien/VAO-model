import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import get_logger, resample_dataframe
from config import CONFIG

logger = get_logger()

# --- Base Class ---
class BaseMindScanDataset(Dataset):
    """Parent class for common logic."""
    def __init__(self, root_dir, subjects, sequences, label_encoder=None, fit_scaler=False, scaler=None):
        self.samples = []
        self.metadata = []
        self.X = None
        self.y = None
        self.le = label_encoder
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        
        # To be defined in children
        self.data_folder = ""
        self.filename = ""
        self.original_fs = 100 
        
    def _load_data(self, subjects, sequences, root_dir):
        X_list = []
        y_list = []
        
        window_len = int(CONFIG['window_size_sec'] * CONFIG['target_fs'])
        step_len = int(CONFIG['step_size_sec'] * CONFIG['target_fs'])
        
        for subj in subjects:
            for seq_id in sequences:
                path_data = os.path.join(root_dir, self.data_folder, subj, f"Sequence_{seq_id}", self.filename)
                path_labels = os.path.join(root_dir, "Events", subj, f"Sequence_{seq_id}", "classif.csv")
                
                if not os.path.exists(path_data) or not os.path.exists(path_labels): continue
                
                try:
                    df_data = self._read_data_file(path_data)
                    df_classif = pd.read_csv(path_labels, sep=';')
                except Exception as e:
                    # logger.warning(f"Skipping {subj}/{seq_id} due to read error: {e}")
                    continue
                
                df_classif.columns = df_classif.columns.str.strip()
                label_col, start_col, end_col = self._parse_label_columns(df_classif)
                if not (start_col and end_col and label_col): continue

                df_resampled = resample_dataframe(df_data, CONFIG['target_fs'], original_fs=self.original_fs)
                data_values = df_resampled.values 
                
                for _, row in df_classif.iterrows():
                    try:
                        action = row[label_col]
                        t_start = float(str(row[start_col]).replace(',', '.'))
                        t_end = float(str(row[end_col]).replace(',', '.'))
                        
                        idx_start = int(t_start * CONFIG['target_fs'])
                        idx_end = int(t_end * CONFIG['target_fs'])
                        
                        if idx_start >= len(data_values): continue
                        if idx_end > len(data_values): idx_end = len(data_values)
                        
                        segment = data_values[idx_start:idx_end]
                        
                        if len(segment) >= window_len:
                            for i in range(0, len(segment) - window_len + 1, step_len):
                                X_list.append(segment[i : i + window_len])
                                y_list.append(action)
                                self.metadata.append((subj, seq_id))
                    except ValueError: continue 
        return X_list, y_list

    def _read_data_file(self, path):
        return pd.read_csv(path, sep=';', low_memory=False)

    def _parse_label_columns(self, df):
        label_col = next((c for c in df.columns if 'action' in c.lower() or 'label' in c.lower()), None)
        start_cols = [c for c in df.columns if ('start' in c.lower() or 'debut' in c.lower()) and 'time' in c.lower()]
        end_cols = [c for c in df.columns if ('end' in c.lower() or 'fin' in c.lower()) and 'time' in c.lower()]
        
        if not label_col: label_col = df.columns[0]
        start_col = start_cols[0] if start_cols else (df.columns[3] if len(df.columns)>=4 else None)
        end_col = end_cols[0] if end_cols else (df.columns[5] if len(df.columns)>=6 else None)
        return label_col, start_col, end_col

    def _finalize_init(self, X_list, y_list):
        if not X_list: raise ValueError(f"[{self.__class__.__name__}] Dataset Empty.")
        self.X = np.array(X_list, dtype=np.float32)
        
        if self.le is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(y_list)
        else:
            known_mask = np.isin(y_list, self.le.classes_)
            self.X = self.X[known_mask]
            y_list = np.array(y_list)[known_mask]
            self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if known_mask[i]]
            self.y = self.le.transform(y_list)

        N, T, F = self.X.shape
        if self.fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X.reshape(-1, F)).reshape(N, T, F)
        elif self.scaler:
            self.X = self.scaler.transform(self.X.reshape(-1, F)).reshape(N, T, F)
    
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long), self.metadata[idx]

# --- Single Modality Datasets ---

class PlantarDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "Plantar_activity"
        self.filename = "insoles.csv"
        self.original_fs = 100
        self._finalize_init(*self._load_data(subjects, sequences, root_dir))

class EMGDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "EMG"
        self.filename = "emg.csv"
        self.original_fs = 1259 
        self._finalize_init(*self._load_data(subjects, sequences, root_dir))

    def _read_data_file(self, path):
        df = pd.read_csv(path, sep=';', low_memory=False)
        df.columns = df.columns.str.strip()
        # Drop Time column
        cols = [c for c in df.columns if 'TIME' not in c]
        return df[cols]

class SkeletonDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "Skeleton"
        self.filename = "skeleton.csv"
        self.original_fs = 120 
        self._finalize_init(*self._load_data(subjects, sequences, root_dir))

    def _read_data_file(self, path):
        # 1. low_memory=False to avoid DtypeWarning
        df = pd.read_csv(path, sep=';', low_memory=False)
        # Drop fully empty columns
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.str.strip()
        
        # 2. Drop non-feature columns
        drop_cols = [c for c in df.columns if 'Frame' in c or 'Time' in c]
        df = df.drop(columns=drop_cols)
        
        # 3. Force numeric conversion (coercing errors to NaN)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # 4. Fill NaNs (Interpolate is best for mocap data, fillna(0) as backup)
        df = df.interpolate(method='linear', limit_direction='both').fillna(0.0)
        
        return df

class IMUDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "IMU"
        self.filename = "imu.csv"
        self.original_fs = 148 
        self._finalize_init(*self._load_data(subjects, sequences, root_dir))

    def _read_data_file(self, path):
        df = pd.read_csv(path, sep=';', low_memory=False)
        df.columns = df.columns.str.strip()
        cols = [c for c in df.columns if 'TIME' not in c]
        return df[cols]

# --- Multimodal Dataset (4 Streams) ---

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, subjects, sequences, label_encoder=None, fit_scaler=False, scalers=None):
        self.metadata = []
        self.le = label_encoder
        self.fit_scaler = fit_scaler
        
        # Init scalers
        self.scalers = scalers if scalers else {
            'plantar': StandardScaler() if fit_scaler else None,
            'emg': StandardScaler() if fit_scaler else None,
            'skeleton': StandardScaler() if fit_scaler else None,
            'imu': StandardScaler() if fit_scaler else None
        }

        # Setup
        self.target_fs = CONFIG['target_fs']
        window_len = int(CONFIG['window_size_sec'] * self.target_fs)
        step_len = int(CONFIG['step_size_sec'] * self.target_fs)
        
        # Temp buffers
        buffers = {'p': [], 'e': [], 's': [], 'i': [], 'y': []}
        
        logger.info(f"[Multimodal-4] Loading {len(subjects)} subjects...")
        
        for subj in subjects:
            for seq_id in sequences:
                # Paths
                paths = {
                    'p': os.path.join(root_dir, "Plantar_activity", subj, f"Sequence_{seq_id}", "insoles.csv"),
                    'e': os.path.join(root_dir, "EMG", subj, f"Sequence_{seq_id}", "emg.csv"),
                    's': os.path.join(root_dir, "Skeleton", subj, f"Sequence_{seq_id}", "skeleton.csv"),
                    'i': os.path.join(root_dir, "IMU", subj, f"Sequence_{seq_id}", "imu.csv"),
                    'lbl': os.path.join(root_dir, "Events", subj, f"Sequence_{seq_id}", "classif.csv")
                }
                
                # Check existence
                if not all(os.path.exists(p) for p in paths.values()): continue
                
                try:
                    # --- Load & Clean with robust types ---
                    
                    # Plantar
                    df_p = pd.read_csv(paths['p'], sep=';', low_memory=False)
                    
                    # EMG
                    df_e = pd.read_csv(paths['e'], sep=';', low_memory=False)
                    df_e = df_e[[c for c in df_e.columns if 'TIME' not in c]]
                    
                    # Skeleton (FIXED HERE)
                    df_s = pd.read_csv(paths['s'], sep=';', low_memory=False).dropna(axis=1, how='all')
                    df_s = df_s.drop(columns=[c for c in df_s.columns if 'Frame' in c or 'Time' in c])
                    # Force numeric to handle "mixed types" error
                    df_s = df_s.apply(pd.to_numeric, errors='coerce')
                    df_s = df_s.interpolate(method='linear', limit_direction='both').fillna(0.0)
                    
                    # IMU
                    df_i = pd.read_csv(paths['i'], sep=';', low_memory=False)
                    df_i = df_i[[c for c in df_i.columns if 'TIME' not in c]]
                    
                    # Labels
                    df_lbl = pd.read_csv(paths['lbl'], sep=';')
                    
                    # Resample all to target_fs
                    df_p = resample_dataframe(df_p, self.target_fs, 100)
                    df_e = resample_dataframe(df_e, self.target_fs, 1259)
                    df_s = resample_dataframe(df_s, self.target_fs, 120)
                    df_i = resample_dataframe(df_i, self.target_fs, 148)
                    
                    vals = {
                        'p': df_p.values, 'e': df_e.values, 
                        's': df_s.values, 'i': df_i.values
                    }
                    
                    # Truncate to min length
                    min_len = min(len(v) for v in vals.values())
                    for k in vals: vals[k] = vals[k][:min_len]

                    # Parse Labels
                    df_lbl.columns = df_lbl.columns.str.strip()
                    label_col = next((c for c in df_lbl.columns if 'action' in c.lower() or 'label' in c.lower()), df_lbl.columns[0])
                    start_col = df_lbl.columns[3] if len(df_lbl.columns) >= 4 else None
                    end_col = df_lbl.columns[5] if len(df_lbl.columns) >= 6 else None
                    
                    if not (start_col and end_col): continue

                    # Windowing
                    for _, row in df_lbl.iterrows():
                        try:
                            action = row[label_col]
                            t_start = float(str(row[start_col]).replace(',', '.'))
                            t_end = float(str(row[end_col]).replace(',', '.'))
                            
                            idx_start = int(t_start * self.target_fs)
                            idx_end = int(t_end * self.target_fs)
                            
                            if idx_start >= min_len: continue
                            if idx_end > min_len: idx_end = min_len
                            
                            # Check window size
                            seg_len = idx_end - idx_start
                            if seg_len >= window_len:
                                for i in range(0, seg_len - window_len + 1, step_len):
                                    start = idx_start + i
                                    end = start + window_len
                                    
                                    buffers['p'].append(vals['p'][start:end])
                                    buffers['e'].append(vals['e'][start:end])
                                    buffers['s'].append(vals['s'][start:end])
                                    buffers['i'].append(vals['i'][start:end])
                                    buffers['y'].append(action)
                                    self.metadata.append((subj, seq_id))
                        except ValueError: continue
                        
                except Exception as e:
                    logger.error(f"Err {subj}/{seq_id}: {e}")
                    continue

        if not buffers['p']: raise ValueError("Multimodal Dataset Empty.")

        # Convert to arrays
        self.X = {
            'plantar': np.array(buffers['p'], dtype=np.float32),
            'emg': np.array(buffers['e'], dtype=np.float32),
            'skeleton': np.array(buffers['s'], dtype=np.float32),
            'imu': np.array(buffers['i'], dtype=np.float32)
        }
        y_list = buffers['y']

        # Labels
        if self.le is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(y_list)
        else:
            known_mask = np.isin(y_list, self.le.classes_)
            if not np.all(known_mask):
                for k in self.X: self.X[k] = self.X[k][known_mask]
                y_list = np.array(y_list)[known_mask]
                self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if known_mask[i]]
            self.y = self.le.transform(y_list)

        # Scaling
        for k in self.X:
            N, T, F = self.X[k].shape
            if self.fit_scaler:
                self.X[k] = self.scalers[k].fit_transform(self.X[k].reshape(-1, F)).reshape(N, T, F)
            elif self.scalers[k]:
                self.X[k] = self.scalers[k].transform(self.X[k].reshape(-1, F)).reshape(N, T, F)

    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        # Return tuple of 4 tensors
        return (
            torch.tensor(self.X['plantar'][idx]),
            torch.tensor(self.X['emg'][idx]),
            torch.tensor(self.X['skeleton'][idx]),
            torch.tensor(self.X['imu'][idx])
        ), torch.tensor(self.y[idx], dtype=torch.long), self.metadata[idx]