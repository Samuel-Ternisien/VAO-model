import os
# import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import get_logger, resample_dataframe
from config import CONFIG

logger = get_logger()

class BaseMindScanDataset(Dataset):
    """Classe parente pour gérer la logique commune (fenêtrage, labels, encodage)"""
    def __init__(self, root_dir, subjects, sequences, label_encoder=None, fit_scaler=False, scaler=None):
        self.samples = []
        self.metadata = []
        self.X = None
        self.y = None
        self.le = label_encoder
        self.scaler = scaler
        self.fit_scaler = fit_scaler
        
        # À définir dans les enfants
        self.data_folder = ""
        self.filename = ""
        self.original_fs = 100 
        
    def _load_data(self, subjects, sequences, root_dir):
        X_list = []
        y_list = []
        
        window_len = int(CONFIG['window_size_sec'] * CONFIG['target_fs'])
        step_len = int(CONFIG['step_size_sec'] * CONFIG['target_fs'])
        
        logger.info(f"[{self.__class__.__name__}] Chargement : {len(subjects)} sujets x {len(sequences)} séquences...")
        
        for subj in subjects:
            for seq_id in sequences:
                # Construction des chemins
                path_data = os.path.join(root_dir, self.data_folder, subj, f"Sequence_{seq_id}", self.filename)
                # Le dossier Events peut être "Sequence_" ou "Sequences_" selon l'historique, on check les deux ou on fixe
                # Ici on assume Sequence_XX pour matcher la demande EMG, mais on garde la logique de check
                path_labels = os.path.join(root_dir, "Events", subj, f"Sequence_{seq_id}", "classif.csv")
                
                if not os.path.exists(path_data): continue
                if not os.path.exists(path_labels): continue
                
                try:
                    # Chargement spécifique à implémenter dans l'enfant si besoin
                    df_data = self._read_data_file(path_data)
                    df_classif = pd.read_csv(path_labels, sep=';')
                except Exception as e:
                    logger.error(f"Erreur lecture {subj}/{seq_id}: {e}")
                    continue
                
                # Nettoyage Labels
                df_classif.columns = df_classif.columns.str.strip()
                label_col, start_col, end_col = self._parse_label_columns(df_classif)
                if not (start_col and end_col and label_col): continue

                # Resampling
                df_resampled = resample_dataframe(df_data, CONFIG['target_fs'], original_fs=self.original_fs)
                data_values = df_resampled.values 
                
                # Windowing
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
                                window = segment[i : i + window_len]
                                X_list.append(window)
                                y_list.append(action)
                                self.metadata.append((subj, seq_id))
                    except ValueError:
                        continue 
        
        return X_list, y_list

    def _read_data_file(self, path):
        # Par défaut, lecture CSV standard
        return pd.read_csv(path, sep=';')

    def _parse_label_columns(self, df):
        # Logique de détection des colonnes labels (identique au script précédent)
        label_col = next((c for c in df.columns if 'action' in c.lower() or 'label' in c.lower() or 'cat' in c.lower()), None)
        start_cols = [c for c in df.columns if ('start' in c.lower() or 'debut' in c.lower()) and 'time' in c.lower()]
        end_cols = [c for c in df.columns if ('end' in c.lower() or 'fin' in c.lower()) and 'time' in c.lower()]
        
        if not label_col: label_col = df.columns[0]
        start_col = start_cols[0] if start_cols else (df.columns[3] if len(df.columns)>=4 else None)
        end_col = end_cols[0] if end_cols else (df.columns[5] if len(df.columns)>=6 else None)
        return label_col, start_col, end_col

    def _finalize_init(self, X_list, y_list):
        if not X_list:
            logger.critical(f"[{self.__class__.__name__}] Aucune donnée chargée !")
            raise ValueError("Dataset vide.")

        self.X = np.array(X_list, dtype=np.float32)
        
        # Encodage Labels
        if self.le is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(y_list)
        else:
            # Filtrage des classes inconnues
            known_mask = np.isin(y_list, self.le.classes_)
            if not np.all(known_mask):
                logger.warning(f"Ignored {np.sum(~known_mask)} samples (unknown classes)")
                self.X = self.X[known_mask]
                y_list = np.array(y_list)[known_mask]
                self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if known_mask[i]]
            self.y = self.le.transform(y_list)

        # Scaling
        N, T, F = self.X.shape
        if self.fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X.reshape(-1, F)).reshape(N, T, F)
        else:
            if self.scaler:
                self.X = self.scaler.transform(self.X.reshape(-1, F)).reshape(N, T, F)
    
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long), self.metadata[idx]


class PlantarDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "Plantar_activity"
        self.filename = "insoles.csv"
        self.original_fs = 100
        
        X_l, y_l = self._load_data(subjects, sequences, root_dir)
        self._finalize_init(X_l, y_l)


class EMGDataset(BaseMindScanDataset):
    def __init__(self, root_dir, subjects, sequences, **kwargs):
        super().__init__(root_dir, subjects, sequences, **kwargs)
        self.data_folder = "EMG"
        self.filename = "emg.csv"
        # Fréquence approx EMG (Souvent >1000Hz, ex: 1259Hz ou 2000Hz). 
        # Important pour le resampling correct vers 50Hz.
        # D'après le header fourni "8.34e-05" de delta t ~= 1/12000 ? 
        # Si delta_t = 0.0000834s -> fs = 11990 Hz? 
        # On va calculer la fs dynamiquement dans _read_data_file si possible, 
        # sinon on fixe une moyenne. Ici on va utiliser le timestamps.
        self.original_fs = 1259 # Valeur par défaut, sera écrasée
        
        X_l, y_l = self._load_data(subjects, sequences, root_dir)
        self._finalize_init(X_l, y_l)

    def _read_data_file(self, path):
        # Lecture avec séparateur point-virgule
        df = pd.read_csv(path, sep=';')
        
        # Le header fourni est : EMG TIME; Arm_Left EMG (mV); ...
        # On doit supprimer la colonne Temps pour les features
        # Et on peut estimer la fréquence d'échantillonnage réelle
        
        # Nettoyage colonnes
        df.columns = df.columns.str.strip()
        time_col = [c for c in df.columns if 'TIME' in c][0]
        
        # Estimation Fs
        if len(df) > 1:
            dt = df[time_col].iloc[1] - df[time_col].iloc[0]
            if dt > 0:
                self.original_fs = 1.0 / dt
        
        # Suppression de la colonne temps pour ne garder que les capteurs
        df_features = df.drop(columns=[time_col])
        
        return df_features
class MultimodalDataset(Dataset):
    def __init__(self, root_dir, subjects, sequences, label_encoder=None, fit_scaler=False, scalers=None):
        """
        Gère Plantar + EMG synchronisés.
        scalers: dict {'plantar': scaler_obj, 'emg': scaler_obj}
        """
        self.samples = []
        self.metadata = []
        
        self.X_plantar = None
        self.X_emg = None
        self.y = None
        
        self.le = label_encoder
        self.fit_scaler = fit_scaler
        
        # Initialisation des Scalers
        if scalers:
            self.scaler_plantar = scalers.get('plantar')
            self.scaler_emg = scalers.get('emg')
        else:
            self.scaler_plantar = StandardScaler() if fit_scaler else None
            self.scaler_emg = StandardScaler() if fit_scaler else None

        # Paramètres
        self.target_fs = CONFIG['target_fs']
        window_len = int(CONFIG['window_size_sec'] * self.target_fs)
        step_len = int(CONFIG['step_size_sec'] * self.target_fs)
        
        # Listes temporaires
        X_p_list = []
        X_e_list = []
        y_list = []
        
        logger.info(f"[Multimodal] Chargement : {len(subjects)} sujets x {len(sequences)} séquences...")
        
        for subj in subjects:
            for seq_id in sequences:
                # Chemins
                path_plantar = os.path.join(root_dir, "Plantar_activity", subj, f"Sequence_{seq_id}", "insoles.csv")
                path_emg = os.path.join(root_dir, "EMG", subj, f"Sequence_{seq_id}", "emg.csv")
                path_labels = os.path.join(root_dir, "Events", subj, f"Sequence_{seq_id}", "classif.csv")
                
                if not (os.path.exists(path_plantar) and os.path.exists(path_emg) and os.path.exists(path_labels)):
                    continue
                
                try:
                    # 1. Lecture
                    df_p = pd.read_csv(path_plantar, sep=';')
                    df_e = pd.read_csv(path_emg, sep=';')
                    df_lbl = pd.read_csv(path_labels, sep=';')
                    
                    # 2. Nettoyage EMG (retirer colonne TIME)
                    df_e.columns = df_e.columns.str.strip()
                    time_cols = [c for c in df_e.columns if 'TIME' in c]
                    if time_cols: df_e = df_e.drop(columns=time_cols)
                    
                    # 3. Resampling vers 50Hz pour ALIGNER les séries
                    # Plantar est ~100Hz -> 50Hz
                    df_p_res = resample_dataframe(df_p, self.target_fs, original_fs=100)
                    # EMG est ~1259Hz -> 50Hz
                    df_e_res = resample_dataframe(df_e, self.target_fs, original_fs=1259)
                    
                    vals_p = df_p_res.values
                    vals_e = df_e_res.values
                    
                    # Si les longueurs diffèrent légèrement après resampling, on tronque au min
                    min_len = min(len(vals_p), len(vals_e))
                    vals_p = vals_p[:min_len]
                    vals_e = vals_e[:min_len]

                    # 4. Parsing Labels
                    df_lbl.columns = df_lbl.columns.str.strip()
                    label_col = next((c for c in df_lbl.columns if 'action' in c.lower() or 'label' in c.lower()), df_lbl.columns[0])
                    # Utilisation des indices par défaut si noms non trouvés (Start: 3, End: 5)
                    start_col = df_lbl.columns[3] if len(df_lbl.columns) >= 4 else None
                    end_col = df_lbl.columns[5] if len(df_lbl.columns) >= 6 else None
                    
                    if not (start_col and end_col): continue

                    # 5. Extraction Fenêtres
                    for _, row in df_lbl.iterrows():
                        try:
                            action = row[label_col]
                            t_start = float(str(row[start_col]).replace(',', '.'))
                            t_end = float(str(row[end_col]).replace(',', '.'))
                            
                            idx_start = int(t_start * self.target_fs)
                            idx_end = int(t_end * self.target_fs)
                            
                            if idx_start >= min_len: continue
                            if idx_end > min_len: idx_end = min_len
                            
                            seg_p = vals_p[idx_start:idx_end]
                            seg_e = vals_e[idx_start:idx_end]
                            
                            if len(seg_p) >= window_len:
                                for i in range(0, len(seg_p) - window_len + 1, step_len):
                                    X_p_list.append(seg_p[i : i + window_len])
                                    X_e_list.append(seg_e[i : i + window_len])
                                    y_list.append(action)
                                    self.metadata.append((subj, seq_id))
                        except ValueError: continue
                        
                except Exception as e:
                    logger.error(f"Err {subj}/{seq_id}: {e}")
                    continue

        if not X_p_list:
            raise ValueError("Aucune donnée multimodale chargée.")

        self.X_plantar = np.array(X_p_list, dtype=np.float32)
        self.X_emg = np.array(X_e_list, dtype=np.float32)
        
        # Encodage Labels
        if self.le is None:
            self.le = LabelEncoder()
            self.y = self.le.fit_transform(y_list)
        else:
            # Filtre classes inconnues
            known_mask = np.isin(y_list, self.le.classes_)
            if not np.all(known_mask):
                self.X_plantar = self.X_plantar[known_mask]
                self.X_emg = self.X_emg[known_mask]
                y_list = np.array(y_list)[known_mask]
                self.metadata = [self.metadata[i] for i in range(len(self.metadata)) if known_mask[i]]
            self.y = self.le.transform(y_list)

        # Scaling INDÉPENDANT pour chaque modalité
        N, T, Fp = self.X_plantar.shape
        _, _, Fe = self.X_emg.shape
        
        if self.fit_scaler:
            self.X_plantar = self.scaler_plantar.fit_transform(self.X_plantar.reshape(-1, Fp)).reshape(N, T, Fp)
            self.X_emg = self.scaler_emg.fit_transform(self.X_emg.reshape(-1, Fe)).reshape(N, T, Fe)
        else:
            if self.scaler_plantar:
                self.X_plantar = self.scaler_plantar.transform(self.X_plantar.reshape(-1, Fp)).reshape(N, T, Fp)
            if self.scaler_emg:
                self.X_emg = self.scaler_emg.transform(self.X_emg.reshape(-1, Fe)).reshape(N, T, Fe)

    def __len__(self): return len(self.y)
    
    def __getitem__(self, idx):
        # Retourne (X_plantar, X_emg), y, metadata
        return (torch.tensor(self.X_plantar[idx]), torch.tensor(self.X_emg[idx])), \
               torch.tensor(self.y[idx], dtype=torch.long), \
               self.metadata[idx]