import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Mécanisme d'attention simple pour pondérer les pas de temps importants."""
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [Batch, Time, Hidden_Dim]
        weights = self.attention(x)             # [Batch, Time, 1]
        weights = F.softmax(weights, dim=1)     # Normalisation sur l'axe temporel
        context = torch.sum(weights * x, dim=1) 
        return context, weights

class ConvBlock(nn.Module):
    def __init__(self, input_dim, filters=64, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        return x

class HybridTwoStreamModel(nn.Module):
    def __init__(self, input_dim_plantar, input_dim_emg, hidden_dim, num_classes):
        super(HybridTwoStreamModel, self).__init__()
        
        # --- Branche Plantar ---
        self.cnn_plantar = ConvBlock(input_dim_plantar, filters=64)
        self.lstm_plantar = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True, 
            dropout=0.0 # CORRECTION: Mis à 0 pour éviter le warning
        )
        
        # --- Branche EMG ---
        self.cnn_emg = ConvBlock(input_dim_emg, filters=64)
        self.lstm_emg = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 # CORRECTION: Mis à 0 pour éviter le warning
        )
        
        # --- Dropout Manuel pour remplacer celui du LSTM ---
        self.lstm_dropout = nn.Dropout(0.3) 

        # --- Attention & Fusion ---
        self.attention = Attention(hidden_dim * 2)
        
        fusion_dim = (hidden_dim * 2) * 2 
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_plantar, x_emg):
        # 1. CNN
        feat_p = self.cnn_plantar(x_plantar) 
        feat_e = self.cnn_emg(x_emg)
        
        # 2. LSTM
        out_p, _ = self.lstm_plantar(feat_p)
        out_e, _ = self.lstm_emg(feat_e)
        
        # 3. Application du Dropout MANUEL ici
        out_p = self.lstm_dropout(out_p)
        out_e = self.lstm_dropout(out_e)
        
        # 4. Attention
        ctx_p, _ = self.attention(out_p)
        ctx_e, _ = self.attention(out_e)
        
        # 5. Fusion & Classif
        combined = torch.cat((ctx_p, ctx_e), dim=1)
        logits = self.classifier(combined)
        
        return logits