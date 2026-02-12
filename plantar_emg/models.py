import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [Batch, Time, Hidden_Dim]
        weights = self.attention(x)             
        weights = F.softmax(weights, dim=1)     
        context = torch.sum(weights * x, dim=1) 
        return context, weights

class ConvBlock(nn.Module):
    def __init__(self, input_dim, filters=64):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        # [Batch, Time, Feat] -> [Batch, Feat, Time]
        x = x.permute(0, 2, 1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        # [Batch, Feat, Time] -> [Batch, Time, Feat]
        x = x.permute(0, 2, 1)
        return x

class Branch(nn.Module):
    """A single modality processing branch (CNN -> BiLSTM -> Attention)."""
    def __init__(self, input_dim, hidden_dim=64):
        super(Branch, self).__init__()
        self.cnn = ConvBlock(input_dim, filters=64)
        
        # LSTM input is 64 (from CNN filters)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.attention = Attention(hidden_dim * 2) # BiLSTM = 2 * hidden

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        ctx, _ = self.attention(x)
        return ctx

class HybridFourStreamModel(nn.Module):
    """Model specifically handling Plantar, EMG, Skeleton, IMU."""
    def __init__(self, input_dims, num_classes, hidden_dim=64):
        super(HybridFourStreamModel, self).__init__()
        
        # input_dims is a dict: {'plantar': int, 'emg': int, 'skeleton': int, 'imu': int}
        
        self.branch_p = Branch(input_dims['plantar'], hidden_dim)
        self.branch_e = Branch(input_dims['emg'], hidden_dim)
        self.branch_s = Branch(input_dims['skeleton'], hidden_dim)
        self.branch_i = Branch(input_dims['imu'], hidden_dim)
        
        # Fusion dimension: 4 branches * (Hidden * 2 directions)
        fusion_dim = 4 * (hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xp, xe, xs, xi):
        cp = self.branch_p(xp)
        ce = self.branch_e(xe)
        cs = self.branch_s(xs)
        ci = self.branch_i(xi)
        
        combined = torch.cat((cp, ce, cs, ci), dim=1)
        return self.classifier(combined)