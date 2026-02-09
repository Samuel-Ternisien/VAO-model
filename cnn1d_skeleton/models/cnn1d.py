from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # input: (B, T, F) -> transpose to (B, F, T)
        self.conv1 = nn.Conv1d(in_features, 256, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B,T,F)
        x = x.transpose(1, 2)  # (B,F,T)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        x = self.drop(F.relu(self.bn3(self.conv3(x))))
        x = x.mean(dim=-1)     # global average pooling over T
        return self.fc(x)