
import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):

    def __init__(self, input_dim=80, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)  # global average pooling
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # x is expected to be [B, T, 80]
        x = x.permute(0, 2, 1)  # [B, T, 80] -> [B, 80, T]
        x = self.conv(x)        # [B, 256, T]
        x = self.pooling(x)     # [B, 256, 1]
        x = self.projection(x)  # [B, embedding_dim]
        return x

