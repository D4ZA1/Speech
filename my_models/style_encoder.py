import torch
import torch.nn as nn


class StyleEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Linear(256, embed_dim)

    def forward(self, mel_spectrogram):
        """
        mel_spectrogram: [B, T, 80]
        """
        x = mel_spectrogram.transpose(1, 2)  # [B, 80, T]
        x = self.conv_layers(x)  # [B, 256, T']
        x = self.global_avg_pool(x).squeeze(-1)  # [B, 256]
        style_embedding = self.embedding_layer(x)  # [B, embed_dim]
        return style_embedding
