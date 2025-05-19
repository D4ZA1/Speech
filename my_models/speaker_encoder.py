import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Linear(512, embed_dim)

    def forward(self, mel_spectrogram):
        """
        mel_spectrogram: [B, T, 80]
        """
        x = mel_spectrogram.transpose(1, 2)  # [B, 80, T]
        x = self.conv_layers(x)  # [B, 512, T']
        x = self.global_avg_pool(x).squeeze(-1)  # [B, 512]
        speaker_embedding = self.embedding_layer(x)  # [B, embed_dim]
        return speaker_embedding
