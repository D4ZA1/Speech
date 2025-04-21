
import librosa
import librosa.feature
import numpy as np
import torch
import torch.nn as nn


class Vocoder(nn.Module):

    def __init__(self, sr=16000, n_fft=1024, hop_length=256):
        super(Vocoder, self).__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, mel_spec):
        """
        mel_spec: Tensor of shape [B, 80, T] or [80, T]
        """
        if isinstance(mel_spec, torch.Tensor):
            mel_spec = mel_spec.squeeze().cpu().numpy()

        # Invert mel-spectrogram to linear
        mel_spec = librosa.db_to_power(mel_spec)  # from dB
        mel_basis = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=80)
        inv_mel = np.dot(np.linalg.pinv(mel_basis), mel_spec)

        # Apply Griffin-Lim to get waveform
        waveform = librosa.griffinlim(inv_mel, n_iter=60, hop_length=self.hop_length, win_length=self.n_fft)

        return torch.tensor(waveform).unsqueeze(0).unsqueeze(0)  # shape [1, 1, T]

