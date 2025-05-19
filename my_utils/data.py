import os

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from my_utils.audio import (extract_mel_spectrogram, normalize_amplitude,
                            random_crop)


class RAVDESSDataset(Dataset):



    def __init__(self, root_dir, audio_length, sr):
        print(f"Initializing RAVDESSDataset with root_dir: {root_dir}")
        self.root_dir = root_dir
        self.file_paths = self._get_file_paths()
        self.audio_length = audio_length
        self.sr = sr
        print(f"Found {len(self.file_paths)} files.")

    def _get_file_paths(self):
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root directory '{self.root_dir}' does not exist.")

        file_paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(".wav"):
                    file_paths.append(os.path.join(dirpath, filename))
        return file_paths

    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        waveform, _ = torchaudio.load(audio_path)
        waveform = self._preprocess_audio(waveform)  # waveform: [T]
        mel_spectrogram = self._extract_mel_spectrogram(waveform.unsqueeze(0))  # [1, T] for mel
        print(f"Shape of waveform before returning: {waveform.shape}")  # Should print: torch.Size([T])
        return waveform, mel_spectrogram

    def _preprocess_audio(self, waveform):
        waveform = waveform.squeeze(0)  # From [1, T] -> [T]
        waveform = normalize_amplitude(waveform.numpy())
        waveform = random_crop(waveform, self.audio_length, sr=self.sr)
        return torch.tensor(waveform, dtype=torch.float32)  # Final shape: [T]
    def _extract_mel_spectrogram(self, waveform):
        mel_spectrogram = extract_mel_spectrogram(waveform, sr=self.sr)
        mel_spectrogram = mel_spectrogram.transpose(1, 2).squeeze(0)
        return mel_spectrogram


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
