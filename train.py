
# Final Training Notebook: speech_style_transfer_training_final.ipynb

import os

import kagglehub
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset

from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.style_modulator import StyleModulator
from models.vocoder import Vocoder

# --- CONFIG ---
SAMPLE_RATE = 16000

AUDIO_LEN = 3 * SAMPLE_RATE  # 3 seconds fixed length

# --- DOWNLOAD DATASET (RAVDESS) ---
print("Downloading RAVDESS dataset...")
DATA_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
print("Dataset downloaded at:", DATA_PATH)

# --- DATASET CLASS ---
class EmotionSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.file_list = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_list.append(os.path.join(root, f))
        print(f"Found {len(self.file_list)} .wav files in dataset.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, size=AUDIO_LEN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db).float().T  # [T, 80] to match [B, T, 80] expectation
        wav_tensor = torch.tensor(y).float()
        return wav_tensor.unsqueeze(0), mel_tensor  # [1, T], [T, 80]

# --- MODEL INITIALIZATION ---
print("Initializing models...")
content_encoder = ContentEncoder()
speaker_encoder = SpeakerEncoder()
style_modulator = StyleModulator(input_dim=768, style_dim=256)
vocoder = Vocoder()
print("Models initialized.")

# --- LOSS & OPTIMIZER ---
recon_loss_fn = nn.L1Loss()
optimizer = optim.Adam(list(content_encoder.parameters()) +
                       list(speaker_encoder.parameters()) +
                       list(style_modulator.parameters()) +
                       list(vocoder.parameters()), lr=1e-4)
print("Optimizer and loss function set.")

# --- TRAINING LOOP ---
print("Preparing DataLoader...")
dataset = EmotionSpeechDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print("Starting training loop...")

for epoch in range(10):
    running_loss = 0.0
    for i, (wave, mel) in enumerate(dataloader):
        wave = wave.squeeze(1)  # shape: [B, T]
        content_feat = content_encoder(wave)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # [T, 80] -> [1, T, 80]

        # Final safety check: ensure mel is [B, T, 80]
        assert mel.shape[2] == 80, f"mel shape invalid before speaker encoder: {mel.shape}"

        # Pad or trim mel to length 300 frames
        current_len = mel.size(1)
        if current_len < 300:
            padding = torch.zeros(mel.size(0), 300 - current_len, 80)
            mel = torch.cat([mel, padding], dim=1)
        elif current_len > 300:
            mel = mel[:, :300, :]  # truncate

        # Normalize mel per sample
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)
        mel_std = mel.std(dim=(1, 2), keepdim=True) + 1e-5
        mel = (mel - mel_mean) / mel_std

        speaker_embed = speaker_encoder(mel)
        style_embed = speaker_encoder(mel)

        modulated = style_modulator(content_feat, style_embed)
        reconstructed = vocoder(modulated)

        if reconstructed.dim() == 3:
            reconstructed = reconstructed.squeeze(1)

        loss = recon_loss_fn(reconstructed, wave)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Batch Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}\n")

# --- SAVE CHECKPOINT ---
print("Saving model checkpoint...")
torch.save({
    'content_encoder': content_encoder.state_dict(),
    'speaker_encoder': speaker_encoder.state_dict(),
    'style_modulator': style_modulator.state_dict(),
    'vocoder': vocoder.state_dict()
}, 'speech_style_model.pth')
print("Model saved as speech_style_model.pth")

