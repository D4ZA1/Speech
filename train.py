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
        mel_tensor = torch.tensor(mel_db).float()  # [80, T]
        wav_tensor = torch.tensor(y).float()  # [T]
        return wav_tensor.unsqueeze(0), mel_tensor  # [1, T], [80, T]

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

        if mel.dim() == 3 and mel.shape[1] != 80:
            mel = mel.permute(0, 2, 1)  # [B, T, 80] -> [B, 80, T]

        # Pad or trim mel to length 300
        current_len = mel.size(-1)
        if current_len < 300:
            padding = torch.zeros(mel.size(0), mel.size(1), 300 - current_len)
            mel = torch.cat([mel, padding], dim=-1)
        elif current_len > 300:
            mel = mel[:, :, :300]  # truncate

        # Normalize mel (optional but helps stability)
        mel = (mel - mel.mean(dim=-1, keepdim=True)) / (mel.std(dim=-1, keepdim=True) + 1e-5)

        speaker_embed = speaker_encoder(mel)
        style_embed = speaker_encoder(mel)  # demo only: same as speaker

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
