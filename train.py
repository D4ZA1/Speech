# Final Training Notebook: speech_style_transfer_training_final.ipynb

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import kagglehub
from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.style_modulator import StyleModulator
from models.vocoder import Vocoder

# --- CONFIG ---
SAMPLE_RATE = 16000
AUDIO_LEN = 3 * SAMPLE_RATE  # 3 seconds fixed length

# --- DOWNLOAD DATASET (RAVDESS) ---
DATA_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

# --- DATASET CLASS ---
class EmotionSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.file_list = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_list.append(os.path.join(root, f))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, AUDIO_LEN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(y).unsqueeze(0), torch.tensor(mel_db)

# --- MODEL INITIALIZATION ---
content_encoder = ContentEncoder()
speaker_encoder = SpeakerEncoder()
style_modulator = StyleModulator(input_dim=768, style_dim=256)
vocoder = Vocoder()

# --- LOSS & OPTIMIZER ---
recon_loss_fn = nn.L1Loss()
optimizer = optim.Adam(list(content_encoder.parameters()) +
                       list(speaker_encoder.parameters()) +
                       list(style_modulator.parameters()) +
                       list(vocoder.parameters()), lr=1e-4)

# --- TRAINING LOOP ---
dataset = EmotionSpeechDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(10):
    for i, (wave, mel) in enumerate(dataloader):
        content_feat = content_encoder(wave)
        speaker_embed = speaker_encoder(mel)
        style_embed = speaker_encoder(mel)  # demo only: same as speaker

        modulated = style_modulator(content_feat, style_embed)
        reconstructed = vocoder(modulated)

        loss = recon_loss_fn(reconstructed, wave)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

# --- SAVE CHECKPOINT ---
torch.save({
    'content_encoder': content_encoder.state_dict(),
    'speaker_encoder': speaker_encoder.state_dict(),
    'style_modulator': style_modulator.state_dict(),
    'vocoder': vocoder.state_dict()
}, 'speech_style_model.pth')
