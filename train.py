
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.style_modulator import StyleModulator
from models.vocoder import Vocoder

# --- CONFIG ---
SAMPLE_RATE = 16000
AUDIO_LEN = 3 * SAMPLE_RATE  # 3 seconds fixed length
DATA_PATH = "./data/CREMA-D"

# --- DATASET ---
class EmotionSpeechDataset(Dataset):
    def __init__(self, root_dir):
        self.file_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        y, sr = librosa.load(file, sr=SAMPLE_RATE)
        y = librosa.util.fix_length(y, AUDIO_LEN)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.tensor(y).unsqueeze(0), torch.tensor(mel_db)

# --- MODELS ---
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
train_dataset = EmotionSpeechDataset(DATA_PATH)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

for epoch in range(10):
    for i, (wave, mel) in enumerate(train_loader):
        content_feat = content_encoder(wave)
        speaker_embed = speaker_encoder(mel)
        style_embed = speaker_encoder(mel)  # For demo, same as speaker

        modulated = style_modulator(content_feat, style_embed)
        reconstructed = vocoder(modulated)

        loss = recon_loss_fn(reconstructed, wave)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

# --- SAVE MODEL ---
torch.save({
    'content_encoder': content_encoder.state_dict(),
    'speaker_encoder': speaker_encoder.state_dict(),
    'style_modulator': style_modulator.state_dict(),
    'vocoder': vocoder.state_dict()
}, 'speech_style_model.pth')
