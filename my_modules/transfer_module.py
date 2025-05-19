import torch
import torch.nn as nn

from config import Config
from my_utils.train import compute_loss


def train_transfer_step(


    content_encoder,
    speaker_encoder,
    style_encoder,
    style_modulator,
    vocoder,
    optimizer,
    wave,
    mel,
    Config,
):
    content_encoder.train()
    speaker_encoder.train()
    style_encoder.train()
    style_modulator.train()
    vocoder.train()

    optimizer.zero_grad()

    # 1. Forward pass
    content_features = content_encoder(wave)  # [B, T', content_dim]
    speaker_embedding = speaker_encoder(mel)  # [B, speaker_dim]
    style_embedding = style_encoder(mel)  # [B, style_dim]
    modulated_content = style_modulator(
        content_features, speaker_embedding, style_embedding
    )  # [B, T', content_dim]

    #  Ensure modulated_content is [B, content_dim, T'] for the vocoder
    modulated_content = modulated_content.transpose(1, 2)  # [B, content_dim, T']
    reconstructed_audio = vocoder(modulated_content)  # [B, 1, T]

    # 2. Compute loss
    reconstruction_loss = nn.functional.l1_loss(
        reconstructed_audio.squeeze(1), wave.squeeze(1)
    )  # Adjust loss as needed
    loss = reconstruction_loss

    # 3. Backpropagate
    loss.backward()
    optimizer.step()

    return {"loss": loss.item(), "reconstruction_loss": reconstruction_loss.item()}
