import torch
import torch.nn as nn
from transformers import Wav2Vec2Model  # Import from transformers


class ContentEncoder(nn.Module):


    def __init__(self, output_dim, pretrained=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base") if pretrained else Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")  # Load from transformers
        self.projection = nn.Linear(768, output_dim)
        self.freeze_base = pretrained

        if self.freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: [B, 1, T] (waveform)
        """
        print(f"Shape of input to wav2vec2: {x.shape}")  # Debugging
        with torch.no_grad() if self.freeze_base else torch.enable_grad():
            features = self.wav2vec2(x).last_hidden_state  # [B, T', 768]
        content = self.projection(features)  # [B, T', output_dim]
        return content
