import torch.nn as nn
import torchaudio

class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()
        self.model = torchaudio.models.wav2vec2_base()

    def forward(self, x):
        return self.model(x)

