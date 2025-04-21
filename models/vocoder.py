import torch
import torch.nn as nn


class Vocoder(nn.Module):


    def __init__(self):
        super(Vocoder, self).__init__()
        print("Loading HiFi-GAN vocoder from PyTorch Hub...")
        self.model = torch.hub.load('nvidia/DeepLearningExamples:torchhub',
                                     'nvidia_hifigan',
                                     model_math='fp32')
        self.model.eval()

    def forward(self, mel_spec):
        if mel_spec.dim() == 3:
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-9)
        return self.model(mel_spec)
