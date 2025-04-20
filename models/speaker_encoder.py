import torch.nn as nn

class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
