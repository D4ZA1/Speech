import torch.nn as nn

class StyleModulator(nn.Module):
    def __init__(self, input_dim, style_dim):
        super(StyleModulator, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + style_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, content, style):
        combined = torch.cat([content, style], dim=-1)
        return self.fusion(combined)
