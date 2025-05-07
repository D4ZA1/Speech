import torch
import torch.nn as nn


class StyleModulator(nn.Module):

    def __init__(self, input_dim, style_dim, hidden_dim=512, dropout=0.2):
        super(StyleModulator, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim + style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, content, style):
        # Handle tuple output from content encoder if needed
        if isinstance(content, tuple):
            content = content[0]
        if isinstance(style, tuple):
            style = style[0]
        if content.dim() != style.dim():
            style = style.unsqueeze(1).expand(-1, content.size(1), -1)
        combined = torch.cat([content, style], dim=-1)
        return self.fusion(combined)

