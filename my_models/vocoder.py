import json
import os

import torch
import torch.nn as nn

from hifi_gan.inference import \
    AttrDict  # <- This is what fixes the config issue
from hifi_gan.models import Generator  # From hifi-gan repo


class HiFiGANVocoder(nn.Module):


    def __init__(self, checkpoint_path="hifi_gan/g_02500000", config_path="hifi_gan/config.json"):
        super().__init__()
        if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
            print("HiFi-GAN checkpoint/config file not found.")
            self.hifigan = None
            return
        with open(config_path) as f:
            config_data = json.load(f)
        h = AttrDict(config_data)   # ✅ FIXED LINE
        self.hifigan = Generator(h).to("cpu")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.hifigan.load_state_dict(state_dict["generator"])
        self.hifigan.eval()
        print("HiFi-GAN vocoder loaded from local checkpoint.")

    def forward(self, mel):
        if self.hifigan is None:
            raise RuntimeError("HiFi-GAN model not loaded.")
        with torch.no_grad():
            return self.hifigan(mel)
