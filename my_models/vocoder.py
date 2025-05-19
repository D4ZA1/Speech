import torch
import torch.nn as nn


class HiFiGANVocoder(nn.Module):


    def __init__(self):
        super().__init__()
        try:
            # Load HiFi-GAN from NVIDIA's repository, explicitly to CPU
            self.hifigan, self.vocoder_train_setup, self.denoiser = torch.hub.load(
                'NVIDIA/DeepLearningExamples:torchhub',
                'nvidia_hifigan',
                map_location=torch.device('cpu')  # Force to CPU
            )
            self.hifigan.eval()  # Set to eval mode
            self.hifigan.to('cpu') # Move the model to CPU
        except Exception as e:
            print("Error loading NVIDIA HiFi-GAN:")
            print(e)
            print("Please check your internet connection and NVIDIA HiFi-GAN details.")
            self.hifigan = None  # Vocoder loading failed

    def forward(self, mel_spectrogram):
        """
        mel_spectrogram: [B, 80, T] (or the input shape expected by NVIDIA's HiFi-GAN)
        """
        if self.hifigan is None:
            raise RuntimeError("HiFi-GAN vocoder model is not loaded.")
        with torch.no_grad():
            audio = self.hifigan(mel_spectrogram.to('cpu')) # Ensure input is on CPU
        return audio

