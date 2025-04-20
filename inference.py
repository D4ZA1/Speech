import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.style_modulator import StyleModulator
from models.vocoder import Vocoder


content_encoder = ContentEncoder()
speaker_encoder = SpeakerEncoder()
style_modulator = StyleModulator(input_dim=768, style_dim=128)
vocoder = Vocoder()

checkpoint = torch.load('speech_style_model.pth', map_location='cpu')
content_encoder.load_state_dict(checkpoint['content_encoder'])
speaker_encoder.load_state_dict(checkpoint['speaker_encoder'])
style_modulator.load_state_dict(checkpoint['style_modulator'])
vocoder.load_state_dict(checkpoint['vocoder'])

content_encoder.eval()
speaker_encoder.eval()
style_modulator.eval()
vocoder.eval()


def preprocess_audio(path, sr=16000, duration=3):
    wav, _ = librosa.load(path, sr=sr)
    wav = librosa.util.fix_length(wav, sr * duration)
    return torch.tensor(wav).unsqueeze(0)


source_wav = preprocess_audio("input_speech.wav")
speaker_ref = preprocess_audio("speaker_reference.wav")
style_ref = preprocess_audio("style_reference.wav")

with torch.no_grad():
    content_feat = content_encoder(source_wav)
    speaker_embed = speaker_encoder(speaker_ref)
    style_embed = speaker_encoder(style_ref) 
    modulated_feat = style_modulator(content_feat, style_embed)
    output_wav = vocoder(modulated_feat)

sf.write("output.wav", output_wav.squeeze().numpy(), 16000)
print("Stylized output saved as output.wav")

