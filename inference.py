import librosa
import torch

from config import Config
from models.content_encoder import ContentEncoder
from models.speaker_encoder import SpeakerEncoder
from models.style_encoder import StyleEncoder
from models.style_modulator import StyleModulator
from models.vocoder import HiFiGANVocoder
from utils.audio import extract_mel_spectrogram, load_audio


def preprocess_input(audio_path, target_sr=16000, audio_length=3):

    """Loads, resamples, and pads/crops the input audio."""

    audio = load_audio(audio_path, sr=target_sr)
    audio = audio[: target_sr * audio_length]  # Truncate if longer
    if len(audio) < target_sr * audio_length:
        padding = np.zeros(target_sr * audio_length - len(audio))
        audio = np.concatenate([audio, padding])
    audio = audio.astype(np.float32)
    audio = (audio - np.mean(audio)) / (np.std(audio) + 1e-9)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    return audio_tensor

def main():
    # 1. Load Models
    content_encoder = ContentEncoder(Config.CONTENT_ENCODER_OUTPUT_DIM).to(Config.DEVICE)
    speaker_encoder = SpeakerEncoder(80, Config.SPEAKER_ENCODER_EMBED_DIM).to(Config.DEVICE)
    style_encoder = StyleEncoder(80, Config.STYLE_ENCODER_EMBED_DIM).to(Config.DEVICE)
    style_modulator = StyleModulator(Config.CONTENT_ENCODER_OUTPUT_DIM, Config.SPEAKER_ENCODER_EMBED_DIM, Config.STYLE_ENCODER_EMBED_DIM, Config.STYLE_MODULATOR_HIDDEN_DIM).to(Config.DEVICE)
    vocoder = HiFiGANVocoder().to(Config.DEVICE)

    checkpoint = torch.load("speech_style_transfer_model.pth", map_location=Config.DEVICE)  # Load checkpoint
    content_encoder.load_state_dict(checkpoint["content_encoder"])
    speaker_encoder.load_state_dict(checkpoint["speaker_encoder"])
    style_encoder.load_state_dict(checkpoint["style_encoder"])
    style_modulator.load_state_dict(checkpoint["style_modulator"])
    vocoder.load_state_dict(checkpoint["vocoder"])

    content_encoder.eval()
    speaker_encoder.eval()
    style_encoder.eval()
    style_modulator.eval()
    vocoder.eval()

    # 2. Preprocess Input
    input_audio = preprocess_input("input_speech.wav", target_sr=Config.SAMPLE_RATE, audio_length=3).to(Config.DEVICE)
    speaker_reference = extract_mel_spectrogram(load_audio("speaker_reference.wav"), sr=Config.SAMPLE_RATE).unsqueeze(0).to(Config.DEVICE)  # [1, T, 80]
    style_reference = extract_mel_spectrogram(load_audio("style_reference.wav"), sr=Config.SAMPLE_RATE).unsqueeze(0).to(Config.DEVICE)  # [1, T, 80]

    # 3. Inference
    with torch.no_grad():
        content_features = content_encoder(input_audio)  # [1, T, content_dim]
        speaker_embedding = speaker_encoder(speaker_reference)  # [1, speaker_dim]
        style_embedding = style_encoder(style_reference)  # [1, style_dim]
        modulated_content = style_modulator(content_features, speaker_embedding, style_embedding)  # [1, T, content_dim]
        output_audio = vocoder(modulated_content.transpose(1,2))  # [1, 1, T]

    # 4. Save Output
    output_audio = output_audio.squeeze().cpu().numpy()
    librosa.save_audio("output.wav", output_audio, sr=Config.SAMPLE_RATE)
    print("Inference complete. Output saved to 'output.wav'")

if __name__ == "__main__":
    main()
