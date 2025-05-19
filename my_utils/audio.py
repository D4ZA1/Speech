import librosa
import numpy as np
import torch
import torchaudio


def load_audio(path, sr=16000):


    """Loads and resamples audio to the target sample rate."""
    wav, _ = librosa.load(path, sr=sr)
    return wav

def extract_mel_spectrogram(
    audio,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000,
):
    """Extracts a mel-spectrogram from the given audio."""

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )(torch.tensor(audio))
    mel_spectrogram = torchaudio.functional.DB_to_amplitude(mel_spectrogram, ref=1.0,power=1.0)
    return mel_spectrogram

def inverse_mel_spectrogram(
    mel_spectrogram,
    sr=16000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
    f_min=0.0,
    f_max=8000,
    n_iter=32,
):
    """Reconstructs audio from a mel-spectrogram using Griffin-Lim."""

    mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()
    inv_mel = librosa.amplitude_to_db(mel_spectrogram)
    waveform = librosa.feature.inverse.mel_to_audio(
        M=inv_mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        fmin=f_min,
        fmax=f_max,
        power=1.0,
        n_iter=n_iter,
    )
    return waveform

def normalize_amplitude(audio):
    """Normalizes the amplitude of the audio to the range [-1, 1]."""

    return audio / np.max(np.abs(audio))

def random_crop(audio, target_length, sr=16000):
    """Randomly crops the audio to the target length."""

    audio_len = len(audio)
    if audio_len < target_length:
        padding = np.zeros(target_length - audio_len)
        audio = np.concatenate([audio, padding])
    elif audio_len > target_length:
        start = np.random.randint(0, audio_len - target_length)
        audio = audio[start : start + target_length]
    return audio
