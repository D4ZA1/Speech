import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def visualize_spectrograms(input_path, style_path, output_path):

    """
    Visualizes and compares spectrograms of input, style reference, and output audio.
    """

    # Load audio files
    input_audio, sr = librosa.load(input_path, sr=16000)
    style_audio, _ = librosa.load(style_path, sr=16000)
    output_audio, _ = librosa.load(output_path, sr=16000)

    # Extract Mel-spectrograms
    input_mel = librosa.feature.melspectrogram(y=input_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    style_mel = librosa.feature.melspectrogram(y=style_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    output_mel = librosa.feature.melspectrogram(y=output_audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)

    input_mel_db = librosa.power_to_db(input_mel, ref=np.max)
    style_mel_db = librosa.power_to_db(style_mel, ref=np.max)
    output_mel_db = librosa.power_to_db(output_mel, ref=np.max)

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    librosa.display.specshow(input_mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
    plt.title("Input Speech")

    plt.subplot(3, 1, 2)
    librosa.display.specshow(style_mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
    plt.title("Style Reference")

    plt.subplot(3, 1, 3)
    librosa.display.specshow(output_mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel')
    plt.title("Stylized Output")

    plt.tight_layout()
    plt.savefig("spectrogram_comparison.png")
    plt.show()

def calculate_cosine_similarity(embedding1, embedding2):
    """Calculates the cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    return dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0

def evaluate_embeddings(input_audio, style_audio, output_audio, model, config):
    """
    Evaluates the model's performance by comparing embeddings.
    """

    content_encoder = model["content_encoder"]
    speaker_encoder = model["speaker_encoder"]
    style_encoder = model["style_encoder"]

    input_audio_tensor = preprocess_input(input_audio, target_sr=config.SAMPLE_RATE, audio_length=3).to(config.DEVICE)
    style_audio_tensor = preprocess_input(style_audio, target_sr=config.SAMPLE_RATE, audio_length=3).to(config.DEVICE)

    input_content_embedding = content_encoder(input_audio_tensor).mean(dim=1).squeeze().cpu().numpy()
    output_content_embedding = content_encoder(preprocess_input(output_audio, target_sr=config.SAMPLE_RATE, audio_length=3).to(config.DEVICE)).mean(dim=1).squeeze().cpu().numpy()
    style_embedding = style_encoder(extract_mel_spectrogram(load_audio(style_audio), sr=config.SAMPLE_RATE).unsqueeze(0).to(config.DEVICE)).squeeze().cpu().numpy()
    output_style_embedding = style_encoder(extract_mel_spectrogram(load_audio(output_audio), sr=config.SAMPLE_RATE).unsqueeze(0).to(config.DEVICE)).squeeze().cpu().numpy()
    speaker_embedding = speaker_encoder(extract_mel_spectrogram(load_audio(input_audio), sr=config.SAMPLE_RATE).unsqueeze(0).to(config.DEVICE)).squeeze().cpu().numpy()
    output_speaker_embedding = speaker_encoder(extract_mel_spectrogram(load_audio(output_audio), sr=config.SAMPLE_RATE).unsqueeze(0).to(config.DEVICE)).squeeze().cpu().numpy()

    content_similarity = calculate_cosine_similarity(input_content_embedding, output_content_embedding)
    style_similarity = calculate_cosine_similarity(style_embedding, output_style_embedding)
    speaker_similarity = calculate_cosine_similarity(speaker_embedding, output_speaker_embedding)

    print(f"Content Similarity: {content_similarity:.4f}")
    print(f"Style Similarity: {style_similarity:.4f}")
    print(f"Speaker Similarity: {speaker_similarity:.4f}")

def main():
    # Example Usage
    input_audio_path = "input_speech.wav"
    style_audio_path = "style_reference.wav"
    output_audio_path = "output.wav"

    # Load models (replace with your model loading logic)
    content_encoder = ContentEncoder(Config.CONTENT_ENCODER_OUTPUT_DIM).to(Config.DEVICE)
    speaker_encoder = SpeakerEncoder(80, Config.SPEAKER_ENCODER_EMBED_DIM).to(Config.DEVICE)
    style_encoder = StyleEncoder(80, Config.STYLE_ENCODER_EMBED_DIM).to(Config.DEVICE)
    model = {
        "content_encoder": content_encoder,
        "speaker_encoder": speaker_encoder,
        "style_encoder": style_encoder,
    }
    checkpoint = torch.load("speech_style_transfer_model.pth", map_location=Config.DEVICE)  # Load checkpoint
    content_encoder.load_state_dict(checkpoint["content_encoder"])
    speaker_encoder.load_state_dict(checkpoint["speaker_encoder"])
    style_encoder.load_state_dict(checkpoint["style_encoder"])
    content_encoder.eval()
    speaker_encoder.eval()
    style_encoder.eval()

    visualize_spectrograms(input_audio_path, style_audio_path, output_audio_path)
    evaluate_embeddings(input_audio_path, style_audio_path, output_audio_path, model, Config)

if __name__ == "__main__":
    main()
