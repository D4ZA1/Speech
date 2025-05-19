
import os

import kagglehub
import torch
import torch.optim as optim

from config import Config

# Import models and modules
try:

    from my_models.content_encoder import ContentEncoder
    print("ContentEncoder imported successfully from my_models.content_encoder")
except ImportError as e:
    print(f"Error importing ContentEncoder: {e}")
    ContentEncoder = None

try:
    from my_models.speaker_encoder import SpeakerEncoder
    print("SpeakerEncoder imported successfully from my_models.speaker_encoder")
except ImportError as e:
    print(f"Error importing SpeakerEncoder: {e}")
    SpeakerEncoder = None

try:
    from my_models.style_encoder import StyleEncoder
    print("StyleEncoder imported successfully from my_models.style_encoder")
except ImportError as e:
    print(f"Error importing StyleEncoder: {e}")
    StyleEncoder = None

try:
    from my_modules.style_modulator import StyleModulator
    print("StyleModulator imported successfully from my_modules.style_modulator")
except ImportError as e:
    print(f"Error importing StyleModulator: {e}")
    StyleModulator = None

try:
    from my_models.vocoder import HiFiGANVocoder
    print("HiFiGANVocoder imported successfully from my_models.vocoder")
except ImportError as e:
    print(f"Error importing HiFiGANVocoder: {e}")
    HiFiGANVocoder = None

try:
    from my_modules.transfer_module import train_transfer_step
    print("train_transfer_step imported successfully from my_modules.transfer_module")
except ImportError as e:
    print(f"Error importing train_transfer_step: {e}")
    train_transfer_step = None

try:
    from my_utils.audio import load_audio
    print("load_audio imported successfully from my_utils.audio")
except ImportError as e:
    print(f"Error importing load_audio: {e}")
    load_audio = None

try:
    from my_utils.data import RAVDESSDataset, create_dataloader
    print("RAVDESSDataset and create_dataloader imported successfully from my_utils.data")
except ImportError as e:
    print(f"Error importing RAVDESSDataset and create_dataloader: {e}")
    RAVDESSDataset = None
    create_dataloader = None


def main():
    print("Downloading RAVDESS dataset...")
    DATA_PATH = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")
    print("Dataset downloaded at:", DATA_PATH)

    downloaded_path = os.path.join("data", "ravdess-emotional-speech-audio")
    if os.path.exists(downloaded_path):
        os.rename(downloaded_path, DATA_PATH)
    else:
        print(f"Warning: Downloaded directory not found at '{downloaded_path}'. Assuming download was successful.")

    if RAVDESSDataset:
        dataset = RAVDESSDataset(DATA_PATH, Config.AUDIO_LEN * Config.SAMPLE_RATE, Config.SAMPLE_RATE)
    else:
        print("RAVDESSDataset is not initialized, cannot proceed.")
        return

    if create_dataloader:
        dataloader = create_dataloader(dataset, Config.BATCH_SIZE, shuffle=True)
    else:
        print("create_dataloader is not initialized, cannot proceed.")
        return

    # Initialize models
    if ContentEncoder:
        content_encoder = ContentEncoder(Config.CONTENT_ENCODER_OUTPUT_DIM).to(Config.DEVICE)
    else:
        print("Content Encoder is not initialized")
        return

    if SpeakerEncoder:
        speaker_encoder = SpeakerEncoder(80, Config.SPEAKER_ENCODER_EMBED_DIM).to(Config.DEVICE)
    else:
        print("Speaker Encoder is not initialized")
        return

    if StyleEncoder:
        style_encoder = StyleEncoder(80, Config.STYLE_ENCODER_EMBED_DIM).to(Config.DEVICE)
    else:
        print("Style Encoder is not initialized")
        return

    if StyleModulator:
        style_modulator = StyleModulator(
            Config.CONTENT_ENCODER_OUTPUT_DIM,
            Config.SPEAKER_ENCODER_EMBED_DIM,
            Config.STYLE_ENCODER_EMBED_DIM,
            Config.STYLE_MODULATOR_HIDDEN_DIM,
        ).to(Config.DEVICE)
    else:
        print("Style Modulator is not initialized")
        return

    if HiFiGANVocoder:
        try:
            vocoder = HiFiGANVocoder()  # Load from local offline files
            if vocoder.hifigan is None:
                print("HiFi-GAN vocoder failed to initialize.")
                return
        except Exception as e:
            print(f"Error initializing HiFiGANVocoder: {e}")  # Print the full exception
            return
    else:
        print("Vocoder class not found.")
        return

    # Optimizer (excluding vocoder)
    optimizer = optim.Adam(
        list(content_encoder.parameters()) +
        list(speaker_encoder.parameters()) +
        list(style_encoder.parameters()) +
        list(style_modulator.parameters()),
        lr=Config.LEARNING_RATE,
    )

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for i, (wave, mel) in enumerate(dataloader):
            wave = wave.to(Config.DEVICE)
            mel = mel.to(Config.DEVICE)

            losses = train_transfer_step(
                content_encoder,
                speaker_encoder,
                style_encoder,
                style_modulator,
                vocoder,  # Inference only
                optimizer,
                wave,
                mel,
                Config,
            )
            total_loss += losses["loss"]
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {losses['loss']:.4f}")
        print(f"Epoch {epoch} Average Loss: {total_loss / len(dataloader):.4f}")

    # Save model
    torch.save(
        {
            "content_encoder": content_encoder.state_dict(),
            "speaker_encoder": speaker_encoder.state_dict(),
            "style_encoder": style_encoder.state_dict(),
            "style_modulator": style_modulator.state_dict(),
            "vocoder": vocoder.hifigan.state_dict(),
        },
        "speech_style_transfer_model.pth",
    )


if __name__ == "__main__":
    main()

