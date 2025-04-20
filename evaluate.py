import matplotlib.pyplot as plt
import librosa.display

# ---- Load Audio Files ----
source, sr = librosa.load("input_speech.wav", sr=16000)
target, _ = librosa.load("style_reference.wav", sr=16000)
output, _ = librosa.load("output.wav", sr=16000)

# ---- Generate Mel Spectrograms ----
def get_mel(wav):
    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
    return librosa.power_to_db(mel, ref=np.max)

mel_source = get_mel(source)
mel_target = get_mel(target)
mel_output = get_mel(output)

# ---- Plotting ----
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(mel_source, sr=16000, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Input Speech")

plt.subplot(3, 1, 2)
librosa.display.specshow(mel_target, sr=16000, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Style Reference")

plt.subplot(3, 1, 3)
librosa.display.specshow(mel_output, sr=16000, hop_length=256, x_axis='time', y_axis='mel')
plt.title("Stylized Output")

plt.tight_layout()
plt.savefig("spectrogram_comparison.png")
print("Spectrogram comparison saved as spectrogram_comparison.png")
