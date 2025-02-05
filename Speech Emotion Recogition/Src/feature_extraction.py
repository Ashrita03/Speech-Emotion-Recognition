import librosa
import numpy as np

def extract_features(file_path):
    """Extracts MFCC, Chroma, and Mel spectrogram features from an audio file."""
    audio, sr = librosa.load(file_path, sr=None)
    
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel])

# Example usage
if __name__ == "__main__":
    file_path = "example.wav"  # Replace with your actual file
    features = extract_features(file_path)
    print("Extracted Features:", features)
