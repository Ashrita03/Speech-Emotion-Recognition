import os
import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path):
    """Loads an audio file and returns the waveform and sample rate."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def remove_noise(audio, sr):
    """Applies noise reduction to the audio signal."""
    return librosa.effects.preemphasis(audio)

def save_audio(file_path, audio, sr):
    """Saves processed audio to a file."""
    sf.write(file_path, audio, sr)

# Example usage
if __name__ == "__main__":
    file_path = "example.wav"  # Replace with your actual file path
    audio, sr = load_audio(file_path)
    clean_audio = remove_noise(audio, sr)
    save_audio("clean_example.wav", clean_audio, sr)
