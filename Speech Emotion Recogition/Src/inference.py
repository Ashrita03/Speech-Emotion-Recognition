import pickle
import sys
from feature_extraction import extract_features

def predict_emotion(model_path, audio_path):
    """Loads trained model and predicts emotion from an audio file."""
    # Load trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Extract features from the new audio file
    features = extract_features(audio_path).reshape(1, -1)

    # Predict emotion
    prediction = model.predict(features)[0]
    return prediction

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model.pkl> <audio.wav>")
        sys.exit(1)

    model_file = sys.argv[1]
    audio_file = sys.argv[2]

    emotion = predict_emotion(model_file, audio_file)
    print(f"Predicted Emotion: {emotion}")
