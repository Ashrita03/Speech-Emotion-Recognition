import os
import glob
import pandas as pd

# Emotion labels from RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_dataset(dataset_path):
    """Loads the dataset and maps emotion labels."""
    file_emotions = []
    file_paths = []
    
    for file in glob.glob(os.path.join(dataset_path, "Actor_*/*.wav")):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions.get(emotion_code, "Unknown")
        
        file_emotions.append(emotion)
        file_paths.append(file)

    # Create a DataFrame
    df = pd.DataFrame({"Emotion": file_emotions, "Path": file_paths})
    return df

# Example usage
if __name__ == "__main__":
    dataset_path = "path/to/RAVDESS"  # Replace with actual dataset path
    dataset = load_dataset(dataset_path)
    print(dataset.head())  # Show first few rows
