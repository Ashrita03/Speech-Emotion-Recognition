import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from feature_extraction import extract_features

def load_data(dataset_path, test_size=0.2):
    """Loads dataset and extracts features."""
    df = pd.read_csv(dataset_path)  # CSV file with paths and emotions
    x, y = [], []
    
    for index, row in df.iterrows():
        features = extract_features(row["Path"])
        x.append(features)
        y.append(row["Emotion"])

    return train_test_split(np.array(x), y, test_size=test_size, random_state=42)

# Train the model
if __name__ == "__main__":
    dataset_csv = "dataset.csv"  # Replace with actual dataset CSV
    x_train, x_test, y_train, y_test = load_data(dataset_csv)

    model = MLPClassifier(hidden_layer_sizes=(300,), max_iter=500, learning_rate='adaptive')
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
