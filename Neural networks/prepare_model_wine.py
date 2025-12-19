"""Module for preparing and training the wine quality prediction model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import json
import os

def load_data(filepath, sep=';'):
    """Load the wine CSV file, scale features, and return X, y, and (mean, std).

    Args:
        filepath: path to the CSV file (default `winequality.csv`).
        sep: column separator in the CSV file.

    Returns:
        X_scaled: feature matrix (float32)
        y: binary labels (0/1)
        (mean, std): statistics used for scaling
    """
    df = pd.read_csv(filepath, sep=sep)
    data_array = df.values.astype('float32')
    target_idx = -1 
    
    # Extract features and label
    X_raw = np.delete(data_array, target_idx, axis=1)
    y_raw = data_array[:, target_idx]
    
    y = (y_raw > 5).astype('int32')
    
    # 4. Scaling
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    X_scaled = (X_raw - mean) / (std + 1e-7)
    
    return X_scaled, y, (mean, std)

def create_model(input_shape, dropout_rate):
    """Create a simple neural network model."""
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax") 
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def predict_quality(model, stats, example_data):
    """Perform a prediction using training statistics to scale the input."""
    mean, std = stats
    
    # Convert input and manually scale
    x_input = np.array([example_data], dtype='float32')
    x_scaled = (x_input - mean) / std
    
    # Prediction
    prediction_probs = model.predict(x_scaled, verbose=0)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    
    if prediction == 1:
        result = "Predicted quality: High (1)"
    else:
        result = "Predicted quality: Low (0)"
    return result

def train_model(validation_split=0.2, dropout_rate=0.2, epochs=10, batch_size=32):
    X, y, stats = load_data("winequality.csv")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    split_idx = int(0.8 * len(indices)) # 80% trening, 20% test
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    train_images, test_images = X[train_idx], X[test_idx]
    train_labels, test_labels = y[train_idx], y[test_idx]

    model = create_model(train_images.shape[1], dropout_rate=dropout_rate)
    print(f"Model wine training started...")
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nModel wine results:")
    print(f"Model Accuracy: {test_acc * 100:.2f}%")
    print(f"Model Loss: {test_loss:.4f}")
    print(f"Mean used for scaling: {stats[0]}")
    print(f"Std used for scaling: {stats[1]}")

    # Zapisz metryki i statystyki skalowania
    metrics_data = {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "mean": stats[0].tolist(),
        "std": stats[1].tolist()
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")

    save_path = os.path.join("models", "wine_model.keras")
    model.save(save_path)

    metrics_filename = save_path.replace(".keras", ".json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Metrics and stats saved as: {metrics_filename}")

    return model, stats
