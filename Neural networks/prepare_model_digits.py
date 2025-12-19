"""Module for preparing and training a digit recognition model (Digits dataset from UCI)."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

def load_data():
    """Fetch the Digits dataset from the UCI repository, normalize and split into train/test sets."""
    digits = fetch_ucirepo(id=80) 
    X = digits.data.features.values.astype('float32')
    y = digits.data.targets.values.flatten().astype('int32')
    
    # Normalization: Digits id=80 has pixel values 0-16
    X = X / 16.0
    
    # 80/20 split
    indices = np.arange(X.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    
    # Reshape to 8x8x1 (these are 8x8 images)
    X_images = X.reshape(-1, 8, 8, 1)
    
    X_train, X_test = X_images[indices[:split]], X_images[indices[split:]]
    y_train, y_test = y[indices[:split]], y[indices[split:]]
    
    return (X_train, y_train), (X_test, y_test)

def create_model(dropout_rate, learning_rate=0.001):
    """Create and compile a CNN model tailored for 8x8 digit images."""
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def predict_digit(model, image_path):
    """Load an image, convert it to shape (1, 8, 8, 1) and return the predicted digit (0-9)."""
    # 1. Read and decode
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_image(img_raw, channels=0)
    
    # 2. Convert to grayscale if needed
    if img.shape[-1] > 1:
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = tf.image.rgb_to_grayscale(img)
    
    # 3. Resize to 8x8
    img = tf.image.resize(img, [8, 8])
    
    # 4. Normalize and cast to float32
    img = tf.cast(img, tf.float32) / 255.0
    
    # 5. Invert (dark background for digits)
    if tf.reduce_mean(img) > 0.5:
        img = 1.0 - img
        
    # 6. Fix dimensions - the model expects (1, 8, 8, 1)
    # Remove any extra dim of size 1 and add a single batch dim at the start
    img = tf.reshape(img, (8, 8, 1))  # Ensure (8,8,1)
    img = tf.expand_dims(img, axis=0)  # Add batch -> (1, 8, 8, 1)
    
    prediction_probs = model.predict(img, verbose=0)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    
    result = prediction
    return result

def train_model(epochs=50, dropout_rate=0.2, learning_rate=0.001, batch_size=16, validation_split=0.1):
    """Train the model on the Digits dataset, save the model and metrics to disk."""
    (X_train, y_train), (X_test, y_test) = load_data()
    
    model = create_model(dropout_rate, learning_rate)
    
    print("Model Numbers training started (Predicting Digits)...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nModel Digits Results:")
    print(f"Model Accuracy: {test_acc * 100:.2f}%")
    print(f"Model Loss: {test_loss:.4f}")
    
    if not os.path.exists("models"):
        os.makedirs("models")

    save_path = os.path.join("models", "digits_model.keras")
    model.save(save_path)
    
    metrics_data = {
        "loss": float(test_loss),
        "accuracy": float(test_acc)
    }
    
    metrics_filename = save_path.replace(".keras", ".json")

    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=4)
    
    if not os.path.exists("models"): os.makedirs("models")
    model.save("models/digits_model.keras")
    
    return model
