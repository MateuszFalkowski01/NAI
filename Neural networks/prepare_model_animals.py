"""Module for preparing and training animal recognition models using CIFAR-10.

Contains functions for loading data, creating models (small/large), prediction and comparisons.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


def load_data():
    """Load and filter the CIFAR-10 dataset, keeping only animal classes.

    Returns (train_images, train_labels), (test_images, test_labels) with values normalized to 0-1.
    """
    dataset = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # 1. Define the indices of classes that are animals
    # bird: 2, cat: 3, deer: 4, dog: 5, frog: 6, horse: 7
    animal_indices = [2, 3, 4, 5, 6, 7]
    
    # 2. Create boolean masks for train and test sets
    train_mask = np.isin(train_labels, animal_indices).flatten()
    test_mask = np.isin(test_labels, animal_indices).flatten()

    # 3. Filter images and labels
    train_images = train_images[train_mask]
    train_labels = train_labels[train_mask]
    
    test_images = test_images[test_mask]
    test_labels = test_labels[test_mask]

    # 4. REMAPPING: change [2, 3, 4...] to [0, 1, 2...]
    # The model expects classes starting at 0 and contiguous
    for i, original_idx in enumerate(animal_indices):
        train_labels[train_labels == original_idx] = i
        test_labels[test_labels == original_idx] = i

    # Normalization
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

data_augmentation = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ]
)

def create_model(model_type="large", dropout_rate=0.2, learning_rate=0.001, augmentation=True):
    """Create a model for animal recognition.

    The `model_type` parameter selects the architecture: 'small' or 'large'.
    """
    if model_type == "small":
        # SMALL NETWORK: Dense-only architecture
        model = keras.Sequential([
            layers.Input(shape=(32, 32, 3)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
    else:
        # LARGE NETWORK: Convolutional Neural Network (CNN)
        model = keras.Sequential([
            layers.Input(shape=(32, 32, 3)),
            data_augmentation if augmentation else layers.Lambda(lambda x: x),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(6, activation="softmax")
        ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def predict_animal(model, image_data):
    """Predict the animal class for the given image or path.

    Returns the class name (e.g., 'bird', 'cat', ...).
    """
    # Check whether image_data is a path (string) or an array
    if isinstance(image_data, str):
        img = keras.utils.load_img(image_data, target_size=(32, 32))
        img = keras.utils.img_to_array(img)
    else:
        img = np.array(image_data, dtype='float32')

    # Normalize (if pixel values are 0-255, convert to 0-1)
    if img.max() > 1.0:
        img = img / 255.0
    
    # Add batch dimension ((32,32,3) -> (1,32,32,3))
    img = np.expand_dims(img, axis=0)
    
    prediction_probs = model.predict(img, verbose=0)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    
    classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']    
    result = classes[prediction]
    
    return result


def train_model(model_type="large", filename="model_animals.keras", validation_split=0.2, epochs=10, dropout_rate=0.2, batch_size=32, learning_rate=0.001, augmentation=True):
    """Train the animal model and save the model and metrics to disk."""
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    model = create_model(model_type=model_type, dropout_rate=dropout_rate, learning_rate=learning_rate, augmentation=augmentation)
    print(f"Model animals {model_type} training started...")
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nModel animals {model_type} results:")
    print(f"Model Accuracy: {test_acc * 100:.2f}%")
    print(f"Model Loss: {test_loss:.4f}")
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    full_filename = f"{model_type}_{filename}"
    save_path = os.path.join("models", full_filename)
    model.save(save_path)

    metrics_data = {"loss": float(test_loss), "accuracy": float(test_acc)}
    with open(save_path.replace(".keras", ".json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    return model, test_acc

def generate_comparison(small_acc, large_acc):
    """Plot a comparison chart of Small vs Large model accuracies."""
    models = ['Small (Dense)', 'Large (CNN)']
    accuracies = [small_acc * 100, large_acc * 100]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['red', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of network sizes for CIFAR-10')
    plt.ylim(0, 100)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.show()
