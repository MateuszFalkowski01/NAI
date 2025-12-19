"""Module for training a model on Fashion-MNIST and generating a confusion matrix."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def load_data():
    """Load the Fashion-MNIST dataset and normalize images to the 0-1 range.

    Returns (train_images, train_labels), (test_images, test_labels).
    """
    dataset = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Add channel dimension ((60000, 28, 28) -> (60000, 28, 28, 1))
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    return (train_images, train_labels), (test_images, test_labels)

data_augmentation = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

def create_model(dropout_rate, learning_rate=0.001, augmentation=True):
    """Create a CNN model for Fashion-MNIST.

    Returns a compiled model instance.
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation if augmentation else layers.Lambda(lambda x: x),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def predict_cloth(model, image_data):
    """Predict the clothing class for an image (path or array)."""
    if isinstance(image_data, str):
        img = keras.utils.load_img(image_data, target_size=(28, 28), color_mode="grayscale")
        img = keras.utils.img_to_array(img)
    else:
        img = np.array(image_data, dtype='float32')
    
    # 2. Normalization
    if img.max() > 1.0:
        img = img / 255.0
    
    # 3. Handle dimensions for grayscale images
    # Ensure shape is (1, 28, 28, 1)
    if len(img.shape) == 2:  # If (28, 28)
        img = np.expand_dims(img, axis=-1)
    
    if img.shape[0] != 1:    # Add batch dimension if missing
        img = np.expand_dims(img, axis=0)
    
    prediction_probs = model.predict(img, verbose=0)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    result = classes[prediction]
    return result

def train_model(validation_split=0.2, dropout_rate=0.2, epochs=10, batch_size=32, learning_rate=0.001, augmentation=True):
    """Train and save the Fashion-MNIST model, then generate a confusion matrix."""
    (train_images, train_labels), (test_images, test_labels) = load_data()
    
    model = create_model(dropout_rate=dropout_rate, learning_rate=learning_rate, augmentation=augmentation)
    print(f"Model fashion training started...")
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nModel fashion results:")
    print(f"Model Accuracy: {test_acc * 100:.2f}%")
    print(f"Model Loss: {test_loss:.4f}")

    metrics_data = {
        "loss": float(test_loss),
        "accuracy": float(test_acc)
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")

    save_path = os.path.join("models", "fashion_model.keras")
    model.save(save_path)

    metrics_filename = save_path.replace(".keras", ".json")
    with open(metrics_filename, "w") as f:
        json.dump(metrics_data, f, indent=4)

    generate_confusion_matrix(model)

    return model

def generate_confusion_matrix(model):
    """Calculate and display the confusion matrix on the Fashion-MNIST test set."""
    (train_images, train_labels), (test_images, test_labels) = load_data()
    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    
    cm = tf.math.confusion_matrix(test_labels, y_pred).numpy()
    
    print("\nMacierz Pomyłek (Confusion Matrix):")
    print(cm)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')    
    plt.title("Confusion Matrix - Fashion MNIST")
    plt.colorbar()
    plt.ylabel('Real class')
    plt.xlabel('Predicted class')
    plt.show()
