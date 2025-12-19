"""Helper module for loading or training the set of models used in the project.

The function `prepare_models` returns the tuple:
(wine_model, wine_stats, large_model_animals, small_model_animals, fashion_model, digits_model)
"""

import tensorflow as tf
import numpy as np
import yaml
import sys
import os
import json
from prepare_model_wine import train_model as train_model_wine
from prepare_model_animals import train_model as train_model_animals, generate_comparison
from prepare_model_fashion import train_model as train_model_fashion, generate_confusion_matrix
from prepare_model_digits import train_model as train_model_digits

CONFIG_PATH = "config.yaml"


def load_config():
    """Load configuration from YAML file `CONFIG_PATH` and return it as a dict.

    The program will exit with a message if the file is missing.
    """
    try:
        with open(CONFIG_PATH, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return config
    except FileNotFoundError:
        print(f"Error: missing config file at {CONFIG_PATH}")
        sys.exit(1)      
        
        
def prepare_models():
    """Load existing models from the `models` directory or train them if files are missing.

    Returns a tuple: (wine_model, wine_stats, large_model_animals, small_model_animals, fashion_model, digits_model).
    """
    config = load_config()
    EPOCHS = config.get("epochs")
    BATCH_SIZE = config.get("batch_size")
    LEARNING_RATE = config.get("learning_rate")
    VALIDATION_SPLIT = config.get("validation_split")
    DATA_AUGMENTATION = config.get("data_augmentation")
    DROPOUT_RATE = config.get("dropout_rate")

    if not os.path.exists("models"):
        os.makedirs("models")

    # śceżki do zapisanych modeli i metryk
    model1_path = os.path.join("models", "wine_model.keras")
    metrics1_path = model1_path.replace(".keras", ".json")
    model2_1_path = os.path.join("models", "large_model_animals.keras")
    metrics2_1_path = model2_1_path.replace(".keras", ".json")
    model2_2_path = os.path.join("models", "small_model_animals.keras")
    metrics2_2_path = model2_2_path.replace(".keras", ".json")
    model3_path = os.path.join("models", "fashion_model.keras")
    metrics3_path = model3_path.replace(".keras", ".json")
    model4_path = os.path.join("models", "digits_model.keras")
    metrics4_path = model4_path.replace(".keras", ".json")


    if os.path.exists(model1_path) and os.path.exists(metrics1_path):
        print("Loading existing wine model...")
        model1 = tf.keras.models.load_model(model1_path)
        with open(metrics1_path, "r") as f:
            metrics_data = json.load(f)
            mean = np.array(metrics_data["mean"])
            std = np.array(metrics_data["std"])
            stats1 = (mean, std)

        print("Wine model loaded.")
        print(f"Model Accuracy: {metrics_data['accuracy'] * 100:.2f}%")
        print(f"Model Loss: {metrics_data['loss']:.4f}")
        print(f"Mean used for scaling: {stats1[0]}")
        print(f"Std used for scaling: {stats1[1]}")
    else:
        print("Preparing new wine model...")
        model1, stats1 = train_model_wine(validation_split=VALIDATION_SPLIT, dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    if os.path.exists(model2_1_path) and os.path.exists(metrics2_1_path) and os.path.exists(model2_2_path) and os.path.exists(metrics2_2_path):        
        print("Loading existing animals models...")
        model2_1 = tf.keras.models.load_model(model2_1_path)
        with open(metrics2_1_path, "r") as f:
            metrics_data = json.load(f)
            test_acc2_1 = metrics_data["accuracy"]

        model2_2 = tf.keras.models.load_model(model2_2_path)
        with open(metrics2_2_path, "r") as f:
            metrics_data = json.load(f)
            test_acc2_2 = metrics_data["accuracy"]

        print("Animals models loaded.")
        print(f"Large Model Accuracy: {test_acc2_1 * 100:.2f}%")
        print(f"Large Model loss: {metrics_data['loss']:.4f}")
        print(f"Small Model Accuracy: {test_acc2_2 * 100:.2f}%")
        print(f"Small Model loss: {metrics_data['loss']:.4f}")
    else:
        print("Preparing new animals models...")
        model2_1, test_acc2_1 = train_model_animals(model_type="large", validation_split=VALIDATION_SPLIT, dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, augmentation=DATA_AUGMENTATION)
        model2_2, test_acc2_2 = train_model_animals(model_type="small", validation_split=VALIDATION_SPLIT, dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    generate_comparison(test_acc2_1, test_acc2_2)
    
    if os.path.exists(model3_path) and os.path.exists(metrics3_path):
        print("Loading existing fashion model...")
        model3 = tf.keras.models.load_model(model3_path)
        with open(metrics3_path, "r") as f:
            metrics_data = json.load(f)

        print("Fashion model loaded.")
        print(f"Model Accuracy: {metrics_data['accuracy'] * 100:.2f}%")
        print(f"Model Loss: {metrics_data['loss']:.4f}")
        generate_confusion_matrix(model3)
    else:
        print("Preparing new fashion model...")
        model3 = train_model_fashion(validation_split=VALIDATION_SPLIT, dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, augmentation=DATA_AUGMENTATION)
        
    if os.path.exists(model4_path) and os.path.exists(metrics4_path):
        print("Loading existing digits model...")
        model4 = tf.keras.models.load_model(model4_path)
        with open(metrics4_path, "r") as f:
            metrics_data = json.load(f)

        print("Digits model loaded.")
        print(f"Model Accuracy: {metrics_data['accuracy'] * 100:.2f}%")
        print(f"Model Loss: {metrics_data['loss']:.4f}")
    else:
        print("Preparing new digits model...")
        model4 = train_model_digits(validation_split=VALIDATION_SPLIT, dropout_rate=DROPOUT_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    return model1, stats1, model2_1, model2_2, model3, model4
