"""Main entry point for the project.

Allows selecting a model and performing predictions:
- model 0: wine quality prediction (--wine_params)
- model 1/2: animal recognition (large/small)
- model 3: clothing recognition
- model 4: digit recognition

Usage example:
    python main.py --model 1 --path ./bird.jpg
"""

import tensorflow as tf
import argparse
from prepare_models import prepare_models
import sys
from prepare_model_wine import predict_quality
from prepare_model_animals import predict_animal
from prepare_model_fashion import predict_cloth
from prepare_model_digits import predict_digit


def main():
    """Parse arguments, load models, and perform predictions based on input.

    Supports wine quality prediction (parameters via --wine_params) and image prediction
    provided via --path using the selected model.
    """
    parser = argparse.ArgumentParser(description="Model selection.")
    parser.add_argument(
        "--wine_params",
        type=str,
        default=None,
        help="Parameters for wine quality prediction model in format: 'fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol'.",
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the image file for prediction.",
    )

    parser.add_argument(
        "--model",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=0,
        help="Choose model for prediction. Options: 0 - Wine, 1 - Animals (large), 2 - Animals (small), 3 - Fashion, 4 - Digits. Default: 0.",
    )

    args = parser.parse_args()
            
    wine_model, wine_stats, large_model_animals, small_model_animals, fashion_model, digits_model = prepare_models()

    if args.wine_params and args.model == 0:
        try:
            params = [float(x) for x in args.wine_params.split(",")]
            if len(params) != 11:
                print("Error: Please provide exactly 11 parameters for wine quality prediction.")
                sys.exit(1)
            result = predict_quality(wine_model, wine_stats, params)
        except ValueError:
            print("Error: Please ensure all wine parameters are numeric values.")
            sys.exit(1)
            
    if args.path:
        if args.model == 1:
            result = predict_animal(large_model_animals, args.path)
        elif args.model == 2:
            result = predict_animal(small_model_animals, args.path)
        elif args.model == 3:
            result = predict_cloth(fashion_model, args.path)
        elif args.model == 4:
            result = predict_digit(digits_model, args.path)

    if (result is not None):
        print("Prediction result:")
        print(result)

if __name__ == "__main__":
    main()
