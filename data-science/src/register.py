import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=_____, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=_____, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=_____, help="Path to write model info JSON")  # Hint: Specify the type for model_info_output_path (str)
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args
    
def main(args):
    # Argument parser setup for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=_____, help='Name under which model will be registered')  # Hint: Specify the type for model_name (str)
    parser.add_argument('--model_path', type=_____, help='Model directory')  # Hint: Specify the type for model_path (str)
    parser.add_argument("--model_info_output_path", type=_____, help="Path to write model info JSON")  #
    args = parser.parse_args()

    # Load the trained model from the provided path
    model = mlflow.sklearn.load_model(Path(args.model))

    print("Registering the best trained used car price prediction model")

    # Register the model in the MLflow Model Registry under the name "used_car_price_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_car_price_prediction_model",  # Descriptive model name for registration
        artifact_path="Random_forest_used_car_price_prediction"  # Path to store model artifacts
    )

  

if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
