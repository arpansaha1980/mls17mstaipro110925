import argparse
import json
import logging
import os
from pathlib import Path

import mlflow


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name under which model will be registered",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory (MLflow model)",
    )
    parser.add_argument(
        "--model_info_output_path",
        type=str,
        required=False,
        help="Path to write model info JSON",
    )
    args, _ = parser.parse_known_args()
    print(f"Arguments: {args}")
    return args


def main(args):
    logging.basicConfig(level=logging.INFO)

    model_name = args.model_name
    model_path = Path(args.model_path)

    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Model will be registered as: {model_name}")

    # Start MLflow run for registration
    with mlflow.start_run():
        # Load the trained model
        model = mlflow.sklearn.load_model(model_path)

        logging.info("Registering the best trained used car price prediction model")

        # Register the model in the MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=model_name,
            artifact_path="Random_forest_used_car_price_prediction",
        )

        run = mlflow.active_run()
        run_id = run.info.run_id if run else None

        # Optionally write model info JSON into the output path
        if args.model_info_output_path:
            out_dir = Path(args.model_info_output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            info = {
                "model_name": model_name,
                "model_path": str(model_path),
                "mlflow_run_id": run_id,
            }
            out_file = out_dir / "model_info.json"
            with open(out_file, "w") as f:
                json.dump(info, f, indent=2)
            logging.info(f"Model info written to: {out_file}")


if __name__ == "__main__":
    # Parse arguments and run main
    args = parse_args()
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}",
    ]
    for line in lines:
        print(line)

    main(args)
