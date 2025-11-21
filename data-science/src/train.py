import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.start_run()  # Start the MLflow experiment run

os.makedirs("./outputs", exist_ok=True)  # Create the "outputs" directory if it doesn't exist


def select_first_file(path: str) -> str:
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees in the random forest (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of the trees (default: None)",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        help="Path of output model",
    )
    args = parser.parse_args()

    # Log hyperparameters to MLflow
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Load datasets (select the first file in each provided folder)
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # ---- Adjust this if your target column has a different name ----
    target_col = "price"

    # Split into features (X) and target (y)
    y_train = train_df[target_col].values
    X_train = train_df.drop(target_col, axis=1).values

    y_test = test_df[target_col].values
    X_test = test_df.drop(target_col, axis=1).values

    # Initialize Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = rf_model.predict(X_test)

    # Evaluate using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE) of Random Forest Regressor on test set: {mse:.4f}")

    # Log MSE to MLflow
    mlflow.log_metric("MSE", float(mse))

    # Optionally save MSE as a small artifact
    mse_path = os.path.join("outputs", "mse.txt")
    with open(mse_path, "w") as f:
        f.write(str(mse))
    mlflow.log_artifact(mse_path)

    # Save the trained model as an MLflow model
    mlflow.sklearn.save_model(rf_model, args.model_output)

    print("Training complete. Final MSE logged to MLflow.")
    mlflow.end_run()  # End the MLflow experiment run


if __name__ == "__main__":
    main()
