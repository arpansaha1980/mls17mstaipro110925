import os
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


os.makedirs("./outputs", exist_ok=True)  # Create the "outputs" directory if it doesn't exist


def select_first_file(path: str) -> str:
    """Selects the first file in a folder, assuming there's only one file."""
    files = os.listdir(path)
    return os.path.join(path, files[0])


def main():

    mlflow.start_run()  # Start MLflow session

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
        default=5,
        help="Maximum depth of the trees (default: 5)",
    )
    parser.add_argument(
        "--model_output",
        type=str,     # ✅ FIXED
        help="Path of output model",
    )
    args = parser.parse_args()

    # Log hyperparameters to MLflow
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    target_col = "price"

    y_train = train_df[target_col].values
    X_train = train_df.drop(target_col, axis=1).values

    y_test = test_df[target_col].values
    X_test = test_df.drop(target_col, axis=1).values

    # Train RF model
    rf_model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")

    mlflow.log_metric("MSE", float(mse))

    # Save artifact
    mse_path = os.path.join("outputs", "mse.txt")
    with open(mse_path, "w") as f:
        f.write(str(mse))
    mlflow.log_artifact(mse_path)

    # Save MLflow model
    mlflow.sklearn.save_model(rf_model, args.model_output)

    print("Training complete.")
    mlflow.end_run()


if __name__ == "__main__":
    main()
