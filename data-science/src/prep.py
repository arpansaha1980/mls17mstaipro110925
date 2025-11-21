import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()  # Create an ArgumentParser object
    parser.add_argument("--data", type=str, help="Path to raw data")  # Specify the type for raw data (str)
    parser.add_argument("--train_data", type=str, help="Path to train dataset")  # Specify the type for train data (str)
    parser.add_argument("--test_data", type=str, help="Path to test dataset")  # Specify the type for test data (str)
   parser.add_argument("--test_train_ratio", type=float, default=0.2)  # Specify the type (float) and default value (0.2) for test-train ratio
    args = parser.parse_args()

    return args
    
def main(args):
   
    args = parser.parse_args()

    # Start MLflow Run
    # mlflow.start_run()

    # Log arguments
    logging.info(f"Input data path: {args.data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")

    # Read data
    df = pd.read_csv(args.data)

    # Encoding the categorical 'Type' column
    label_encoder = LabelEncoder()
    df['Segment'] = label_encoder.fit_transform(df['Segment'])

    # Log the first few rows of the dataframe
    logging.info(f"Transformed Data:\n{df.head()}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    train_df.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Log completion
    mlflow.log_metric("train_size", len(train_df))
    mlflow.log_metric("test_size", len(test_df))
    # mlflow.end_run()

 if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args() # Call the function to parse arguments
    main(args)

    mlflow.end_run()
   

