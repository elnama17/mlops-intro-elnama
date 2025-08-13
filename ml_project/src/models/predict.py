import os

import joblib
import pandas as pd


def load_pipeline(path="models/pipeline_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline not found at {path}. Run train.py first.")
    return joblib.load(path)


def load_new_data(file_path="data/processed/new_data.csv"):
    df = pd.read_csv(file_path)
    return df


def main():
    pipeline = load_pipeline()
    X_new = load_new_data()
    y_pred = pipeline.predict(X_new)
    print("Predictions for new data:")
    print(y_pred)


if __name__ == "__main__":
    main()
