import os

import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

numeric_col = ["tenure"]
binary_cols = ["is_dualsim", "is_featurephone", "is_smartphone"]
multi_cat_cols = ["trf", "gndr", "dev_man", "device_os_name", "region"]

# Define preprocessing

numeric_object_pipeline = Pipeline(
    [
        (
            "to_numeric",
            FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors="coerce")),
        ),
        ("impute", SimpleImputer(strategy="median")),
    ]
)

binary_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder()),
    ]
)

multi_cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("binary_cat", binary_pipeline, binary_cols),
        ("multi_cat", multi_cat_pipeline, multi_cat_cols),
        ("num", numeric_object_pipeline, numeric_col),
    ]
)

# Load dataset


def load_data():
    df = pd.read_parquet("data/processed/my_dataset.parquet")
    return df


def split_features_target(df, target_col="target", test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Hyperparameter search

param_distributions = {
    "xgb__n_estimators": randint(50, 300),
    "xgb__max_depth": randint(3, 10),
    "xgb__learning_rate": uniform(0.01, 0.3),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
}

# Train pipeline


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_features_target(df)

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("xgb", XGBClassifier(eval_metric="logloss"))]
    )

    rand_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    rand_search.fit(X_train, y_train)

    print("Best parameters found:", rand_search.best_params_)
    print("Best CV accuracy:", rand_search.best_score_)
    y_pred = rand_search.predict(X_test)
    print("Test accuracy:", y_train, y_pred)  # optional
    print("Classification report:")
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(rand_search.best_estimator_, "models/pipeline_model.pkl")
    print("Pipeline saved to models/pipeline_model.pkl")


if __name__ == "__main__":
    main()
