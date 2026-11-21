import joblib
import os
import mlflow
import mlflow.sklearn
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_MAP = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


def fit_imputation_values(train_df):
    imputation_values = {}

    for species in train_df["species"].unique():
        species_data = train_df[train_df["species"] == species]

        for feature in FEATURES:
            last_10 = species_data[feature].dropna().tail(10)

            if not last_10.empty:
                imputation_values[(species, feature)] = last_10.mean()
            else:
                imputation_values[(species, feature)] = train_df[feature].mean()

    return imputation_values


def apply_imputation(df, imputation_values):
    df = df.copy()

    for (species, feature), value in imputation_values.items():
        mask = (df["species"] == species) & (df[feature].isna())
        df.loc[mask, feature] = value

    return df


def run_iteration(name, data):

    data = data.copy()

    X = data[FEATURES]
    y = data["species"].map(TARGET_MAP)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = data.loc[X_train.index].copy()
    val_df = data.loc[X_val.index].copy()

    imputation_values = fit_imputation_values(train_df)
    train_df = apply_imputation(train_df, imputation_values)
    val_df = apply_imputation(val_df, imputation_values)

    max_depth = 3
    random_state = 42

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )

    model.fit(train_df[FEATURES], y_train)
    predictions = model.predict(val_df[FEATURES])

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision_macro": precision_score(
            y_val, predictions, average="macro"
        ),
        "recall_macro": recall_score(
            y_val, predictions, average="macro"
        ),
        "f1_macro": f1_score(
            y_val, predictions, average="macro"
        ),
    }

    report = classification_report(y_val, predictions)

    with mlflow.start_run(run_name=name):

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset_size", len(data))
        mlflow.log_param("features", FEATURES)

        mlflow.log_metrics(metrics)

        mlflow.log_text(
            report,
            f"{name.replace(' ', '_')}_classification_report.txt"
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

    print(f"\n===== {name} =====")
    print(json.dumps(metrics, indent=2))
    print(report)

    return metrics



def tune_hyperparameters(data):

    data = data.copy()
    X = data[FEATURES]
    y = data["species"].map(TARGET_MAP)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_df = data.loc[X_train.index].copy()
    val_df = data.loc[X_val.index].copy()

    imputation_values = fit_imputation_values(train_df)
    train_df = apply_imputation(train_df, imputation_values)
    val_df = apply_imputation(val_df, imputation_values)

    best_f1 = -1
    best_depth = None
    best_model = None
    best_run_id = None

    for depth in [2, 3, 4, 5]:

        with mlflow.start_run(run_name=f"Tuning_max_depth_{depth}"):

            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=42,
            )

            model.fit(train_df[FEATURES], y_train)
            predictions = model.predict(val_df[FEATURES])

            accuracy = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average="macro")

            mlflow.log_param("max_depth", depth)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_macro", f1)

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            run_id = mlflow.active_run().info.run_id

            print(
                f"max_depth={depth}, "
                f"accuracy={accuracy:.4f}, f1={f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_depth = depth
                best_model = model
                best_run_id = run_id

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(best_model, "artifacts/best_model.pkl")

    best_info = {
        "best_max_depth": best_depth,
        "best_f1_macro": best_f1,
        "best_run_id": best_run_id,
    }

    with open("artifacts/best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=2)

    print("\nBest model saved!")
    print(json.dumps(best_info, indent=2))

    return best_info



def main():

    mlflow.set_experiment("Iris_Classification")

    v0 = pd.read_csv("iris_v0.csv")
    v1 = pd.read_csv("iris_v1.csv")

    # Create combined dataset once
    combined = pd.concat([v0, v1], ignore_index=True)

    results = {
        "iteration_1": run_iteration("Iteration 1: iris_v0", v0),
        "iteration_2": run_iteration(
            "Iteration 2: iris_v0 + iris_v1",
            combined,
        ),
    }

    tune_hyperparameters(combined)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nMetrics saved to outputs/metrics.json")


if __name__ == "__main__":
    main()
