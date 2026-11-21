import joblib
import pandas as pd

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def predict_species(sample, model_path="artifacts/best_model.pkl"):
    model = joblib.load(model_path)

    if isinstance(sample, dict):
        sample = pd.DataFrame([sample])
    else:
        sample = pd.DataFrame(sample, columns=FEATURES)

    prediction = model.predict(sample[FEATURES])[0]
    return TARGET_MAP[int(prediction)]


if __name__ == "__main__":
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    print("Predicted species:", predict_species(sample))
