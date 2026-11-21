
# IITM BS MLOps OPPE-1 Mock

## Project Overview

This project implements an incremental machine learning pipeline using two versions of the Iris dataset. It demonstrates DVC data versioning, model training, MLflow experiment tracking, hyperparameter tuning, model inference, automated testing and CI using GitHub Actions.

## Repository Structure and File Descriptions

| File / Directory | Description |
|---|---|
| `.dvc/` | Contains DVC configuration and metadata for tracking datasets and the remote storage. |
| `.dvc/config` | Specifies the default DVC remote, including the Google Cloud Storage bucket. |
| `.dvc/.gitignore` | Contains DVC-generated ignore rules. |
| `.dvcignore` | Specifies files and directories that DVC should ignore during its operations. |
| `.gitignore` | Specifies files Git should exclude, such as CSV datasets, Python cache files, MLflow runs and database files. |
| `iris_v0.csv.dvc` | DVC pointer file for the first version of the Iris dataset. |
| `iris_v1.csv.dvc` | DVC pointer file for the second version of the Iris dataset. |
| `train.py` | Implements data loading, species encoding, missing-value imputation, incremental training, validation, metric calculation, MLflow logging and hyperparameter tuning. |
| `inference.py` | Loads the saved best model and provides a function to predict the Iris species for new samples. |
| `requirements.txt` | Lists the Python dependencies required to execute the project. |
| `artifacts/` | Stores the selected best model and its metadata. |
| `artifacts/best_model.pkl` | Serialized best Decision Tree model selected through hyperparameter tuning. |
| `artifacts/best_model_info.json` | Stores the best hyperparameter, validation F1-score and corresponding MLflow run ID. |
| `outputs/` | Stores model evaluation outputs. |
| `outputs/metrics.json` | Contains accuracy, macro precision, macro recall and macro F1-score for both training iterations. |
| `mlruns/` | Local MLflow experiment artifact directory. This directory is excluded from Git. |
| `tests/` | Contains automated tests for model inference. |
| `tests/test_inference.py` | Tests valid predictions, known sample predictions and list-based input. |
| `.github/workflows/` | Contains GitHub Actions workflow definitions. |
| `.github/workflows/mlops-ci.yml` | Defines the CI pipeline that authenticates with GCP, retrieves datasets using DVC, executes training and runs pytest. |
| `README.md` | Documents the project, repository structure, files and execution instructions. |
| `AI_USAGE_DOC.md` | Documents the use of AI assistance during the assignment. |

## Dataset Versioning

The two Iris dataset versions are tracked using DVC:

- `iris_v0.csv`
- `iris_v1.csv`

The actual CSV files are stored in Google Cloud Storage through DVC. Their `.dvc` pointer files are tracked in Git.

Retrieve datasets using:

```bash
dvc pull
```

## Training and Evaluation

The training pipeline performs two iterations:

1. Iteration 1: Training using `iris_v0`.
2. Iteration 2: Training using the combined `iris_v0` and `iris_v1` datasets.

A stratified 80:20 train-validation split is used with random state 42.

Missing feature values are imputed using the mean of the last 10 available training observations of the same species and feature.

Species encoding:

| Species | Encoding |
|---|---:|
| setosa | 0 |
| versicolor | 1 |
| virginica | 2 |

A Decision Tree Classifier is used. Accuracy, macro precision, macro recall and macro F1-score are calculated.

## MLflow Experiment Tracking

The MLflow experiment is named `Iris_Classification`.

The pipeline logs parameters, metrics, classification reports and model artifacts.

Hyperparameter tuning evaluates maximum tree depths of 2, 3, 4 and 5. Each trial is recorded as a separate MLflow run.

The pipeline logs the following artifacts for each hyperparameter tuning run:
- Classification report (`classification_report.txt`)
- Confusion matrix (`confusion_matrix.json`)
- Trained model (`model/`)

## Model Inference

Run:

```bash
python inference.py
```

## Automated Testing

Run:

```bash
python -m pytest -v
```

## Continuous Integration

GitHub Actions executes the following steps on pushes and pull requests to `main`:

1. Checkout repository.
2. Set up Python.
3. Authenticate with Google Cloud using Workload Identity Federation.
4. Install dependencies.
5. Pull datasets using DVC.
6. Run training and evaluation.
7. Run pytest.

## Reproducing the Project

```bash
pip install -r requirements.txt
dvc pull
python train.py
python -m pytest -v
```
