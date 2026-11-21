
# OPPE AI Usage Documentation

## AI Tools Utilized and Conversation History

### Tool Name: ChatGPT (GPT-6)

- **Purpose:**
  - Understanding and implementing DVC dataset versioning.
  - Developing the incremental machine learning training pipeline.
  - Implementing species encoding and missing-value imputation.
  - Configuring MLflow experiment tracking and hyperparameter tuning.
  - Debugging Python and MLflow errors.
  - Developing the model inference script and pytest tests.
  - Configuring Google Cloud Workload Identity Federation.
  - Implementing GitHub Actions CI with DVC and GCP.
  - Preparing project documentation.

- **Shared Chat Link:**
  https://chatgpt.com/share/6b0190cc-9fe8-83e8-bf1c-65bd702e20d6

- **Notes:**
  ChatGPT was used as an AI-assisted learning and development tool. Suggestions and generated code were reviewed, executed and tested in the practice GCP environment. Errors encountered during implementation were investigated and corrected.

---

## Prompts and Responses Used

### Tool Name #1: ChatGPT (GPT-6)

The following prompts represent the main areas where AI assistance contributed to the implementation. The complete prompt-response history is available through the shared conversation link above.

#### Prompt 1: DVC Dataset Versioning
- **Prompt:** How do I configure DVC to track two versions of the Iris dataset and push the data to Google Cloud Storage?
- **Response Log:** Guidance on DVC initialization, remote configuration, dataset tracking, pushing data and verifying DVC status.

#### Prompt 2: Incremental Model Training
- **Prompt:** How do I implement incremental training using iris_v0 and iris_v1, including validation metrics?
- **Response Log:** Guidance on combining datasets, performing stratified train-validation splits, training a Decision Tree classifier and calculating accuracy, precision, recall and F1-score.

#### Prompt 3: Missing Value Imputation
- **Prompt:** How can missing feature values be imputed using the mean of the last 10 available samples of the same species?
- **Response Log:** Guidance on calculating species-specific feature means from training data and applying the imputation values to training and validation datasets.

#### Prompt 4: MLflow Integration
- **Prompt:** How do I log model parameters, evaluation metrics, classification reports and model artifacts using MLflow?
- **Response Log:** - Additional guidance on logging classification reports, confusion matrices and trained models as MLflow artifacts for each hyperparameter tuning run.

#### Prompt 5: Hyperparameter Tuning
- **Prompt:** How do I perform hyperparameter tuning with separate MLflow runs for each trial and identify the best model?
- **Response Log:** Implemented Decision Tree tuning using different max_depth values and compared validation F1-scores.

#### Prompt 6: Model Inference and Testing
- **Prompt:** How do I create an inference script and write pytest tests to verify model predictions?
- **Response Log:** Guidance on loading the saved model, predicting Iris species and testing inference functionality.

#### Prompt 7: GitHub Actions CI
- **Prompt:** How do I configure GitHub Actions to authenticate with GCP, pull DVC datasets and run training and pytest automatically?
- **Response Log:** Guidance on creating the GitHub Actions workflow and configuring Google Cloud Workload Identity Federation.

#### Prompt 8: Documentation
- **Prompt:** How should I document the repository files and explain the implementation in README.md?
- **Response Log:** Guidance on preparing file descriptions, project structure and execution instructions.

#### Prompt 9: Testing and Verification
- **Prompt:** How do I verify inference functionality, execute pytest tests and validate the GitHub Actions CI workflow?
- **Response Log:** Guidance on running inference, resolving pytest module-import issues, verifying all three inference tests and checking successful DVC retrieval and pytest execution in GitHub Actions.
