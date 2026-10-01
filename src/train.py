import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

def train_model(processed_dir: str):
    """
    Loads train and test datasets, trains a Random Forest model with class weighting,
    evaluates performance, and logs parameters/metrics using MLflow.
    """
    print("=" * 50)
    print("PHASE 3: MODEL TRAINING & MLFLOW TRACKING")
    print("=" * 50)

    # 1. Load processed data
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed train/test files not found. Please run src/features.py first.")

    print("Loading processed train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    # 2. Set up MLflow Experiment
    mlflow.set_experiment("Credit_Card_Fraud_Detection")

    with mlflow.start_run():
        # Hyperparameters
        n_estimators = 100
        max_depth = 10
        random_state = 42

        print(f"Training Random Forest Classifier (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        # Initialize model with class_weight='balanced' to handle fraud imbalance
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )

        # Fit model
        model.fit(X_train, y_train)

        # 3. Predictions and Evaluation
        print("Evaluating model performance on test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        print(f"  - ROC-AUC Score: {roc_auc:.4f}")
        print(f"  - Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 4. Log to MLflow
        print("Logging parameters, metrics, and model to MLflow...")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)

        # Log trained model artifact
        mlflow.sklearn.log_model(
            model, 
            name="random_forest_model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print("\nModel training and MLflow logging completed successfully!")

if __name__ == "__main__":
    PROCESSED_DIR = "data/processed"
    train_model(processed_dir=PROCESSED_DIR)