import os
import pandas as pd

def ingest_data(raw_data_path: str, processed_dir: str):
    """
    Loads raw credit card fraud data, performs basic exploratory checks,
    and ensures the processed data directory exists.
    """
    print("=" * 50)
    print("PHASE 1: DATA INGESTION & SANITY CHECKS")
    print("=" * 50)

    # 1. Check if raw file exists
    if os.path.isdir(raw_data_path):
        raw_data_path = os.path.join(raw_data_path, "creditcard.csv")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(
            f"Dataset not found at {raw_data_path}. Please download it from Kaggle and place it in data/raw/."
        )

    # 2. Load the dataset
    print(f"Loading dataset from: {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Dataset successfully loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")

    # 3. Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values in dataset: {missing_values}")

    # 4. Check class distribution (0 = Legitimate, 1 = Fraud)
    print("\nChecking Class Distribution (Target: 'Class'):")
    class_counts = df['Class'].value_counts()
    fraud_count = class_counts.get(1, 0)
    legit_count = class_counts.get(0, 0)
    fraud_percentage = (fraud_count / len(df)) * 100

    print(f"  - Legitimate Transactions (0): {legit_count:,}")
    print(f"  - Fraudulent Transactions (1): {fraud_count:,}")
    print(f"  - Fraud Percentage: {fraud_percentage:.4f}%")

    # 5. Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    print(f"\nDirectory verified/created: {processed_dir}")
    print("Data ingestion and sanity checks completed successfully!\n")
    
    return df

if __name__ == "__main__":
    # Define paths relative to root
    RAW_PATH = "data/raw/creditcard.csv"
    PROCESSED_DIR = "data/processed"
    
    ingest_data(raw_data_path=RAW_PATH, processed_dir=PROCESSED_DIR)