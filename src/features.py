import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def process_features(raw_data_path: str, processed_dir: str):
    """
    Loads raw data, scales Time and Amount features, performs a stratified 
    train-test split, and saves the resulting datasets to data/processed/.
    """
    print("=" * 50)
    print("PHASE 2: FEATURE ENGINEERING & PREPROCESSING")
    print("=" * 50)

    # 1. Load raw data
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}.")
    
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # 2. Scale 'Time' and 'Amount' columns
    print("Scaling 'Time' and 'Amount' features using StandardScaler...")
    scaler = StandardScaler()
    df['Scaled_Time'] = scaler.fit_transform(df[['Time']])
    df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])

    # Drop the original unscaled columns
    df = df.drop(['Time', 'Amount'], axis=1)

    # Reorder columns to put 'Class' (target) at the end or keep it structured
    print("Preparing features (X) and target (y)...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Stratified Train-Test Split (80% train, 20% test)
    # Stratification is critical for imbalanced data like fraud detection
    print("Performing stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Recombine features and target for saving
    train_df = X_train.copy()
    train_df['Class'] = y_train

    test_df = X_test.copy()
    test_df['Class'] = y_test

    # 4. Save processed datasets
    os.makedirs(processed_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    print(f"Saving training data to {train_path}...")
    train_df.to_csv(train_path, index=False)

    print(f"Saving testing data to {test_path}...")
    test_df.to_csv(test_path, index=False)

    print("\nFeature engineering and preprocessing completed successfully!")
    print(f"  - Train shape: {train_df.shape}")
    print(f"  - Test shape: {test_df.shape}\n")

if __name__ == "__main__":
    RAW_PATH = "data/raw/creditcard.csv"
    PROCESSED_DIR = "data/processed"
    
    process_features(raw_data_path=RAW_PATH, processed_dir=PROCESSED_DIR)