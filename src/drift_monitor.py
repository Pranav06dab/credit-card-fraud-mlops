import os
import pandas as pd
from scipy.stats import ks_2samp
import subprocess

def check_data_drift(reference_path: str, current_path: str, drift_threshold: float = 0.05):
    """
    Compares live production data against training data to detect statistical drift.
    Triggers retraining if significant drift is found.
    """
    print("=" * 50)
    print("PHASE 7: DATA DRIFT MONITORING")
    print("=" * 50)

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data missing at {reference_path}")
    
    if not os.path.exists(current_path):
        print(f"No live logs found at {current_path}. Not enough data to check for drift yet.")
        return

    print("Loading reference (training) and current (live log) datasets...")
    ref_df = pd.read_csv(reference_path)
    curr_df = pd.read_csv(current_path)

    # We only need a minimum amount of new data to run a valid statistical test
    if len(curr_df) < 10:
        print(f"Only {len(curr_df)} live transactions logged. Waiting for more data before running drift test.")
        return

    print(f"Analyzing {len(curr_df)} new live transactions for statistical drift...")
    
    # We will monitor the 'Scaled_Amount' feature as our primary indicator of changing transaction behavior
    feature_to_monitor = "Scaled_Amount"
    
    # Run the Kolmogorov-Smirnov test for goodness of fit
    statistic, p_value = ks_2samp(ref_df[feature_to_monitor], curr_df[feature_to_monitor])
    
    print(f"  - Monitored Feature: {feature_to_monitor}")
    print(f"  - KS Statistic: {statistic:.4f}")
    print(f"  - P-Value: {p_value:.4f}")

    # If the p-value is below our threshold, the distributions are statistically different
    if p_value < drift_threshold:
        print("\n[WARNING] SIGNIFICANT DATA DRIFT DETECTED!")
        print(f"The live data distribution for {feature_to_monitor} has shifted away from the training data.")
        
        # Trigger Phase 8: Automated Retraining
        print("\nInitializing Automated Retraining Pipeline...")
        trigger_retraining()
    else:
        print("\n[OK] No significant data drift detected. The current model is still valid.")

def trigger_retraining():
    """
    Executes the training script automatically to adapt to the drifted data.
    """
    try:
        # In a full production environment, this would also include appending 
        # the new verified logs to the raw data and rerunning features.py.
        # For this MLOps demonstration, we will trigger the model training pipeline.
        print("Running src/train.py...")
        subprocess.run(["python", "src/train.py"], check=True)
        print("\nAutomated Retraining Complete! The API will pull the new model on its next startup.")
    except subprocess.CalledProcessError as e:
        print(f"Automated retraining failed: {e}")

if __name__ == "__main__":
    import pathlib
    # Get the absolute root directory of your project (D:\MLOPS_CCFD)
    ROOT_DIR = pathlib.Path(__file__).parent.parent
    
    TRAIN_DATA_PATH = str(ROOT_DIR / "data" / "processed" / "train.csv")
    LIVE_LOGS_PATH = str(ROOT_DIR / "data" / "logs" / "predictions.csv")
    
    print(f"Checking reference path: {TRAIN_DATA_PATH}")
    print(f"Checking live logs path: {LIVE_LOGS_PATH}")
    
    check_data_drift(reference_path=TRAIN_DATA_PATH, current_path=LIVE_LOGS_PATH)