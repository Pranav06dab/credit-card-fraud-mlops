import os
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 1. Initialize the FastAPI Web Application
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An MLOps REST API serving a Random Forest fraud detection model."
)

# 2. Define the expected input data structure
class TransactionInput(BaseModel):
    # We expect a dictionary of the 30 features (V1-V28, Scaled_Time, Scaled_Amount)
    features: dict

def load_latest_model():
    """
    Bypasses the MLflow database entirely and performs a raw physical search 
    inside the Docker container for the model artifact.
    """
    import os
    import mlflow.sklearn
    
    print("Bypassing database to physically locate the model inside Docker...")
    
    # os.walk will aggressively search every single folder inside /app/mlruns
    for root, dirs, files in os.walk("/app/mlruns"):
        if "MLmodel" in files:  # The unique signature of an MLflow model folder
            print(f"Model found physically at: {root}")
            try:
                model = mlflow.sklearn.load_model(root)
                print("Model loaded successfully!\n")
                return model
            except Exception as e:
                print(f"Failed to load the found model: {e}")
                return None
                
    print("Critical Error: No MLmodel file exists in /app/mlruns.")
    return None
# Load the model globally when the server starts
MODEL = load_latest_model()

# 3. Define the Root Endpoint
@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is Live!"}

# 4. Define the Prediction Endpoint
@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    """
    Accepts transaction features, makes a prediction, and returns the result.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert the incoming dictionary into a Pandas DataFrame
        input_df = pd.DataFrame([transaction.features])
        
        # Make the binary prediction (0 = Legitimate, 1 = Fraud)
        prediction = MODEL.predict(input_df)[0]
        
        # Get the confidence probability of the transaction being fraud (Class 1)
        probability = MODEL.predict_proba(input_df)[0][1]
        
        result_label = "Fraud" if prediction == 1 else "Legitimate"
        
        return {
            "prediction": result_label,
            "fraud_probability_score": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    # Start the local development server
    print("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)