import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow
from dotenv import load_dotenv

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from api.schemas import IrisRequest, IrisResponse
from explainability.explainer import ModelExplainer

app = FastAPI(title="IRIS Classification API with Explainability")

# Load environment variables
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# Global variables to hold model and explainer
model = None
explainer = None

@app.on_event("startup")
def load_model():
    global model, explainer
    print("Loading latest model from MLFlow...")
    try:
        # Search for the latest run in the experiment
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "IRIS_Classification")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise Exception("Experiment not found in MLFlow")
            
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], 
                                  order_by=["metrics.accuracy DESC"], max_results=1)
        if runs.empty:
            raise Exception("No runs found in experiment")
            
        best_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{best_run_id}/random_forest_model"
        
        # Download and load the model directly from the remote MLFlow server!
        model = mlflow.sklearn.load_model(model_uri)
        explainer = ModelExplainer(model)
        print("Model and SHAP explainer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict", response_model=IrisResponse)
def predict(request: IrisRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded from MLFlow yet.")
        
    # Convert request to pandas DataFrame
    df = pd.DataFrame([{
        "sepal_length": request.sepal_length,
        "sepal_width": request.sepal_width,
        "petal_length": request.petal_length,
        "petal_width": request.petal_width
    }])
    
    # 1. Make Prediction
    prediction = model.predict(df)[0]
    
    # 2. Get Explainability (SHAP values)
    explanation = explainer.explain_prediction(df, prediction)
    
    return IrisResponse(
        prediction=str(prediction),
        explanation=explanation
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
