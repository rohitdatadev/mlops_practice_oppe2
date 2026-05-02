import mlflow
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

experiment = mlflow.get_experiment_by_name("IRIS_Classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
latest_run_id = runs.iloc[0]["run_id"]
print(f"Latest run id: {latest_run_id}")
model_uri = f"runs:/{latest_run_id}/random_forest_model"
print(f"Loading from {model_uri}")
try:
    model = mlflow.sklearn.load_model(model_uri)
    print("Loaded successfully")
except Exception as e:
    print(f"Failed to load: {e}")
