import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("IRIS_Classification")
    
    if experiment:
        if experiment.lifecycle_stage == "deleted":
            print(f"Restoring deleted experiment '{experiment.name}' (ID: {experiment.experiment_id})...")
            client.restore_experiment(experiment.experiment_id)
            print("Successfully restored! You can now use the name 'IRIS_Classification' again.")
        else:
            print("Experiment is active and ready to use.")
    else:
        print("Experiment not found.")
else:
    print("No tracking URI found in .env.")
