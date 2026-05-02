import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Set the tracking URI if provided in .env
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using remote MLFlow Tracking URI: {tracking_uri}")
    else:
        print("No MLFLOW_TRACKING_URI found in .env, using local tracking.")

    # Set up MLFlow experiment
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "IRIS_Classification")
    mlflow.set_experiment(experiment_name)

    # Load data from the provided data.csv
    # Assuming script is run from the project root folder
    data_path = "raw_data.csv"
    if not os.path.exists(data_path):
        data_path = "../../raw_data.csv" # fallback if run from inside src/models
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Features and target
    X = df.drop("species", axis=1)
    y = df["species"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    with mlflow.start_run():
        print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Initialize and train model
        clf = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"Metrics - Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # Log parameters to MLFlow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Log metrics to MLFlow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model artifact
        mlflow.sklearn.log_model(clf, "random_forest_model")
        print("Model and metrics successfully logged to MLFlow!")

if __name__ == "__main__":
    main()