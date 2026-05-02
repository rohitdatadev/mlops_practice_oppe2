import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from dotenv import load_dotenv

# Load environment variables (MLFLOW_TRACKING_URI)
load_dotenv()

def run_data_poisoning_attack(data_path="raw_data.csv", poison_rate=0.10):
    """
    Simulates a Data Poisoning attack where an adversary modifies training labels
    to degrade the model's accuracy. Logs the results to MLFlow.
    """
    print(f"\n--- Initiating Data Poisoning Attack ({poison_rate*100}% Poisoning) ---")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['species'])
    y = df['species']
    
    # 2. Split Data (Before poisoning so the test set remains pure/unpoisoned)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Inject Poison into Training Data ONLY
    num_to_poison = int(len(y_train) * poison_rate)
    poison_indices = np.random.choice(y_train.index, num_to_poison, replace=False)
    
    # Randomly shuffle the labels of the selected poisoned indices
    y_train_poisoned = y_train.copy()
    y_train_poisoned.loc[poison_indices] = np.random.permutation(y_train.loc[poison_indices])
    
    print(f"Poisoned {num_to_poison} rows of training data.")

    # 4. Connect to MLFlow
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment("IRIS_Classification")
    
    # 5. Train Model & Log to MLFlow
    with mlflow.start_run(run_name=f"Poisoned_Attack_{int(poison_rate*100)}pct"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_poisoned)
        
        # Test on the PURE test set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Log metrics and parameters
        mlflow.log_param("poison_rate", poison_rate)
        mlflow.log_param("num_poisoned_rows", num_to_poison)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"Model Accuracy after {poison_rate*100}% poisoning: {accuracy:.4f}")
        print("Logged attack metrics to MLFlow successfully.\n")

if __name__ == "__main__":
    # Test various levels of poisoning
    run_data_poisoning_attack(poison_rate=0.05) # 5% poisoning
    run_data_poisoning_attack(poison_rate=0.10) # 10% poisoning
    run_data_poisoning_attack(poison_rate=0.50) # 50% poisoning
