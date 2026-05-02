import shap
import pandas as pd
import numpy as np

class ModelExplainer:
    def __init__(self, model):
        self.model = model
        # For RandomForest, TreeExplainer is best
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_prediction(self, features: pd.DataFrame, prediction_label: str):
        """
        Returns a dictionary containing the SHAP values for the features
        to explain which feature contributed most to the prediction.
        """
        shap_values = self.explainer.shap_values(features)
        
        # Get the numeric index of the predicted label from the model's classes array
        class_idx = np.where(self.model.classes_ == prediction_label)[0][0]
        
        # Depending on the shap version and model type, shap_values might be a list or a 3D array
        if isinstance(shap_values, list):
            class_shap_values = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3:
            # shape is (n_samples, n_features, n_classes)
            class_shap_values = shap_values[0, :, class_idx]
        else:
            class_shap_values = shap_values[0]

        feature_names = features.columns.tolist()
        
        explanation = {
            feature_names[i]: float(class_shap_values[i]) 
            for i in range(len(feature_names))
        }
        
        return explanation
