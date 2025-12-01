from fastapi import HTTPException
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

class PredictionService:
    def __init__(self):
        # Resolve model files relative to this module so loading works regardless of cwd
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_dir = os.path.normpath(os.path.join(base, 'models'))

        model_path = os.path.join(models_dir, 'nn_stress_model.h5')
        scaler_path = os.path.join(models_dir, 'nn_scaler.joblib')
        encoders_path = os.path.join(models_dir, 'nn_label_encoders.joblib')
        target_enc_path = os.path.join(models_dir, 'nn_target_encoder.joblib')

        # Helpful error messages when files are missing
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise RuntimeError(f"Scaler not found: {scaler_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        self.label_encoder = joblib.load(target_enc_path)

    def preprocess_input(self, input_dict):
        input_df = pd.DataFrame([input_dict])
        
        for col in input_df.columns:
            if pd.isnull(input_df[col]).any():
                if col in self.label_encoders:
                    input_df[col] = input_df[col].fillna(input_df[col].mode()[0])
                else:
                    input_df[col] = input_df[col].fillna(input_df[col].mean())
            
            if col in self.label_encoders:
                input_df[col] = self.label_encoders[col].transform(input_df[col])
        
        X_scaled = self.scaler.transform(input_df)
        return X_scaled

    def predict_stress_level(self, input_dict):
        try:
            X_scaled = self.preprocess_input(input_dict)
            pred_prob = self.model.predict(X_scaled)
            pred_class = np.argmax(pred_prob, axis=1)
            return self.label_encoder.inverse_transform(pred_class)[0]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))