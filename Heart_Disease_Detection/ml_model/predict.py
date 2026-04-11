"""
ml_model/predict.py
Load the saved model and run predictions on new patient data.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
CLEANED_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned_data.csv')


def load_model(path: str = MODEL_PATH):
    clf = joblib.load(path)
    print(f"[✔] Model loaded from: {path}")
    return clf


def predict_single(patient_dict: dict, model=None) -> dict:
    """
    Predict for a single patient.

    patient_dict must match the feature columns of the cleaned dataset
    (excluding 'target').
    """
    if model is None:
        model = load_model()

    # Build a DataFrame with the correct column order
    sample_df = pd.read_csv(CLEANED_PATH).drop(columns=['target'])
    cols = sample_df.columns.tolist()

    row = pd.DataFrame([patient_dict])[cols]
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0]

    return {
        "prediction": int(pred),
        "label": "Heart Disease" if pred == 1 else "No Heart Disease",
        "probability_no_disease": round(prob[0], 4),
        "probability_disease":    round(prob[1], 4),
    }


def predict_batch(csv_path: str, model=None) -> pd.DataFrame:
    """Predict for a CSV file of patients."""
    if model is None:
        model = load_model()
    df = pd.read_csv(csv_path)
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    preds = model.predict(df)
    probs = model.predict_proba(df)
    df['prediction'] = preds
    df['label'] = df['prediction'].map({0: 'No Heart Disease', 1: 'Heart Disease'})
    df['prob_no_disease'] = probs[:, 0].round(4)
    df['prob_disease']    = probs[:, 1].round(4)
    return df


if __name__ == "__main__":
    # Demo single prediction
    sample = {
        'age': 0.6, 'sex': 1, 'cp': 0, 'trestbps': 0.7,
        'chol': 0.65, 'fbs': 1, 'restecg': 1, 'thalach': 0.4,
        'exang': 1, 'oldpeak': 0.5, 'slope': 1, 'ca': 2, 'thal': 3
    }
    result = predict_single(sample)
    print("\n[🩺] Prediction Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
