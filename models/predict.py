import joblib
import os
import pandas as pd


def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

    model = joblib.load(os.path.join(ARTIFACT_DIR, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACT_DIR, "feature_names.pkl"))

    return model, scaler, feature_names


def predict(input_data, threshold=0.4):
    model, scaler, feature_names = load_artifacts()

    # Convert input to one-row DataFrame
    df = pd.DataFrame([input_data]).copy()

    # Simple missing handling for required numeric fields
    numeric_defaults = {
        "CreditScore": 600,
        "Age": 40,
        "Tenure": 5,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 0.0,
    }
    for col, default_val in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default_val
        df[col] = df[col].fillna(default_val)

    # Feature engineering (must match training)
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductDensity"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProduct"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

    # If raw category columns are provided, one-hot encode them
    cat_cols = [c for c in ["Geography", "Gender"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add any missing training columns and align order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale and predict
    df_scaled = scaler.transform(df)
    prob = float(model.predict_proba(df_scaled)[:, 1][0])
    pred = int(prob >= threshold)
    return prob, pred
