import joblib
import pandas as pd

# Load artifacts
model = joblib.load("../artifacts/churn_model.pkl")
scaler = joblib.load("../artifacts/scaler.pkl")
feature_names = joblib.load("../artifacts/feature_names.pkl")


# Example NEW customer input

input_data = {
    "CreditScore": 600,
    "Age": 40,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 70000,
    "Geography_Germany": 0,
    "Geography_Spain": 1,
    "Gender_Male": 1
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Ensure correct feature order
df = df[feature_names]

# Scale input
df_scaled = scaler.transform(df)

# Predict
prob = model.predict_proba(df_scaled)[:, 1]

# Apply threshold (your best)
threshold = 0.4
prediction = (prob >= threshold).astype(int)

# Output
print("Churn Probability:", round(prob[0], 3))
print("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")