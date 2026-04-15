# Code for training models


def train_logistic(X_train_scaled, y_train, scaler, X_train):
    from sklearn.linear_model import LogisticRegression
    import joblib
    import os

    print("Training started...")

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    print("Model trained")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(MODEL_DIR, "feature_names.pkl"))

    print("Files saved successfully.")

    return model