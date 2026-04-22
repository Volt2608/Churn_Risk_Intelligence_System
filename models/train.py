import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic(X_train_scaled, y_train):
    """Baseline model used for interpretability."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)
    return model


def train_decision_tree(X_train_scaled, y_train):
    """Simple tree baseline."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train_scaled, y_train):
    """Tree ensemble model."""
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model


def train_gradient_boosting(X_train_scaled, y_train):
    """Advanced boosting model (required in PRD)."""
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model


def train_xgboost_optional(X_train_scaled, y_train):
    """Optional advanced model. Returns None if xgboost is unavailable."""
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model


def train_all_models(X_train_scaled, y_train):
    """Beginner-friendly helper to train all PRD models in one call."""
    models = {
        "logistic_regression": train_logistic(X_train_scaled, y_train),
        "decision_tree": train_decision_tree(X_train_scaled, y_train),
        "random_forest": train_random_forest(X_train_scaled, y_train),
        "gradient_boosting": train_gradient_boosting(X_train_scaled, y_train),
    }
    xgb_model = train_xgboost_optional(X_train_scaled, y_train)
    if xgb_model is not None:
        models["xgboost"] = xgb_model
    return models


def save_artifacts(model, scaler, feature_names):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(ARTIFACT_DIR, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(ARTIFACT_DIR, "feature_names.pkl"))

    print("Artifacts saved successfully.")
