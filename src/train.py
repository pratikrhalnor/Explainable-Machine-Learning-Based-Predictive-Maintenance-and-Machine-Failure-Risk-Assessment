import joblib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, average_precision_score, confusion_matrix)
from src.data_loader import load_data
from src.preprocessing import split_data, get_preprocessor

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "evaluation"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    prep = get_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                                 eval_metric="logloss", random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name, best_score, best_pipe = None, -1, None
    for name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        r = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "pr_auc": float(average_precision_score(y_test, y_proba)),
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = r
        if r["f1"] > best_score:
            best_score = r["f1"]; best_name = name; best_pipe = pipe
        print(name, r)

    joblib.dump(best_pipe, MODELS / "trained_model.pkl")
    joblib.dump(get_preprocessor().fit(X_train), MODELS / "preprocessing_pipeline.pkl")
    # save preprocessor fitted already inside pipeline; store separate fitted prep
    fitted_prep = best_pipe.named_steps["prep"]
    joblib.dump(fitted_prep, MODELS / "preprocessing_pipeline.pkl")
    with open(REPORTS / "model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    print("Best:", best_name)

if __name__ == "__main__":
    main()
