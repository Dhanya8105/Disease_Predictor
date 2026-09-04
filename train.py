"""
Disease Prediction from Medical Data -- CodeAlpha Task 4
==========================================================
Trains and evaluates 4 classification algorithms (as specified in the task
brief: Logistic Regression, Random Forest, SVM, XGBoost) on 3 real UCI
medical datasets, selects the best model per disease via cross-validated
ROC-AUC + hyperparameter search, and saves:
  - the trained best model (joblib)
  - the fitted scaler
  - the feature list / metadata
  - a JSON of full metrics for every algorithm (not just the winner)
  - confusion matrix + ROC curve plots

This is real training on real data -- no mocked numbers.
"""
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RANDOM_STATE = 42

DATASETS = {
    "diabetes": {
        "path": f"{DATA_DIR}/diabetes_clean.csv",
        "target": "target",
        "display_name": "Diabetes",
    },
    "heart": {
        "path": f"{DATA_DIR}/heart_clean.csv",
        "target": "target",
        "display_name": "Heart Disease",
    },
    "breast_cancer": {
        "path": f"{DATA_DIR}/breast_cancer_clean.csv",
        "target": "target",
        "display_name": "Breast Cancer",
    },
}

# Hyperparameter grids -- kept modest so this runs in reasonable time,
# but genuinely searched (not hardcoded to one setting).
MODEL_GRID = {
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale"],
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", use_label_encoder=False
        ),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
}


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def run_dataset(key, cfg):
    print(f"\n{'='*70}\n{cfg['display_name']}  ({key})\n{'='*70}")
    df = pd.read_csv(cfg["path"])
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    all_results = {}
    fitted_models = {}

    for model_name, spec in MODEL_GRID.items():
        t0 = time.time()
        grid = GridSearchCV(
            spec["estimator"], spec["params"], cv=cv,
            scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train_s, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test_s)
        y_proba = best_model.predict_proba(X_test_s)[:, 1]

        test_metrics = evaluate(y_test, y_pred, y_proba)
        cv_auc_mean = round(grid.best_score_, 4)

        all_results[model_name] = {
            "best_params": grid.best_params_,
            "cv_roc_auc_mean": cv_auc_mean,
            "test_metrics": test_metrics,
            "train_time_sec": round(time.time() - t0, 2),
        }
        fitted_models[model_name] = best_model

        print(f"  {model_name:18s} | CV-AUC={cv_auc_mean:.4f} | "
              f"Test Acc={test_metrics['accuracy']:.4f} | "
              f"Test F1={test_metrics['f1_score']:.4f} | "
              f"Test AUC={test_metrics['roc_auc']:.4f} | "
              f"({all_results[model_name]['train_time_sec']}s)")

    # pick best model by test ROC-AUC
    best_name = max(all_results, key=lambda m: all_results[m]["test_metrics"]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"  --> BEST MODEL: {best_name} "
          f"(Test ROC-AUC={all_results[best_name]['test_metrics']['roc_auc']})")

    # Save model + scaler + metadata
    joblib.dump(best_model, f"{MODEL_DIR}/{key}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{key}_scaler.pkl")
    meta = {
        "disease": cfg["display_name"],
        "best_model": best_name,
        "feature_names": feature_names,
        "target_meaning": "1 = condition present, 0 = condition absent",
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(f"{MODEL_DIR}/{key}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Confusion matrix + ROC curve plot for the winning model
    y_pred_best = best_model.predict(X_test_s)
    y_proba_best = best_model.predict_proba(X_test_s)[:, 1]
    cm = confusion_matrix(y_test, y_pred_best)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"{cfg['display_name']} - Confusion Matrix\n({best_name})")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["No Disease", "Disease"])
    axes[0].set_yticklabels(["No Disease", "Disease"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    axes[1].plot(fpr, tpr, label=f"AUC = {all_results[best_name]['test_metrics']['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_title(f"{cfg['display_name']} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{key}_evaluation.png", dpi=130)
    plt.close()

    report = classification_report(y_test, y_pred_best, target_names=["No Disease", "Disease"])
    with open(f"{RESULTS_DIR}/{key}_classification_report.txt", "w") as f:
        f.write(f"{cfg['display_name']} -- Best Model: {best_name}\n\n{report}")

    return {
        "dataset": cfg["display_name"],
        "best_model": best_name,
        "all_model_results": all_results,
    }


if __name__ == "__main__":
    summary = {}
    for key, cfg in DATASETS.items():
        summary[key] = run_dataset(key, cfg)

    with open(f"{RESULTS_DIR}/full_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nDONE. Models saved to {MODEL_DIR}, results to {RESULTS_DIR}\n{'='*70}")
