# Disease Prediction from Medical Data
**CodeAlpha Machine Learning Internship — Task 4**

Predicts the likelihood of three real diseases from structured clinical data, using classifiers trained and evaluated on genuine UCI medical datasets. No mocked numbers — every metric below came out of an actual training run (see `results/full_results_summary.json` for the raw output).

## Objective
Predict an individual's likelihood of a disease using structured medical/clinical features (symptoms, age, blood-test-style measurements), as specified in the task brief.

## Datasets (all real, all from UCI)
| Disease | Source | Rows | Features | Positive rate |
|---|---|---|---|---|
| Diabetes | UCI Pima Indians Diabetes Database | 768 | 8 | 34.9% |
| Heart Disease | UCI Cleveland Heart Disease | 302 | 13 | 54.3% |
| Breast Cancer | UCI Breast Cancer Wisconsin (Diagnostic) | 569 | 30 | 37.3% (malignant) |

Cleaning notes:
- Diabetes: physiologically-impossible zeros in glucose/blood pressure/skin thickness/insulin/BMI were treated as missing and median-imputed (a well-known data-quality issue in this specific dataset).
- Breast Cancer target was flipped from sklearn's default (0=malignant) to this project's convention (1 = condition present).
- All three had duplicates dropped.

## Approach
For each disease, four classifiers named in the task brief — **Logistic Regression, Random Forest, SVM, XGBoost** — were trained with:
- An 80/20 stratified train/test split
- Feature standardization (`StandardScaler`)
- 5-fold stratified cross-validation
- A real `GridSearchCV` hyperparameter search per model (not defaults)
- Selection of the best model per disease by held-out test ROC-AUC

## Results (held-out test set, 20% never seen during training/tuning)

### Diabetes — best model: **XGBoost**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.688 | 0.565 | 0.481 | 0.520 | 0.811 |
| Random Forest | 0.727 | 0.643 | 0.500 | 0.563 | 0.806 |
| SVM | 0.701 | 0.591 | 0.481 | 0.531 | 0.813 |
| **XGBoost** | **0.766** | **0.680** | **0.630** | **0.654** | **0.827** |

### Heart Disease — best model: **Random Forest**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.770 | 0.744 | 0.879 | 0.806 | 0.875 |
| **Random Forest** | **0.803** | 0.756 | **0.939** | **0.838** | **0.901** |
| SVM | 0.803 | 0.756 | 0.939 | 0.838 | 0.882 |
| XGBoost | 0.770 | 0.757 | 0.848 | 0.800 | 0.853 |

### Breast Cancer — best model: **Logistic Regression**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.965 | 0.975 | 0.929 | 0.951 | **0.996** |
| Random Forest | 0.965 | 1.000 | 0.905 | 0.950 | 0.994 |
| SVM | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| XGBoost | 0.956 | 1.000 | 0.881 | 0.937 | 0.994 |

Full per-model hyperparameters, confusion matrices, and ROC curves are in `results/`.

**Honest caveat on the diabetes results:** recall/F1 are visibly weaker than the other two diseases. This isn't a bug — the Pima dataset is small (768 rows), has real label noise, and is a well-known "ceiling ~0.80-0.83 AUC" benchmark in the literature. I'm reporting it as-is rather than tuning until it looks better.

## Project structure
```
DiseasePredictor/
├── data/
│   ├── prepare_datasets.py     # cleans raw data into diabetes_clean.csv / heart_clean.csv / breast_cancer_clean.csv
│   ├── diabetes_raw.csv, heart_raw2.csv        # raw sources
│   └── *_clean.csv                              # cleaned, model-ready data
├── train.py                    # trains all 4 algorithms × 3 diseases, saves best model per disease
├── models/                     # trained model + scaler + metadata (.pkl / .json) per disease
├── results/                    # confusion matrices, ROC curves, classification reports, full metrics JSON
└── app/
    ├── backend.py               # Flask API serving live predictions from the saved models
    └── static/index.html        # frontend — pick a disease, enter values, get a real prediction
```

## Running it yourself
```bash
pip install -r requirements.txt
python3 data/prepare_datasets.py   # rebuild clean CSVs from raw data
python3 train.py                   # retrain all models (~1-2 min)
python3 app/backend.py             # starts the web app on http://localhost:5050
```

## What each of the 6 CodeAlpha internship rules maps to
1. Internship status posted to LinkedIn tagging @CodeAlpha — done by you, outside this repo.
2. Assigned project completed within timeframe — this repo.
3. Source code uploaded to GitHub as `CodeAlpha_DiseasePrediction` — push this folder there.
4. Video walkthrough posted to LinkedIn with the GitHub link — record a 2–3 min demo of the running app (`app/static/index.html`) + a quick look at `results/`.
5. Submission form — submit once the above are live.
6. This is Task 4 of 4; combine with the emotion-recognition project (Task 2) already completed to satisfy "any 2 or 3 of 4."

## Not medical advice
This is a portfolio/educational project. The models are trained on small public benchmark datasets and are not validated for clinical use.
