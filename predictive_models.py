import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import os
import numpy as np

def run_svm(csv_path):
    df = pd.read_csv(csv_path)

    y = df['Unnamed: 0']
    X = df.drop(columns=['Unnamed: 0'])

    counts = y.value_counts()
    total = len(y)

    weights = {
        c: total / (2 * counts[c])
        for c in counts.index
    }

    print("Class weights:", weights)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = LinearSVC(class_weight=weights, max_iter=10000)
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    svm.fit(X_train_scaled, y_train)

    calibrator = CalibratedClassifierCV(svm, cv='prefit', method='sigmoid')
    calibrator.fit(X_train_scaled, y_train)

    idx = list(calibrator.classes_).index('KICH')
    y_proba = calibrator.predict_proba(X_test_scaled)[:, idx]

    y_test_binary = (y_test == 'KICH').astype(int)

    auc = roc_auc_score(y_test_binary, y_proba)

    print("ROC-AUC (KICH vs KIRC):", auc)

    return {
        "svm": svm,
        "calibrator": calibrator,
        "scaler": scaler,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": auc
    }

def run_xgboost(data_path):

    print(f"--- Running Classification on: {os.path.basename(data_path)} ---")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Data file not found at {data_path}")
        return 0.0

    X = df.iloc[:, 1:]
    y_str = df.iloc[:, 0]

    le = LabelEncoder()
    y_numeric = le.fit_transform(y_str)
    tumor_names = le.classes_.tolist()

    counts = y_str.value_counts()
    c0 = counts[le.classes_[0]]
    c1 = counts[le.classes_[1]]
    scale_weight = max(c0, c1) / min(c0, c1)

    print(f"Successfully loaded {X.shape[0]} samples and {X.shape[1]} features (genes).")
    print(f"Imbalance detected: Majority={max(c0, c1)}, Minority={min(c0, c1)}")
    print(f"Setting scale_pos_weight = {scale_weight:.2f}")

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=100,
        random_state=42,
        tree_method='hist',
        scale_pos_weight=scale_weight
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=tumor_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_np = X.to_numpy()
    y_np = y_numeric
    scores = cross_val_score(model, X_np, y_np, cv=cv, scoring='accuracy', n_jobs=-1)
    avg_accuracy = scores.mean()

    print("\n" + "="*50)
    print("MODEL: XGBOOST CLASSIFIER")
    print("="*50)
    print(f"Tumor Pair: {tumor_names[0]} vs {tumor_names[1]}")
    print(f"Individual Fold Accuracies: {np.round(scores, 4)}")
    print(f"Average 5-Fold CV Accuracy: {avg_accuracy:.4f}")
    print(f"NOTE: Class weights applied (scale_pos_weight={scale_weight:.2f})")
    print("="*50)

    return {
        "model": model,
        "accuracy": avg_accuracy,
        "fold_scores": scores,
        "classification_report": classification_report(y_test, y_pred, target_names=tumor_names, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "tumor_names": tumor_names
    }

results_svm = [run_svm("./csvs/top200_zscored_by_full_distribution_TumorLabels.csv"),
               run_svm("./csvs/clock_expression_zscored_TumorLabels.csv"),
               run_svm("./csvs/top_100+11_clock_z_scores.csv")]

results_xgb = [run_xgboost("./csvs/top200_zscored_by_full_distribution_TumorLabels.csv"),
               run_xgboost("./csvs/clock_expression_zscored_TumorLabels.csv"),
               run_xgboost("./csvs/top_100+11_clock_z_scores.csv")]


