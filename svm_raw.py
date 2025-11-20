import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Removed StandardScaler import since we aren't using it
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import glob
import os

os.makedirs("plots", exist_ok=True)

# ================= CONFIGURATION =================
files_found = glob.glob("csvs/*top200_raw_expression_TumorLabels.csv")
if not files_found:
    raise FileNotFoundError("Could not find the 'TumorLabels.csv' file.")
    
INPUT_FILE = files_found[0]
print(f"Using input file: {INPUT_FILE}")
# =================================================

def run_svm_classification():
    
    # 1. Load Data
    df = pd.read_csv(INPUT_FILE, index_col=0)
    y = df.index
    X = df.values
    
    print(f"Loaded Data: {X.shape[0]} samples, {X.shape[1]} genes")

    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # 4. Scaling (REMOVED)
    # We are skipping StandardScaler because the data is already Log2 normalized.
    # If your data was raw counts, this step would be mandatory.
    print("Skipping Z-score scaling (using Log2 data directly)...")
    
    # 5. Train SVM
    svm_model = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    
    # --- CHANGED: Passing X_train directly instead of X_train_scaled ---
    svm_model.fit(X_train, y_train)
    
    # 6. Predict Class and Probabilities
    # --- CHANGED: Passing X_test directly ---
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    
    # 7. Evaluate Accuracy
    print("\n--- Model Performance ---")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Calculate ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {le.classes_[0]} vs {le.classes_[1]} (No Scaling)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('plots/SVM_ROC_Curve_NoScaling.png', dpi=300)
    plt.close()
    print("ROC Curve saved to 'plots/SVM_ROC_Curve_NoScaling.png'")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix\n(Accuracy: {acc:.2%})')
    plt.tight_layout()
    plt.savefig('plots/SVM_Confusion_Matrix_NoScaling.png', dpi=300)
    plt.close()
    print("Confusion matrix saved to 'plots/SVM_Confusion_Matrix_NoScaling.png'")

if __name__ == "__main__":
    run_svm_classification()