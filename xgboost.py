import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier 
import os

# ==========================================
#              USER SETTINGS
# ==========================================
# IMPORTANT: This CSV should contain ONLY the 200 genes + 1 column for Tumor Label
# The data within the gene columns is assumed to be Z-score standardized.
DATA_CSV_PATH = "./csvs/top200_zscored_by_full_distribution_TumorLabels.csv" 
# ==========================================

def run_xgboost_classification(data_path):
    """
    Loads pre-processed, standardized data and trains an XGBoost classifier 
    using 5-Fold Cross-Validation, applying weights to correct for class imbalance.
    """
    print(f"--- Running Classification on: {os.path.basename(data_path)} ---")

    # 1. LOAD DATA
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Data file not found at {data_path}")
        return 0.0

    # Assume the last column contains the Tumor Label (KICH/KIRC)
    label_column = df.columns[0]
    
    # 2. SEPARATE FEATURES (X) AND LABELS (Y)
    X = df.iloc[:, :0] # All columns except the last one (labels)
    y_str = df[label_column]

    # 3. ENCODE LABELS (REQUIRED STEP)
    le = LabelEncoder()
    y_numeric = le.fit_transform(y_str)
    
    tumor_names = le.classes_.tolist()
    
    # --- CRITICAL NEW STEP: Calculate Class Weight ---
    
    # Count samples for each class
    counts = y_str.value_counts()
    
    # The convention is to map the minority class to '1' (positive)
    # and the majority class to '0' (negative). 
    # LabelEncoder assigns classes alphabetically by default.
    
    # Determine which class is the minority and which is the majority
    # Get the label that was encoded as '0' and the label encoded as '1'
    count_0 = counts[le.classes_[0]]
    count_1 = counts[le.classes_[1]]
    
    # Calculate scale_pos_weight: Ratio of (Negative/Majority) to (Positive/Minority)
    if count_0 > count_1:
        # Class 0 is majority, Class 1 is minority (our positive class)
        scale_weight = count_0 / count_1
        positive_class_label = le.classes_[1]
    else:
        # Class 1 is majority, Class 0 is minority (our positive class)
        scale_weight = count_1 / count_0
        positive_class_label = le.classes_[0]
        
    print(f"Successfully loaded {X.shape[0]} samples and {X.shape[1]} features (genes).")
    print(f"Imbalance detected: Majority={max(count_0, count_1)}, Minority={min(count_0, count_1)}")
    print(f"Setting scale_pos_weight = {scale_weight:.2f} (for {positive_class_label}).")
    
    # 4. MODEL SETUP (Includes new weighting parameter)
    model = XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False, 
        eval_metric='logloss',
        n_estimators=100, 
        random_state=42,
        tree_method='hist',
        # --- NEW PARAMETER ---
        scale_pos_weight=scale_weight 
        # --- NEW PARAMETER ---
    )
    
    # 5. CROSS-VALIDATION PROTOCOL (The Gold Standard)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nStarting 5-Fold Cross-Validation...")
    
    # Calculate accuracy for each fold
    scores = cross_val_score(model, X, y_numeric, cv=cv, scoring='accuracy', n_jobs=-1)
    
    avg_accuracy = scores.mean()
    
    # 6. RESULTS
    print("\n" + "="*50)
    print("MODEL 1: TOP 200 GENES (XGBOOST)")
    print("="*50)
    print(f"Tumor Pair: {tumor_names[0]} vs {tumor_names[1]}")
    print(f"Individual Fold Accuracies: {np.round(scores, 4)}")
    print(f"Average 5-Fold CV Accuracy: {avg_accuracy:.4f}")
    print(f"NOTE: Class weights applied (scale_pos_weight={scale_weight:.2f})")
    print("="*50)
    
    return avg_accuracy

if __name__ == "__main__":
    run_xgboost_classification(DATA_CSV_PATH)