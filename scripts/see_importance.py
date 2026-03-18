import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Data Loading and Robust Preprocessing
# ==========================================
# Load the dataset from the specified path
df = pd.read_csv(r'E:\Master\symbolic_2026\feature_12+54+768.csv')

# Exclude data related to 'Clara Schumann' to focus on other composers
df = df[df['composer'] != 'Clara Schumann'].copy()

# Define non-feature columns explicitly (filename and composer)
# This is more robust than relying on column name prefixes
non_feature_cols = ['filename', 'composer']
feature_names = [col for col in df.columns if col not in non_feature_cols]

print(f"[*] Initially identified features: {len(feature_names)}")

# Force conversion to numeric types; coerce errors to NaN to handle dirty data
X_df = df[feature_names].apply(pd.to_numeric, errors='coerce')

# Handle missing values: Fill with median instead of dropping rows/columns
# This preserves data volume while mitigating the impact of outliers or errors
if X_df.isnull().sum().sum() > 0:
    print(f"[!] Warning: Found {X_df.isnull().sum().sum()} invalid data points. Filling with median values.")
    X_df = X_df.fillna(X_df.median())

# Check for and remove any columns that are entirely empty after cleaning
empty_cols = X_df.columns[X_df.isnull().all()].tolist()
if empty_cols:
    print(f"[!] Error: The following columns contain only invalid data and have been removed: {empty_cols}")
    X_df = X_df.drop(columns=empty_cols)

# Update feature list and extract raw numerical array
feature_names = X_df.columns.tolist()
X_raw = X_df.values

# Critical check: Ensure there are valid features remaining
if X_raw.shape[1] == 0:
    print("Column list:", df.columns.tolist())
    raise ValueError("Fatal Error: No valid numerical feature columns found! Please check the CSV headers.")

# Encode target labels (composer names) into integers
y_labels = df['composer'].values
le = LabelEncoder()
y = le.fit_transform(y_labels)
classes = le.classes_.tolist()

# Scale features using StandardScaler (mean=0, std=1) for SVM optimization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"[*] Final number of features determined: {X_raw.shape[1]}")

# ==========================================
# 2. Phase 1: Random Forest Feature Importance Ranking
# ==========================================
print("[*] Calculating Random Forest feature importance...")
# Initialize RF with balanced class weights and a high number of estimators for stability
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort indices by importance in descending order

# Reorganize feature names and scores based on the sorted indices
sorted_features = [feature_names[i] for i in indices]
sorted_scores = [float(importances[i]) for i in indices]

# ==========================================
# 3. Phase 2: Incremental Validation (Enhanced Information)
# ==========================================
print("[*] Starting incremental feature experimentation...")
# Use Stratified K-Fold to maintain class distribution across splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_history = []

# Iterate from 1 feature up to all features, adding them in order of importance
for k in range(1, len(sorted_features) + 1):
    current_feature_indices = indices[:k]
    X_subset = X_scaled[:, current_feature_indices]

    # Initialize SVM with RBF kernel and balanced class weights
    svm = SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)

    # Perform cross-validation and capture scores for each fold
    fold_scores = cross_val_score(svm, X_subset, y, cv=cv, scoring='balanced_accuracy')

    # Construct a detailed result dictionary for this step
    res = {
        "num_features": k,
        "last_added_feature": sorted_features[k - 1],
        "last_feature_importance": sorted_scores[k - 1],
        "current_feature_set": sorted_features[:k],  # Save the full set of features used
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_dev": float(np.std(fold_scores)),
        "raw_fold_scores": fold_scores.tolist()  # Save individual fold scores for deeper analysis
    }
    results_history.append(res)

    # Print progress every 10 steps or at the first step
    if k % 10 == 0 or k == 1:
        print(f"  - Features: {k:2d}, Acc = {res['mean_accuracy']:.4f}")

# ==========================================
# 4. Save Enhanced Results to JSON
# ==========================================
output_data = {
    "metadata": {
        "project": "Composer Classification",
        "model": "SVM (RBF, C=5.0)",
        "feature_selection_method": "Random Forest Importance",
        "total_samples": len(df)
    },
    "composer_classes": classes,
    "feature_ranking_full": [
        {"rank": i + 1, "feature": f, "importance": s}
        for i, (f, s) in enumerate(zip(sorted_features, sorted_scores))
    ],
    "incremental_steps": results_history
}

json_filename = 'feature_selection_results_12+54+768.json'
with open(json_filename, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"\n[+] Experiment complete! Enhanced data saved to: {json_filename}")

# ==========================================
# 5. Visualization (Professional Style)
# ==========================================
# Extract data for plotting
k_values = [r['num_features'] for r in results_history]
acc_values = [r['mean_accuracy'] for r in results_history]
std_values = [r['std_dev'] for r in results_history]

plt.figure(figsize=(10, 6))
# Plot mean accuracy curve
plt.plot(k_values, acc_values, color='#8e44ad', linewidth=2, label='Mean Balanced Accuracy')
# Fill area representing standard deviation (stability)
plt.fill_between(k_values,
                 np.array(acc_values) - np.array(std_values),
                 np.array(acc_values) + np.array(std_values),
                 color='#9b59b6', alpha=0.2, label='Stability (±Std Dev)')

# Identify the optimal number of features (highest mean accuracy)
best_k = k_values[np.argmax(acc_values)]
best_acc = max(acc_values)

# Mark the optimal point on the graph
plt.axvline(x=best_k, color='r', linestyle='--', alpha=0.5)
plt.scatter(best_k, best_acc, color='red', s=50, zorder=5)

plt.title('SVM Feature Selection Curve (RF Ranked)', fontsize=14)
plt.xlabel('Number of Top Features')
plt.ylabel('Balanced Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('feature_accuracy_curve_12+54+768.png', dpi=300)
plt.show()