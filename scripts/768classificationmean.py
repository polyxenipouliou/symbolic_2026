import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import warnings

# Ignore warning messages to keep the output clean
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load MidiBERT 768-dimensional Feature Data
# ==========================================
# Define the path to the CSV file containing extracted features
csv_path = r"E:\Master\symbolic_2026\Adversarial-MidiBERT\composer_corpus_bert_features.csv"
print(f"[*] Loading large model feature data from: {csv_path}")

try:
    # Read the dataset into a DataFrame
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # Raise an error if the file is missing, guiding the user to check preprocessing steps
    raise FileNotFoundError(f"[!] File not found: {csv_path}. Please ensure the feature extraction script has completed successfully.")

# Exclude 'Clara Schumann' to maintain consistency with the Baseline experiment
if 'Clara Schumann' in df['composer'].values:
    df = df[df['composer'] != 'Clara Schumann'].copy()

# Extract feature columns (columns named 'bert_dim_0' through 'bert_dim_767')
feature_cols = [col for col in df.columns if col.startswith('bert_dim_')]
X_raw = df[feature_cols].values
y_labels = df['composer'].values

# Encode composer names into integer labels for model training
le = LabelEncoder()
y = le.fit_transform(y_labels)
classes = le.classes_

# Standardize features: Critical for SVM convergence in high-dimensional spaces (768D)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

print(f"[*] Data preparation complete: Samples={len(df)}, Feature Dimensions={len(feature_cols)}")

# ==========================================
# 2. Define SVM Model and Cross-Validation Strategy
# ==========================================
# Initialize SVM with RBF kernel, C=5.0, and balanced class weights to handle imbalance
svm_model = SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)

# Configure 5-Fold Stratified Cross-Validation to preserve class distribution in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 3. Execute Evaluation and Generate Results
# ==========================================
print("[*] Executing 5-Fold Cross-Validation...")

# Calculate balanced accuracy scores for each fold
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='balanced_accuracy')

# Generate predictions across all folds for detailed reporting
y_pred = cross_val_predict(svm_model, X_scaled, y, cv=cv)

# Generate text report containing precision, recall, and f1-score per class
report = classification_report(y, y_pred, target_names=classes)

# Calculate the overall aggregated balanced accuracy
bal_acc_final = balanced_accuracy_score(y, y_pred)

# ==========================================
# 4. Export Detailed Results to TXT File
# ==========================================
results_filename = 'results_768-3.txt'
with open(results_filename, 'w', encoding='utf-8') as f:
    f.write("=====================================================\n")
    f.write("    MidiBERT (768D) SVM Model Performance Report     \n")
    f.write("=====================================================\n\n")
    f.write(f"Model: Support Vector Machine (RBF Kernel, C=5.0)\n")
    f.write(f"Input: Adversarial-MidiBERT Embeddings (768 dimensions)\n")
    f.write(f"Classes: {', '.join(classes)}\n\n")

    f.write("--- 5-Fold Cross-Validation (Balanced Accuracy) ---\n")
    f.write(f"Individual Fold Scores: {cv_scores}\n")
    f.write(f"Mean Score: {np.mean(cv_scores):.4f}\n")
    f.write(f"Standard Deviation: {np.std(cv_scores):.4f}\n\n")

    f.write("--- Detailed Classification Report ---\n")
    f.write(report)
    f.write(f"\nOverall Balanced Accuracy (Aggregated): {bal_acc_final:.4f}\n")

print(f"[+] Detailed evaluation results saved to: {results_filename}")

# ==========================================
# 5. Plot and Save Confusion Matrix (Purple Theme)
# ==========================================
# Compute the confusion matrix based on true vs. predicted labels
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(7, 5))
# Create a heatmap with purple color map, displaying counts in each cell
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=classes, yticklabels=classes,
            cbar=False, annot_kws={"size": 14})

plt.title('Confusion Matrix: MidiBERT-768D (SVM)', fontsize=14, fontweight='bold')
plt.ylabel('True Composer', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Composer', fontsize=12, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()

# Save the figure with high resolution (300 DPI)
plt.savefig('confusion_matrix_midibert_svm_3.png', dpi=300)
print("[+] Confusion matrix image saved as 'confusion_matrix_midibert_svm_3.png'")
plt.show()