import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
# Load the dataset containing statistical features
# Note: Ensure the file path is correct for your environment
df = pd.read_csv(r'E:\Master\symbolic_2026\features\filtered_features_statistical.csv')

# Exclude 'Clara Schumann' to focus analysis on the three core composers
df = df[df['composer'] != 'Clara Schumann'].copy()

# Separate features (X) and target labels (y)
# Drop 'filename' and 'composer' columns to get pure numerical features
X_raw = df.drop(columns=['filename', 'composer']).values
y_labels = df['composer'].values

# Encode string labels into integers for model compatibility
le = LabelEncoder()
y = le.fit_transform(y_labels)
classes = le.classes_  # Store original class names for later reporting

# Feature Scaling: Crucial for SVM as it is sensitive to feature magnitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ==========================================
# 2. Define SVM Model and Cross-Validation Strategy
# ==========================================
# Initialize SVM with RBF kernel
# 'class_weight=balanced' automatically adjusts weights inversely proportional to class frequencies
svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

# Configure Stratified K-Fold Cross-Validation (5 splits)
# Stratification ensures each fold has the same proportion of classes as the whole dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# 3. Execute Evaluation and Generate Results
# ==========================================

# Calculate Balanced Accuracy scores across the 5 folds
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='balanced_accuracy')

# Generate out-of-fold predictions for the entire dataset to build a comprehensive report
y_pred = cross_val_predict(svm_model, X_scaled, y, cv=cv)

# Generate detailed classification report (Precision, Recall, F1-Score per class)
report = classification_report(y, y_pred, target_names=classes)

# Calculate the overall Balanced Accuracy based on all predictions
bal_acc_final = balanced_accuracy_score(y, y_pred)

# ==========================================
# 4. Export Detailed Results to TXT File
# ==========================================
results_filename = 'results_12.txt'
with open(results_filename, 'w', encoding='utf-8') as f:
    f.write("=====================================================\n")
    f.write("       SVM Model Performance Detailed Report         \n")
    f.write("=====================================================\n\n")
    f.write(f"Model: Support Vector Machine (RBF Kernel)\n")
    f.write(f"Features: Statistical Features (Filtered)\n")
    f.write(f"Classes: {', '.join(classes)}\n\n")

    f.write("--- 5-Fold Cross-Validation (Balanced Accuracy) ---\n")
    f.write(f"Individual Fold Scores: {cv_scores}\n")
    f.write(f"Mean Score: {np.mean(cv_scores):.4f}\n")
    f.write(f"Standard Deviation: {np.std(cv_scores):.4f}\n\n")

    f.write("--- Detailed Classification Report ---\n")
    f.write(report)
    f.write(f"\nOverall Balanced Accuracy: {bal_acc_final:.4f}\n")

print(f"[+] Detailed evaluation results saved to: {results_filename}")

# ==========================================
# 5. Plot and Save Confusion Matrix
# ==========================================
# Compute the confusion matrix based on true vs. predicted labels
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(7, 5))
# Draw heatmap using Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            cbar=False, annot_kws={"size": 14})

plt.title('Confusion Matrix: SVM (Statistical Features)', fontsize=14, fontweight='bold')
plt.ylabel('True Composer', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Composer', fontsize=12, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()

# Save the figure with high resolution
plt.savefig('confusion_matrix_svm.png', dpi=300)
print("[+] Confusion matrix image saved as 'confusion_matrix_svm.png'")
plt.show()