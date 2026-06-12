import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')

# Load features
print("📂 Loading features and best model...")
X_test  = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_test.npz'))
y_test  = pd.read_csv(os.path.join(FEATURES_DIR, 'y_test.csv'))['class_label'].values
X_train = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_train.npz'))
y_train = pd.read_csv(os.path.join(FEATURES_DIR, 'y_train.csv'))['class_label'].values
X_val   = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_val.npz'))
y_val   = pd.read_csv(os.path.join(FEATURES_DIR, 'y_val.csv'))['class_label'].values

# Load best model and scaler
best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# Scale
X_test_scaled  = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Predictions
y_pred_test  = best_model.predict(X_test_scaled)
y_pred_train = best_model.predict(X_train_scaled)
y_pred_val   = best_model.predict(X_val_scaled)

classes = sorted(list(set(y_test)))

# ── 1. Confusion Matrix ────────────────────────────────────────
print("\n📊 Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_test, labels=classes)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, ax=ax)
ax.set_title('Confusion Matrix - Best Model (Logistic Regression)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'confusion_matrix.png'))
plt.close()
print("   Saved confusion_matrix.png")

# ── 2. Classification Report ───────────────────────────────────
print("\n📋 Classification Report:")
report = classification_report(y_test, y_pred_test, target_names=classes)
print(report)

report_dict = classification_report(y_test, y_pred_test,
                                    target_names=classes, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(OUTPUTS_DIR, 'classification_report.csv'))
print("   Saved classification_report.csv")

# ── 3. Per Class Accuracy Bar Chart ───────────────────────────
print("\n📊 Generating per class accuracy chart...")
per_class_acc = {}
for cls in classes:
    mask = y_test == cls
    if mask.sum() > 0:
        per_class_acc[cls] = accuracy_score(y_test[mask], y_pred_test[mask])

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(per_class_acc.keys(), per_class_acc.values(),
              color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'],
              edgecolor='black')
for bar, val in zip(bars, per_class_acc.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontweight='bold')
ax.set_title('Per Class Accuracy - Best Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'per_class_accuracy.png'))
plt.close()
print("   Saved per_class_accuracy.png")

# ── 4. Train vs Val vs Test Accuracy ──────────────────────────
train_acc = accuracy_score(y_train, y_pred_train)
val_acc   = accuracy_score(y_val,   y_pred_val)
test_acc  = accuracy_score(y_test,  y_pred_test)

fig, ax = plt.subplots(figsize=(8, 5))
splits  = ['Train', 'Validation', 'Test']
accs    = [train_acc, val_acc, test_acc]
colors  = ['#4ECDC4', '#45B7D1', '#FF6B6B']
bars    = ax.bar(splits, accs, color=colors, edgecolor='black', width=0.4)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontweight='bold')
ax.set_title('Train vs Validation vs Test Accuracy',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'train_val_test_accuracy.png'))
plt.close()
print("   Saved train_val_test_accuracy.png")

# ── 5. Summary ────────────────────────────────────────────────
print("\n" + "="*50)
print("        EVALUATION SUMMARY")
print("="*50)
print(f"  Best Model    : Logistic Regression")
print(f"  Train Acc     : {train_acc:.4f}")
print(f"  Val Acc       : {val_acc:.4f}")
print(f"  Test Acc      : {test_acc:.4f}")
print(f"  Classes       : {len(classes)}")
print("="*50)
print("\n✅ All evaluation files saved to outputs/")
