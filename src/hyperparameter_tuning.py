import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')

print("📂 Loading features...")
X_train = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_train.npz'))
X_val   = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_val.npz'))
X_test  = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_test.npz'))
y_train = pd.read_csv(os.path.join(FEATURES_DIR, 'y_train.csv'))['class_label'].values
y_val   = pd.read_csv(os.path.join(FEATURES_DIR, 'y_val.csv'))['class_label'].values
y_test  = pd.read_csv(os.path.join(FEATURES_DIR, 'y_test.csv'))['class_label'].values

X_trainval = sp.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

scaler        = MaxAbsScaler()
X_trainval_sc = scaler.fit_transform(X_trainval)
X_test_sc     = scaler.transform(X_test)

print("\n🔍 Running GridSearchCV on Logistic Regression...")
param_grid = {
    'C':        [0.01, 0.1, 1, 10, 100],
    'solver':   ['lbfgs', 'saga'],
    'max_iter': [500, 1000]
}
grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_trainval_sc, y_trainval)
print(f"\n✅ Best Parameters : {grid_search.best_params_}")
print(f"   Best CV Score   : {grid_search.best_score_:.4f}")

best_tuned = grid_search.best_estimator_
test_acc   = accuracy_score(y_test, best_tuned.predict(X_test_sc))
print(f"   Test Accuracy   : {test_acc:.4f}")

print("\n📊 Running 5-Fold Cross Validation...")
cv_scores = cross_val_score(best_tuned, X_trainval_sc, y_trainval, cv=5)
print(f"   CV Scores  : {cv_scores}")
print(f"   CV Mean    : {cv_scores.mean():.4f}")
print(f"   CV Std Dev : {cv_scores.std():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, 6), cv_scores, color='#45B7D1', edgecolor='black')
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax.set_title('5-Fold Cross Validation Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'cv_scores.png'))
plt.close()
print("   Saved cv_scores.png")

print("\n📊 Plotting C parameter effect...")
c_values = [0.01, 0.1, 1, 10, 100]
c_scores = []
for c in c_values:
    model  = LogisticRegression(C=c, max_iter=1000, random_state=42)
    scores = cross_val_score(model, X_trainval_sc, y_trainval, cv=5)
    c_scores.append(scores.mean())

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(c_values, c_scores, marker='o', color='#FF6B6B', linewidth=2)
ax.set_xscale('log')
ax.set_title('C Parameter Effect on Accuracy', fontsize=14, fontweight='bold')
ax.set_xlabel('C Value (log scale)', fontsize=12)
ax.set_ylabel('Cross Val Accuracy', fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, 'c_parameter_effect.png'))
plt.close()
print("   Saved c_parameter_effect.png")

joblib.dump(best_tuned, os.path.join(MODELS_DIR, 'best_model_tuned.pkl'))
joblib.dump(scaler,     os.path.join(MODELS_DIR, 'scaler_tuned.pkl'))

tuning_results = {
    'best_params':   grid_search.best_params_,
    'best_cv_score': round(grid_search.best_score_, 4),
    'test_accuracy': round(test_acc, 4),
    'cv_mean':       round(cv_scores.mean(), 4),
    'cv_std':        round(cv_scores.std(),  4),
}
with open(os.path.join(OUTPUTS_DIR, 'tuning_results.json'), 'w') as f:
    json.dump(tuning_results, f, indent=2)

print("\n" + "="*50)
print("     HYPERPARAMETER TUNING SUMMARY")
print("="*50)
print(f"  Best Params : {grid_search.best_params_}")
print(f"  CV Score    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Test Acc    : {test_acc:.4f}")
print("="*50)
print("\n✅ Tuned model saved to models/")
print("✅ Results saved to outputs/")
