import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MaxAbsScaler

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR  = os.path.join(BASE_DIR, 'outputs')

print("📂 Loading feature matrices...")
X_train = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_train.npz'))
X_val   = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_val.npz'))
X_test  = sp.load_npz(os.path.join(FEATURES_DIR, 'tfidf_test.npz'))

y_train = pd.read_csv(os.path.join(FEATURES_DIR, 'y_train.csv'))['class_label'].values
y_val   = pd.read_csv(os.path.join(FEATURES_DIR, 'y_val.csv'))['class_label'].values
y_test  = pd.read_csv(os.path.join(FEATURES_DIR, 'y_test.csv'))['class_label'].values

print(f"   X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

scaler         = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

models = {
    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear_SVM':          LinearSVC(max_iter=1000, random_state=42),
    'Naive_Bayes':         MultinomialNB(),
    'Random_Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient_Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5),
}

scale_models = ['Logistic_Regression', 'Linear_SVM', 'KNN']
results = {}
print("\n🤖 Training 6 classifiers...\n")

for name, model in models.items():
    print(f"   Training {name}...")
    if name in scale_models:
        Xtr, Xvl, Xte = X_train_scaled, X_val_scaled, X_test_scaled
    else:
        Xtr, Xvl, Xte = X_train, X_val, X_test
    model.fit(Xtr, y_train)
    val_preds  = model.predict(Xvl)
    test_preds = model.predict(Xte)
    val_acc  = accuracy_score(y_val,  val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    results[name] = {
        'val_accuracy':  round(val_acc,  4),
        'test_accuracy': round(test_acc, 4),
    }
    print(f"   Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
    joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.pkl'))

best_model_name = max(results, key=lambda x: results[x]['val_accuracy'])
best_val_acc    = results[best_model_name]['val_accuracy']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Val Accuracy: {best_val_acc:.4f}")

best_model = joblib.load(os.path.join(MODELS_DIR, f'{best_model_name}.pkl'))
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(scaler,     os.path.join(MODELS_DIR, 'scaler.pkl'))

results['best_model'] = best_model_name
with open(os.path.join(OUTPUTS_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n📊 Model Comparison:")
print(f"{'Model':<25} {'Val Acc':>10} {'Test Acc':>10}")
print("-" * 47)
for name, res in results.items():
    if name == 'best_model':
        continue
    marker = " <-- BEST" if name == best_model_name else ""
    print(f"{name:<25} {res['val_accuracy']:>10.4f} {res['test_accuracy']:>10.4f}{marker}")

print("\n✅ All models saved to models/")
print("✅ Results saved to outputs/model_results.json")
