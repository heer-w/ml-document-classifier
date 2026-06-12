import os
import sys
import string
import joblib
import numpy as np
import scipy.sparse as sp
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords', quiet=True)

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

# Load model, scaler and vectorizer
model      = joblib.load(os.path.join(MODELS_DIR,   'best_model_tuned.pkl'))
scaler     = joblib.load(os.path.join(MODELS_DIR,   'scaler_tuned.pkl'))
vectorizer = joblib.load(os.path.join(FEATURES_DIR, 'tfidf_vectoriser.pkl'))

# Text cleaning
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    text   = ''.join([ch for ch in text if not ch.isdigit()])
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_document(filepath):
    # Check file exists
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    # Read file
    with open(filepath, 'r') as f:
        text = f.read()

    if not text.strip():
        print("❌ File is empty!")
        sys.exit(1)

    # Clean and vectorize
    cleaned    = clean_text(text)
    tfidf_vec  = vectorizer.transform([cleaned])

    # Add handcrafted features
    word_count     = len(text.split())
    avg_word_len   = np.mean([len(w) for w in text.split()]) if text.split() else 0
    digit_ratio    = sum(c.isdigit() for c in text) / len(text) if text else 0
    hc_features    = np.array([[word_count, avg_word_len, digit_ratio]])

    # Combine
    X = sp.hstack([tfidf_vec, sp.csr_matrix(hc_features)])
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    # Confidence using decision function
    decision   = model.decision_function(X_scaled)[0]
    exp_scores = np.exp(decision - np.max(decision))
    confidence = exp_scores / exp_scores.sum()
    max_conf   = confidence.max() * 100

    # Print results
    print("\n" + "="*45)
    print("       DOCUMENT CLASSIFIER RESULT")
    print("="*45)
    print(f"  File       : {os.path.basename(filepath)}")
    print(f"  Prediction : {prediction}")
    print(f"  Confidence : {max_conf:.1f}%")
    print("="*45)

    # Show all class probabilities
    classes = model.classes_
    print("\n  All Class Scores:")
    for cls, conf in sorted(zip(classes, confidence), key=lambda x: x[1], reverse=True):
        bar = '█' * int(conf * 20)
        print(f"  {cls:<20} {conf*100:5.1f}% {bar}")
    print()

    return prediction

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py <path_to_document>")
        print("Example: python3 src/predict.py data/raw/Invoice/doc_001.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    predict_document(filepath)
