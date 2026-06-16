
import os, joblib, numpy as np, scipy.sparse as sp, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords', quiet=True)

BASE_DIR     = '.'
FEATURES_DIR = 'features'
MODELS_DIR   = 'models'

vectorizer = joblib.load(os.path.join(FEATURES_DIR, 'tfidf_vectoriser.pkl'))
scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

models = {
    'Logistic_Regression': joblib.load(os.path.join(MODELS_DIR, 'Logistic_Regression.pkl')),
    'Linear_SVM':          joblib.load(os.path.join(MODELS_DIR, 'Linear_SVM.pkl')),
    'Naive_Bayes':         joblib.load(os.path.join(MODELS_DIR, 'Naive_Bayes.pkl')),
    'Random_Forest':       joblib.load(os.path.join(MODELS_DIR, 'Random_Forest.pkl')),
    'Gradient_Boosting':   joblib.load(os.path.join(MODELS_DIR, 'Gradient_Boosting.pkl')),
    'KNN':                 joblib.load(os.path.join(MODELS_DIR, 'KNN.pkl')),
}

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text   = text.lower()
    text   = text.translate(str.maketrans('', '', string.punctuation))
    text   = ''.join([ch for ch in text if not ch.isdigit()])
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def predict_all_models(filepath, true_label):
    with open(filepath, 'r') as f:
        text = f.read()
    cleaned      = clean_text(text)
    tfidf_vec    = vectorizer.transform([cleaned])
    word_count   = len(text.split())
    avg_word_len = np.mean([len(w) for w in text.split()]) if text.split() else 0
    digit_ratio  = sum(c.isdigit() for c in text) / len(text) if text else 0
    hc           = np.array([[word_count, avg_word_len, digit_ratio]])
    X            = sp.hstack([tfidf_vec, sp.csr_matrix(hc)])
    X_scaled     = scaler.transform(X)
    results = {}
    for name, model in models.items():
        pred    = model.predict(X_scaled)[0]
        correct = 'YES' if pred == true_label else 'NO'
        results[name] = {'prediction': pred, 'correct': correct}
    return results

samples = [
    ('data/raw/Invoice/doc_001.txt',        'Invoice'),
    ('data/raw/Invoice/doc_005.txt',        'Invoice'),
    ('data/raw/Purchase_Order/doc_002.txt', 'Purchase_Order'),
    ('data/raw/Purchase_Order/doc_007.txt', 'Purchase_Order'),
    ('data/raw/Resume/doc_003.txt',         'Resume'),
    ('data/raw/Resume/doc_010.txt',         'Resume'),
    ('data/raw/Bank_Statement/doc_004.txt', 'Bank_Statement'),
    ('data/raw/Bank_Statement/doc_008.txt', 'Bank_Statement'),
    ('data/raw/Email/doc_005.txt',          'Email'),
    ('data/raw/Email/doc_009.txt',          'Email'),
]

correct_count = {name: 0 for name in models}

print('=' * 70)
print('SAMPLE DOCUMENT TESTING - ALL 6 MODELS')
print('=' * 70)

for filepath, true_label in samples:
    results  = predict_all_models(filepath, true_label)
    filename = os.path.basename(filepath)
    folder   = os.path.basename(os.path.dirname(filepath))
    print('')
    print('File: ' + folder + '/' + filename + '  |  True Label: ' + true_label)
    print('Model                     Prediction           Correct?')
    print('-' * 55)
    for name, res in results.items():
        print(name.ljust(25) + ' ' + res['prediction'].ljust(20) + ' ' + res['correct'])
        if res['correct'] == 'YES':
            correct_count[name] += 1

print('')
print('=' * 70)
print('FINAL MODEL COMPARISON')
print('=' * 70)
print('Model                     Correct    Accuracy')
print('-' * 50)
sorted_models = sorted(correct_count.items(), key=lambda x: x[1], reverse=True)
for rank, (name, correct) in enumerate(sorted_models, 1):
    accuracy = correct / len(samples) * 100
    star = ' <-- BEST' if rank == 1 else ''
    print(name.ljust(25) + ' ' + str(correct).rjust(5) + '/10  ' + str(round(accuracy, 1)).rjust(7) + '%' + star)
print('')
print('Best Model: ' + sorted_models[0][0])
print('=' * 70)
