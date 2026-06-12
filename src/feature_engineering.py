import os
import json
import string
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
SPLITS_DIR  = os.path.join(BASE_DIR, 'data', 'splits')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

# ── Step 1: Text Cleaning ──────────────────────────────────────
stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([ch for ch in text if not ch.isdigit()])
    # Tokenize
    tokens = text.split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── Step 2: Load splits ────────────────────────────────────────
def load_split(split_name):
    df = pd.read_csv(os.path.join(SPLITS_DIR, f'{split_name}.csv'))
    texts = []
    for filepath in df['file_path']:
        full_path = os.path.join(BASE_DIR, filepath)
        with open(full_path, 'r') as f:
            texts.append(f.read())
    df['text']         = texts
    df['clean_text']   = df['text'].apply(clean_text)
    return df

print("📂 Loading splits...")
train_df = load_split('train')
val_df   = load_split('val')
test_df  = load_split('test')
print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Step 3: TF-IDF Vectorizer ──────────────────────────────────
print("\n🔤 Fitting TF-IDF vectorizer on train split only...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),       # unigrams and bigrams
    sublinear_tf=True         # apply log normalization
)

# Fit ONLY on train — transform all three
X_train_tfidf = vectorizer.fit_transform(train_df['clean_text'])
X_val_tfidf   = vectorizer.transform(val_df['clean_text'])
X_test_tfidf  = vectorizer.transform(test_df['clean_text'])

print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

# ── Step 4: Handcrafted Features ───────────────────────────────
def handcrafted_features(df):
    features = pd.DataFrame()
    # word_count: longer docs may be resumes or bank statements
    features['word_count'] = df['text'].apply(lambda x: len(x.split()))
    # avg_word_length: financial docs have longer technical words
    features['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    # digit_ratio: invoices/bank statements have lots of numbers
    features['digit_ratio'] = df['text'].apply(
        lambda x: sum(c.isdigit() for c in x) / len(x) if len(x) > 0 else 0
    )
    return features.values

print("\n🔧 Engineering handcrafted features...")
hc_train = handcrafted_features(train_df)
hc_val   = handcrafted_features(val_df)
hc_test  = handcrafted_features(test_df)

# ── Step 5: Combine TF-IDF + Handcrafted ──────────────────────
X_train = sp.hstack([X_train_tfidf, sp.csr_matrix(hc_train)])
X_val   = sp.hstack([X_val_tfidf,   sp.csr_matrix(hc_val)])
X_test  = sp.hstack([X_test_tfidf,  sp.csr_matrix(hc_test)])

print(f"\n📐 Final feature matrix shapes:")
print(f"   X_train : {X_train.shape}")
print(f"   X_val   : {X_val.shape}")
print(f"   X_test  : {X_test.shape}")

# ── Step 6: Save Everything ────────────────────────────────────
print("\n💾 Saving feature matrices...")

sp.save_npz(os.path.join(FEATURES_DIR, 'tfidf_train.npz'), X_train)
sp.save_npz(os.path.join(FEATURES_DIR, 'tfidf_val.npz'),   X_val)
sp.save_npz(os.path.join(FEATURES_DIR, 'tfidf_test.npz'),  X_test)

import joblib
joblib.dump(vectorizer, os.path.join(FEATURES_DIR, 'tfidf_vectoriser.pkl'))

# Save labels
train_df['class_label'].to_csv(os.path.join(FEATURES_DIR, 'y_train.csv'), index=False)
val_df['class_label'].to_csv(os.path.join(FEATURES_DIR,   'y_val.csv'),   index=False)
test_df['class_label'].to_csv(os.path.join(FEATURES_DIR,  'y_test.csv'),  index=False)

# Save metadata
meta = {
    'n_features':   X_train.shape[1],
    'vocab_size':   len(vectorizer.vocabulary_),
    'ngram_range':  [1, 2],
    'class_names':  list(train_df['class_label'].unique()),
    'train_size':   len(train_df),
    'val_size':     len(val_df),
    'test_size':    len(test_df)
}
with open(os.path.join(FEATURES_DIR, 'feature_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✅ All files saved to features/")
print(f"\n📋 Feature Metadata:")
print(json.dumps(meta, indent=2))