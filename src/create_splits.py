import os
import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
RAW_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
SPLITS_DIR = os.path.join(BASE_DIR, 'data', 'splits')

classes = ['Invoice', 'Purchase_Order', 'Resume', 'Bank_Statement', 'Email']

# ── Step 1: Build master list ──────────────────────────────────
all_docs = []

for class_name in classes:
    folder = os.path.join(RAW_DIR, class_name)
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.txt'):
            filepath = os.path.join('data', 'raw', class_name, filename)

            # Read file and count words
            with open(os.path.join(BASE_DIR, filepath), 'r') as f:
                text = f.read()
            word_count = len(text.split())

            all_docs.append({
                'file_path':   filepath,
                'class_label': class_name,
                'word_count':  word_count,
                'source':      'faker_generated'
            })

# Save all_docs.csv
df = pd.DataFrame(all_docs)
df.to_csv(os.path.join(SPLITS_DIR, 'all_docs.csv'), index=False)
print(f"✅ all_docs.csv saved — {len(df)} documents")
print("\nClass distribution:")
print(df['class_label'].value_counts())

# ── Step 2: Train / Val / Test Split (80/10/10) ────────────────
# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['class_label']
)

# Second split: split the 20% into 10% val and 10% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['class_label']
)

# Save splits
train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(SPLITS_DIR,   'val.csv'),   index=False)
test_df.to_csv(os.path.join(SPLITS_DIR,  'test.csv'),  index=False)

print(f"\n✅ Splits saved!")
print(f"   Train : {len(train_df)} docs")
print(f"   Val   : {len(val_df)} docs")
print(f"   Test  : {len(test_df)} docs")