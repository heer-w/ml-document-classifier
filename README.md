# ML Document Classifier - Module 1 Pipeline

A machine learning pipeline that classifies documents into 5 categories:
Invoice, Purchase Order, Resume, Bank Statement, and Email.

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install scikit-learn pandas numpy matplotlib seaborn nltk faker joblib jupyter wordcloud

## Pipeline Steps

### Task 1 - Data Preparation
- Generated 100 synthetic documents using Faker library
- 5 classes: Invoice, Purchase_Order, Resume, Bank_Statement, Email
- 20 documents per class - balanced dataset
- 80/10/10 stratified train/val/test split

### Task 2 - Exploratory Data Analysis
- Class distribution chart, word count box plot, word clouds
- Top 20 TF-IDF terms per class
- Jaccard similarity heatmap
- Key finding: Purchase_Order and Bank_Statement most similar (Jaccard: 0.13)

### Task 3 - Feature Engineering
- Text cleaning: lowercasing, stopword removal, Porter stemming
- TF-IDF vectorizer: 5000 features, bigrams, sublinear TF
- 3 handcrafted features: word count, avg word length, digit ratio
- Final feature matrix: 4343 features per document

### Task 4 - Model Training
- Logistic Regression  Val: 1.00  Test: 1.00  BEST
- Linear SVM           Val: 1.00  Test: 1.00
- Naive Bayes          Val: 0.20  Test: 0.20
- Random Forest        Val: 1.00  Test: 1.00
- Gradient Boosting    Val: 1.00  Test: 1.00
- KNN                  Val: 0.80  Test: 0.80

### Task 5 - Deep Evaluation
- Confusion matrix, classification report, per class accuracy chart

### Task 6 - Hyperparameter Tuning
- GridSearchCV with 5-fold cross validation
- Best parameters: C=0.01, solver=lbfgs, max_iter=500
- CV Score: 1.0000

### Task 7 - Prediction CLI Tool
- python3 src/predict.py data/raw/Invoice/doc_001.txt

### Task 8 - Unsupervised Clustering
- KMeans k=5: ARI=0.7730, Silhouette=0.0774
- DBSCAN: found 6 clusters, 28 noise points

## Key Findings
- Logistic Regression, SVM, Random Forest, Gradient Boosting all achieved 100% accuracy
- Naive Bayes failed (20%) due to incompatibility with scaled features
- KNN achieved 80% - weaker than linear models
- KMeans ARI of 0.77 confirms classes are naturally well separated
- 100% accuracy expected with synthetic data - real world would be 85-95%

## How to Run Full Pipeline
- python3 src/generate_data.py
- python3 src/create_splits.py
- python3 src/feature_engineering.py
- python3 src/train_models.py
- python3 src/evaluate.py
- python3 src/hyperparameter_tuning.py
- python3 src/clustering.py

## Author
Heer Wadhwani
Internship: Amicus Technology
