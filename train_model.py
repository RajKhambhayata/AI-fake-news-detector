"""
train_model.py
---------------
Run this AFTER prepare_data.py has created dataset.csv.

Usage:
    python train_model.py

Output:
    ../detector/model.pkl
    ../detector/vectorizer.pkl
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH = 'dataset.csv'
OUTPUT_DIR   = '.'  # Root directory for Flask app
MAX_FEATURES = 5000
TEST_SIZE    = 0.2
RANDOM_STATE = 42
# ───────────────────────────────────────────────────────────────────────────────


def main():
    # 1. Load data
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna()
    print(f"  Rows: {len(df)}")

    X = df['content'].values
    Y = df['label'].values

    # 2. TF-IDF Vectorization
    print(f"Fitting TF-IDF vectorizer (max_features={MAX_FEATURES}) ...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    vectorizer.fit(X)
    X_vec = vectorizer.transform(X)

    # 3. Train/Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_vec, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE
    )
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing  samples : {X_test.shape[0]}")

    # 4. Model Training
    print("Training Logistic Regression model ...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    # 5. Evaluation
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_acc  = accuracy_score(Y_test,  model.predict(X_test))
    print(f"\n  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  Accuracy : {test_acc  * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(Y_test, model.predict(X_test), target_names=['Fake', 'Real']))

    # 6. Save model and vectorizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path      = os.path.join(OUTPUT_DIR, 'model.pkl')
    vectorizer_path = os.path.join(OUTPUT_DIR, 'vectorizer.pkl')

    pickle.dump(model,      open(model_path,      'wb'))
    pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

    print(f"\n✅ Model saved      : {model_path}")
    print(f"✅ Vectorizer saved : {vectorizer_path}")


if __name__ == '__main__':
    main()
