"""
prepare_data.py
----------------
Run this script FIRST to merge and preprocess the Kaggle dataset.

Usage:
    Place Fake.csv and True.csv in the same directory as this script,
    then run: python prepare_data.py
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

port_stem = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))


def preprocess_text(content: str) -> str:
    """Lowercase, remove punctuation/numbers, remove stopwords, apply stemming."""
    # Remove non-alphabetical characters
    text = re.sub('[^a-zA-Z]', ' ', str(content))
    # Lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords and stem remaining words
    words = [port_stem.stem(word) for word in words if word not in STOP_WORDS]
    return ' '.join(words)


def main():
    print("Loading Fake.csv ...")
    fake = pd.read_csv('Fake.csv')
    fake['label'] = 0  # 0 = Fake

    print("Loading True.csv ...")
    real = pd.read_csv('True.csv')
    real['label'] = 1  # 1 = Real

    print("Merging datasets ...")
    df = pd.concat([fake, real], axis=0).reset_index(drop=True)

    # Fill any NaN values
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')

    # Combine title + text for more signal
    df['content'] = df['title'] + ' ' + df['text']
    df = df[['content', 'label']]

    print(f"Total rows: {len(df)}  |  Fake: {(df.label==0).sum()}  |  Real: {(df.label==1).sum()}")

    print("Preprocessing text (this may take a few minutes) ...")
    df['content'] = df['content'].apply(preprocess_text)

    output_path = 'dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Preprocessed dataset saved to: {output_path}")


if __name__ == '__main__':
    main()
