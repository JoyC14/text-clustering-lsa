import csv
import sys
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def load_data(file_path):
    """
    Load the dataset from a tab-separated text file.
    Expected columns: ID, Sentence
    """
    try:
        df = pd.read_csv(file_path, sep='\t', quoting=csv.QUOTE_NONE)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def clean_text(text):
    """
    Basic text cleaning:
    - convert to lowercase
    - remove extra spaces
    - remove newline characters
    """
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_and_vectorize(file_path):
    """
    Load data, clean text, and convert cleaned sentences into TF-IDF features.
    """
    # 1. Load dataset
    df = load_data(file_path)
    if df is None:
      sys.exit(1)

    # 2. Check required column
    text_column = 'Sentence'
    if text_column not in df.columns:
        text_column = df.columns[1]

    # 3. Clean text
    df['cleaned_text'] = df[text_column].apply(clean_text)

    # 4. Convert cleaned sentences into TF-IDF features.
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.90,
        min_df=2,
        max_features=1000
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    # 5. Apply TruncatedSVD for dimensionality reduction (LSA)
    # Compress the 1000 dimensions into 50 core semantic features
    n_components = 50
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    
    # Calculate the amount of original information retained by these 50 dimensions (Explained Variance)
    explained_variance = svd.explained_variance_ratio_.sum()
    
    print(f"\n--- Feature Engineering Results ---")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Reduced matrix shape: {reduced_matrix.shape}")
    print(f"Top {n_components} dimensions explain {explained_variance * 100:.2f}% of the variance\n")

    return reduced_matrix, df, tfidf_vectorizer

if __name__ == "__main__":
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        file_path = sys.argv[1]
    else:
        file_path = "data_train.txt"

    matrix, dataset, vectorizer = preprocess_and_vectorize(file_path)