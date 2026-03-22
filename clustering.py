import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import preprocess_and_vectorize

def find_best_k(X, k_range=range(1, 11), random_state=42):
    """
    Try different values of K and compute silhouette scores.
    Returns:
        best_k: K with the highest silhouette score
        scores: list of silhouette scores
    """
    scores = []
    best_k = None
    best_score = -1

    for k in k_range:
        if k == 1:
            # K=1 cannot be calculated, so 0 are manually added.
            scores.append(0.0)
            print(f"K = {k}, Silhouette Score = 0.0000 (Not defined)")
            continue

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)
        scores.append(score)

        print(f"K = {k}, Silhouette Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores

def plot_silhouette_scores(k_range, scores, output_file="silhouette_plot.png"):
    """
    Plot silhouette scores for different K values.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), scores, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for K-Means Clustering")
    plt.xticks(list(k_range))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Silhouette plot saved as: {output_file}")

def generate_labels(X, k, output_file="label.txt", random_state=42):
    """
    Fit KMeans with the selected K and save predicted labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    labels = labels + 1

    np.savetxt(output_file, labels, fmt="%d")
    print(f"Cluster labels saved to: {output_file}")

    return labels

def main():
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        file_path = sys.argv[1]
    else:
        file_path = "data_train.txt"

    # 1. Preprocess and vectorize
    X, df, vectorizer = preprocess_and_vectorize(file_path)

    # 2. Try K from 1 to 10
    k_range = range(1, 11)
    best_k, scores = find_best_k(X, k_range=k_range)

    print(f"\nBest K based on silhouette score: {best_k}")

    # 3. Save silhouette plot
    plot_silhouette_scores(k_range, scores, output_file="silhouette_plot.png")

    # 4. Generate final labels
    generate_labels(X, best_k, output_file="label.txt")

if __name__ == "__main__":
    main()