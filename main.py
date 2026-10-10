import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Mall_Customers.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. Load and inspect the dataset
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())

    # 2. Select numeric customer attributes
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    missing_columns = [col for col in features if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Expected columns {missing_columns}. Found columns: {list(df.columns)}"
        )

    X = df[features].copy()
    X = X.fillna(X.median(numeric_only=True))

    # 3. Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. PCA to two dimensions for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance retained:", round(pca.explained_variance_ratio_.sum(), 4))

    # 5. Find a suitable K using silhouette score
    scores = {}
    max_k = min(8, len(df) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(X_scaled, labels)

    print("\nSilhouette scores:")
    for k, score in scores.items():
        print(f"K={k}: {score:.4f}")

    if not scores:
        raise ValueError("Not enough data to calculate silhouette scores.")

    best_k = max(scores, key=scores.get)
    print(f"\nBest K based on tested silhouette scores: {best_k}")

    # 6. Final K-Means clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    # 7. DBSCAN clustering (starting parameters; can be tuned)
    dbscan = DBSCAN(eps=0.7, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df["DBSCAN_Cluster"] = dbscan_labels

    dbscan_clusters = set(dbscan_labels) - {-1}
    if len(dbscan_clusters) >= 2:
        mask = dbscan_labels != -1
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        print(f"DBSCAN silhouette (excluding noise): {dbscan_silhouette:.4f}")
    else:
        print("DBSCAN found fewer than two non-noise clusters; silhouette not calculated.")

    # 8. Save PCA/K-Means plot
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df["KMeans_Cluster"], cmap="tab10", s=55, alpha=0.85
    )
    plt.title("Mall Customer Segmentation: K-Means + PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="K-Means cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_pca_clusters.png"), dpi=160)
    plt.close()

    # 9. Save silhouette plot
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("K Selection using Silhouette Score")
    plt.xticks(list(scores.keys()))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_scores.png"), dpi=160)
    plt.close()

    # 10. Profile the customer clusters
    profile = df.groupby("KMeans_Cluster")[features].mean().round(2)
    profile["Customer_Count"] = df.groupby("KMeans_Cluster").size()
    profile.to_csv(os.path.join(OUTPUT_DIR, "mall_cluster_profiles.csv"))

    # 11. Save complete customer data with labels
    df.to_csv(os.path.join(OUTPUT_DIR, "mall_clustered_customers.csv"), index=False)

    print("\nCustomer cluster profiles:\n", profile)
    print("\nSaved files in the outputs folder:")
    print("- kmeans_pca_clusters.png")
    print("- silhouette_scores.png")
    print("- mall_cluster_profiles.csv")
    print("- mall_clustered_customers.csv")
    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
