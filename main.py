import numpy as np
from kmeans import KMeans, plot_clusters, plot_inertia

def make_sample_data(n_per_cluster=100, random_state=42):
    rng = np.random.default_rng(random_state)
    centers = np.array([[0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]])
    clusters = [rng.normal(center, 0.6, size=(n_per_cluster, 2)) for center in centers]
    X = np.vstack(clusters)
    rng.shuffle(X)
    return X

def main():
    X = make_sample_data()

    # elbow plot to help choose K
    model = KMeans(random_state=42)
    inertias = model.elbow(X, max_clusters=10)
    print("Inertias for K=1..10:")
    for k, inertia in enumerate(inertias, start=1):
        print(f"  K={k}: {inertia:.2f}")
    plot_inertia(inertias)

    # fit with chosen K
    k = 4
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    print(f"\nFitted K={k}")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Centroids:\n{model.centroids}")

    plot_clusters(X, model.labels, model.centroids, title=f"K-Means (K={k})")


if __name__ == "__main__":
    main()
