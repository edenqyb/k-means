import matplotlib.pyplot as plt
import numpy as np


def plot_inertia(inertias, title="Elbow Method"):
    inertias = np.asarray(inertias)
    ks = np.arange(1, len(inertias) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.title(title)
    plt.xticks(ks)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_clusters(X, labels, centroids, title="K-Means Clusters"):
    X = np.asarray(X)
    labels = np.asarray(labels)
    centroids = np.asarray(centroids)

    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=25, alpha=0.8)
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="X",
        s=160,
        label="Centroids",
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
