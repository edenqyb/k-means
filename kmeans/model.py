import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.SSE_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array")
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        # initialize centroids
        index = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[index].copy()

        # iterate until convergence
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # update centroids
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)
                else:
                    new_centroids[k] = X[rng.integers(0, n_samples)]

            centroids = new_centroids

            # check for convergence before overwriting
            if np.linalg.norm(new_centroids - centroids, axis=1).max() < self.tol:
                break

        self.centroids = centroids
        self.labels = labels
        # sum of squared distances to assigned centroid
        self.SSE_ = ((X - centroids[labels]) ** 2).sum()
        return self

    # elbow method to find the optimal number of clusters
    def elbow(self, X, max_clusters=10):
        X = np.asarray(X, dtype=float)
        SSE = np.empty(max_clusters)
        for k in range(1, max_clusters + 1):
            model = KMeans(
                n_clusters=k,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )
            model.fit(X)
            SSE[k - 1] = model.SSE_
        return SSE

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D array")
        if self.centroids is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


