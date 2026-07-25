# K-Means Clustering

A simple implementation of the **K-Means** clustering algorithm from scratch using NumPy.

## What is K-Means?

K-Means is an **unsupervised** machine learning algorithm used to group similar data points into a fixed number of clusters (`K`).

It works by repeatedly assigning points to the nearest cluster center (centroid) and then updating those centers until the clusters stabilize.

### How the algorithm works

1. **Choose K**  
   Decide how many clusters you want.

2. **Initialize centroids**  
   Randomly pick `K` data points as the starting centroids.

3. **Assign points**  
   Assign every data point to the closest centroid (Euclidean distance).

4. **Update centroids**  
   Recalculate each centroid as the mean of all points assigned to it.

5. **Repeat**  
   Keep doing steps 3 and 4 until the centroids stop moving significantly (or a maximum number of iterations is reached).

## Features

- Pure Python / NumPy implementation (no scikit-learn)
- `fit` / `predict` API
- Elbow method via SSE to help choose `K`
- Matplotlib plots for the elbow curve and final clusters
- Reproducible runs with `random_state`

## Project structure

```
k-means/
├── main.py              # demo with synthetic 2D data
├── requirements.txt
├── README.md
└── kmeans/
    ├── __init__.py
    ├── model.py         # KMeans class (fit, predict, elbow)
    └── viz.py           # plot_SSE, plot_clusters
```

## Installation

```bash
git clone https://github.com/edenqyb/k-means.git
cd k-means
pip install -r requirements.txt
```

## Usage

Run the demo (data → elbow plot → fit → cluster plot):

```bash
python main.py
```

The demo:

1. Generates 4 Gaussian clusters in 2D
2. Computes SSE for `K = 1..10` and plots the elbow curve
3. Fits K-Means with `K = 4`
4. Plots the clustered points and centroids

### Quick example

```python
import numpy as np
from kmeans import KMeans, plot_clusters, plot_SSE

X = np.random.randn(300, 2)

# choose K with the elbow method
model = KMeans(random_state=42)
SSE = model.elbow(X, max_clusters=10)
plot_SSE(SSE)

# fit and visualize
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
print(model.SSE_)
print(model.centroids)

plot_clusters(X, model.labels, model.centroids)
```

### `KMeans` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clusters` | `3` | Number of clusters (`K`) |
| `max_iter` | `100` | Maximum assign/update iterations |
| `tol` | `1e-4` | Stop early when max centroid shift is below this |
| `random_state` | `None` | Seed for reproducible centroid initialization |

After `fit`, the model stores `centroids`, `labels`, and `SSE_`.
