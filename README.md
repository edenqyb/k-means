# K-Means Clustering

A simple implementation of the **K-Means** clustering algorithm from scratch.

## What is K-Means?

K-Means is an **unsupervised** machine learning algorithm used to group similar data points into a fixed number of clusters (`K`).

It works by repeatedly assigning points to the nearest cluster center (centroid) and then updating those centers until the clusters stabilize.

### How the algorithm works

1. **Choose K**  
   Decide how many clusters you want.

2. **Initialize centroids**  
   Randomly pick `K` points as the starting centroids (or use a smarter method like K-Means++).

3. **Assign points**  
   Assign every data point to the closest centroid (usually using Euclidean distance).

4. **Update centroids**  
   Recalculate each centroid as the mean of all points assigned to it.

5. **Repeat**  
   Keep doing steps 3 and 4 until the centroids stop moving significantly (or a maximum number of iterations is reached).


## Features

- Pure Python / NumPy implementation (no scikit-learn)
- Easy to understand and modify
- Includes visualization of the clustering process
- Supports elbow method to help choose `K`

## Installation

```bash
git clone https://github.com/your-username/kmeans.git
cd kmeans
pip install -r requirements.txt
