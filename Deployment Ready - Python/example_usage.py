"""
Example usage of BoltzmannShannonIndex function.

This script demonstrates how to use the BoltzmannShannonIndex function
to evaluate clustering quality.
"""

import numpy as np
from BoltzmannShannonIndex import BoltzmannShannonIndex

# Example 1: Simple 2D data with 3 clusters
print("=" * 60)
print("Example 1: Simple 2D data with 3 clusters")
print("=" * 60)

# Generate synthetic data with 3 well-separated clusters
np.random.seed(42)
n_samples_per_cluster = 100

# Cluster 1: centered at (2, 2)
cluster1 = np.random.randn(n_samples_per_cluster, 2) + np.array([2, 2])

# Cluster 2: centered at (6, 6)
cluster2 = np.random.randn(n_samples_per_cluster, 2) + np.array([6, 6])

# Cluster 3: centered at (4, 0)
cluster3 = np.random.randn(n_samples_per_cluster, 2) + np.array([4, 0])

# Combine data
Data = np.vstack([cluster1, cluster2, cluster3])

# Create labels: 0 for cluster1, 1 for cluster2, 2 for cluster3
Label = np.array([0] * n_samples_per_cluster + 
                 [1] * n_samples_per_cluster + 
                 [2] * n_samples_per_cluster)

# Calculate BSI
BSI = BoltzmannShannonIndex(Data, Label)
print(f"Data shape: {Data.shape}")
print(f"Number of clusters: {len(np.unique(Label))}")
print(f"Boltzmann-Shannon Index (BSI): {BSI:.4f}")
print(f"(Higher values indicate better clustering quality)")
print()


# Example 2: Using different label formats (strings)
print("=" * 60)
print("Example 2: Using string labels")
print("=" * 60)

# Same data as above, but with string labels
Label_str = np.array(['A'] * n_samples_per_cluster + 
                     ['B'] * n_samples_per_cluster + 
                     ['C'] * n_samples_per_cluster)

BSI2 = BoltzmannShannonIndex(Data, Label_str)
print(f"BSI with string labels: {BSI2:.4f}")
print()


# Example 3: Imbalanced clusters
print("=" * 60)
print("Example 3: Imbalanced clusters")
print("=" * 60)

# Create imbalanced data (different cluster sizes)
cluster1_imbalanced = np.random.randn(400, 2) + np.array([2, 2])
cluster2_imbalanced = np.random.randn(100, 2) + np.array([6, 6])
cluster3_imbalanced = np.random.randn(100, 2) + np.array([4, 0])

Data_imbalanced = np.vstack([cluster1_imbalanced, cluster2_imbalanced, cluster3_imbalanced])
Label_imbalanced = np.array([0] * 400 + [1] * 100 + [2] * 100)

BSI_imbalanced = BoltzmannShannonIndex(Data_imbalanced, Label_imbalanced)
print(f"Data shape: {Data_imbalanced.shape}")
print(f"Cluster sizes: {np.bincount(Label_imbalanced)}")
print(f"BSI for imbalanced clusters: {BSI_imbalanced:.4f}")
print()


# Example 4: Higher dimensional data (e.g., 4D like Iris dataset)
print("=" * 60)
print("Example 4: Higher dimensional data (4D)")
print("=" * 60)

# Generate 4D data (simulating something like the Iris dataset)
np.random.seed(123)
Data_4d = np.random.randn(150, 4)  # 150 samples, 4 features
# Create 3 clusters
Label_4d = np.array([0] * 50 + [1] * 50 + [2] * 50)

BSI_4d = BoltzmannShannonIndex(Data_4d, Label_4d)
print(f"Data shape: {Data_4d.shape}")
print(f"BSI for 4D data: {BSI_4d:.4f}")
print()


# Example 5: Using with clustering results (e.g., after k-means)
print("=" * 60)
print("Example 5: Using with clustering results")
print("=" * 60)

# Generate data
np.random.seed(42)
test_data = np.vstack([
    np.random.randn(50, 2) + np.array([2, 2]),
    np.random.randn(50, 2) + np.array([5, 5]),
    np.random.randn(50, 2) + np.array([2, 5])
])

# Simulate clustering labels (in practice, you'd use sklearn.cluster.KMeans)
# For demonstration, we'll use ground truth labels
test_labels = np.array([0] * 50 + [1] * 50 + [2] * 50)

BSI_test = BoltzmannShannonIndex(test_data, test_labels)
print(f"Test data shape: {test_data.shape}")
print(f"Number of clusters: {len(np.unique(test_labels))}")
print(f"BSI value: {BSI_test:.4f}")
print()

print("=" * 60)
print("Usage Notes:")
print("=" * 60)
print("1. The function accepts numpy arrays or array-like objects")
print("2. Labels can be integers, strings, or any hashable type")
print("3. The BSI value ranges from 0 to 1 (typically)")
print("4. Higher BSI values indicate better clustering quality")
print("5. The function automatically normalizes labels to consecutive integers")
print("=" * 60)
