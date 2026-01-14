# Boltzmann-Shannon Index (BSI)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust and information-theoretic metric for evaluating clustering quality. The Boltzmann-Shannon Index (BSI) provides a principled approach to assess cluster validity by comparing frequency-based and volume-based probability distributions using information theory.

## Overview

The Boltzmann-Shannon Index is a clustering quality metric that measures the similarity between two probability distributions:
- **p**: Frequency-based distribution (proportional to cluster sizes)
- **q**: Volume-based distribution (derived from Singular Value Decomposition of cluster data)

The index ranges from 0 to 1, where **higher values indicate better clustering quality**, including better separation, balance, and overall cluster structure.

## Key Features

- **Information-theoretic foundation**: Based on the Jensen-Shannon divergence
- **Dimension-agnostic**: Works with data of any dimensionality
- **Handles imbalanced clusters**: Robust to varying cluster sizes
- **Multiple implementations**: Available in both MATLAB and Python
- **Easy to use**: Simple function interface with minimal dependencies

## Installation

### Python

The Python implementation requires:
- `numpy` (>= 1.19.0)
- `scipy` (>= 1.5.0)

Install dependencies:
```bash
pip install numpy scipy
```

Then simply copy `Deployment Ready - Python/BoltzmannShannonIndex.py` to your project directory.

### MATLAB

The MATLAB implementation requires the Statistics and Machine Learning Toolbox (for `unique` function with the `'rows'` option in older versions). Simply add the `Deployment Ready - MATLAB` folder to your MATLAB path.

## Quick Start

### Python

```python
import numpy as np
from BoltzmannShannonIndex import BoltzmannShannonIndex

# Generate sample data with 3 clusters
data = np.vstack([
    np.random.randn(100, 2) + np.array([2, 2]),  # Cluster 0
    np.random.randn(100, 2) + np.array([6, 6]),  # Cluster 1
    np.random.randn(100, 2) + np.array([4, 0])   # Cluster 2
])

# Cluster labels
labels = np.array([0]*100 + [1]*100 + [2]*100)

# Calculate BSI
bsi_value = BoltzmannShannonIndex(data, labels)
print(f"BSI: {bsi_value:.4f}")  # Higher values indicate better clustering
```

### MATLAB

```matlab
% Generate sample data
X = [randn(100,2) + [2 2]; randn(100,2) + [6 6]; randn(100,2) + [4 0]];
L = [ones(100,1); 2*ones(100,1); 3*ones(100,1)];

% Calculate BSI
BSI = BoltzmannShannonIndex(X, L);
fprintf('BSI: %.4f\n', BSI);
```

## Usage Examples

### Python

See `Deployment Ready - Python/example_usage.py` for comprehensive examples including:
- Simple multi-cluster scenarios
- String labels
- Imbalanced clusters
- Higher-dimensional data
- Integration with clustering algorithms

### MATLAB

The `Paper Reproduction` folder contains extensive examples and scripts for:
- Gaussian mixture models
- Iris dataset benchmarking
- Resource allocation analysis
- Figure generation

## Methodology

The BSI computation involves three main steps:

1. **Frequency Probability (p)**: Computes the distribution based on cluster sizes
   ```python
   p = cluster_sizes / total_samples
   ```

2. **Volume Probability (q)**: Uses SVD to compute volume-based probabilities
   - Performs SVD on centered cluster data
   - Computes singular values for each cluster
   - Determines volume products and normalizes to probabilities

3. **Index Calculation**: Computes the information-theoretic divergence
   ```python
   BSI = 1 - 0.5 * (KL(p||m) + KL(q||m))
   ```
   where `m = 0.5*(p+q)` is the midpoint distribution and KL is the Kullback-Leibler divergence.

## Interpretation

- **BSI ≈ 1.0**: Excellent clustering with well-separated, balanced clusters
- **BSI ≈ 0.5-0.8**: Good clustering quality
- **BSI < 0.5**: Poor clustering with overlapping or highly imbalanced clusters
- **BSI = 0.0**: Degenerate case (all clusters are empty or identical)

## Project Structure

```
Boltzmann_Shannon_Index/
│
├── Deployment Ready - MATLAB/
│   └── BoltzmannShannonIndex.m      # MATLAB implementation
│
├── Deployment Ready - Python/
│   ├── BoltzmannShannonIndex.py     # Python implementation
│   └── example_usage.py             # Usage examples
│
├── Paper Reproduction/
│   ├── ClustersBSIE.m               # Core computation function
│   ├── Table01_Iris.m               # Iris dataset benchmarking
│   ├── Fig03_GaussianMixture.m      # Gaussian mixture examples
│   ├── Fig04_ResourceAlloc.m        # Resource allocation analysis
│   └── ...                          # Additional reproduction scripts
│
├── BSI.pdf                          # Research paper
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Citation

If you use the Boltzmann-Shannon Index in your research, please cite the associated paper:

```bibtex
@article{almomani2025boltzmann,
  title={Boltzmann-Shannon Index: An Information-Theoretic Approach to Clustering Quality},
  author={AlMomani, Abd AlRahman Rasheed},
  year={2025},
  journal={[Journal/Conference Name]}
}
```

Please refer to `BSI.pdf` for the complete citation details.

## Requirements

### Python
- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0

### MATLAB
- MATLAB R2019b or later (for `unique` with `'rows'` option)
- Statistics and Machine Learning Toolbox (for older MATLAB versions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Abd AlRahman Rasheed AlMomani**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## References

- Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal.
- Kullback, S., & Leibler, R. A. (1951). On information and sufficiency. The Annals of Mathematical Statistics.
- See `BSI.pdf` for the complete theoretical framework and experimental validation.

## Acknowledgments

This implementation is based on the research presented in the associated paper. The code has been designed for both research and practical applications in cluster analysis and validation.

---

For questions, issues, or contributions, please open an issue on GitHub.
