import numpy as np
from scipy.linalg import svd


def BoltzmannShannonIndex(Data, Label, *varargin):
    """
    Calculate the Boltzmann-Shannon Index (BSI) for cluster analysis.
    
    Parameters:
    -----------
    Data : array-like
        Input data matrix (n_samples, n_features)
    Label : array-like
        Cluster labels (n_samples,)
    *varargin : additional arguments (currently unused)
    
    Returns:
    --------
    BSI : float
        Boltzmann-Shannon Index value
    """
    BSI = ClustersBSIE(Data, Label)
    return BSI


def ClustersBSIE(X, L):
    """
    Compute BSI for clusters.
    
    Parameters:
    -----------
    X : array-like
        Data matrix (n_samples, n_features)
    L : array-like
        Cluster labels (n_samples,)
    
    Returns:
    --------
    E : float
        BSI value
    """
    # Convert labels to unique integer indices
    _, L = np.unique(L, return_inverse=True)
    
    p, _ = getFreqProb(L)
    q, _ = getSVDProb(X, L)
    
    E = distributions_bsi(p, q)
    return E


def getFreqProb(Label):
    """
    Get frequency probabilities from labels.
    
    Parameters:
    -----------
    Label : array-like
        Cluster labels (n_samples,)
    
    Returns:
    --------
    p : ndarray
        Probability distribution based on label frequencies
    H : ndarray
        Frequency counts
    """
    _, ic = np.unique(Label, return_inverse=True)
    H = np.bincount(ic)
    p = H / np.sum(H)
    return p, H


def getSVDProb(Data, Label):
    """
    Get probabilities based on SVD analysis of clusters.
    
    Parameters:
    -----------
    Data : array-like
        Data matrix (n_samples, n_features)
    Label : array-like
        Cluster labels (n_samples,)
    
    Returns:
    --------
    q : ndarray
        Probability distribution based on SVD
    V : ndarray
        Volume products
    """
    S = []
    for i in range(np.max(Label) + 1):
        ix = (Label == i)
        x = Data[ix, :]
        if np.sum(ix) == 0:
            continue
        # Center the data
        x_centered = x - np.mean(x, axis=0)
        # SVD with economy mode (full_matrices=False)
        # In Python, svd returns singular values as a 1D array (unlike MATLAB's diagonal matrix)
        _, s, _ = svd(x_centered, full_matrices=False)
        S.append(s)
    
    if len(S) == 0:
        raise ValueError("No data points found for any cluster")
    
    S = np.vstack(S)
    
    # Cumulative sum along columns (axis=1)
    W = np.cumsum(S**2, axis=1) / np.sum(S**2, axis=1, keepdims=True)
    B = (W > 0.975)
    
    for i in range(B.shape[1]):
        if np.all(B[:, i]):
            S = S[:, :i+1]
            break
    
    # Product along rows (axis=1)
    V = np.prod(S, axis=1)
    q = V / np.sum(V)
    return q, V


def distributions_bsi(p, q):
    """
    Compute BSI from two probability distributions.
    
    Parameters:
    -----------
    p : array-like
        First probability distribution
    q : array-like
        Second probability distribution
    
    Returns:
    --------
    E : float
        BSI value
    """
    p = np.array(p)
    q = np.array(q)
    
    # Find indices where either p or q is zero
    ix = np.logical_or(p == 0, q == 0)
    
    if np.all(ix):
        return 0.0
    
    # Remove zero entries
    p = p[~ix]
    q = q[~ix]
    
    m = 0.5 * (p + q)
    
    E = 1 - 0.5 * (np.dot(p, np.log2(p / m)) + np.dot(q, np.log2(q / m)))
    return E