"""
Statistical metrics for drift detection.

This module provides various statistical metrics for measuring
distribution differences.
"""

import numpy as np
from typing import Union, Tuple
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance as scipy_wasserstein


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-10
) -> float:
    """Calculate Population Stability Index (PSI).
    
    PSI measures the shift in distributions between two samples.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference distribution.
    current : np.ndarray
        Current distribution to compare.
    n_bins : int
        Number of bins for discretization.
    eps : float
        Small value to avoid log(0).
        
    Returns
    -------
    float
        PSI value. < 0.1 (no shift), 0.1-0.25 (small shift), > 0.25 (large shift).
    """
    # Create bins based on reference data
    if len(reference.shape) > 1:
        reference = reference[:, 0]
    if len(current.shape) > 1:
        current = current[:, 0]
    
    # Use quantiles for binning
    bins = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    
    # Calculate frequencies
    ref_freq, _ = np.histogram(reference, bins=bins)
    curr_freq, _ = np.histogram(current, bins=bins)
    
    # Normalize to probabilities
    ref_prob = (ref_freq + eps) / np.sum(ref_freq + eps)
    curr_prob = (curr_freq + eps) / np.sum(curr_freq + eps)
    
    # Calculate PSI
    psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
    
    return psi


def calculate_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """Calculate Kullback-Leibler divergence.
    
    KL divergence measures how one probability distribution diverges
    from a reference distribution.
    
    Parameters
    ----------
    p : np.ndarray
        Reference distribution (must sum to 1).
    q : np.ndarray
        Comparison distribution (must sum to 1).
    eps : float
        Small value to avoid log(0).
        
    Returns
    -------
    float
        KL divergence value (always >= 0).
    """
    # Ensure distributions are normalized
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Add epsilon to avoid log(0)
    p = p + eps
    q = q + eps
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return entropy(p, q)


def calculate_jensen_shannon_divergence(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """Calculate Jensen-Shannon divergence.
    
    JS divergence is a symmetric version of KL divergence.
    
    Parameters
    ----------
    p : np.ndarray
        First distribution.
    q : np.ndarray
        Second distribution.
    eps : float
        Small value for numerical stability.
        
    Returns
    -------
    float
        JS divergence value (between 0 and 1).
    """
    # Ensure distributions are normalized
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Add epsilon
    p = p + eps
    q = q + eps
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # JS divergence is the square of JS distance
    js_distance = jensenshannon(p, q)
    return js_distance ** 2


def calculate_wasserstein_distance(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """Calculate Wasserstein distance (Earth Mover's Distance).
    
    Measures the minimum cost of transforming one distribution into another.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference distribution samples.
    current : np.ndarray
        Current distribution samples.
        
    Returns
    -------
    float
        Wasserstein distance.
    """
    # Flatten if multidimensional
    if len(reference.shape) > 1:
        reference = reference.flatten()
    if len(current.shape) > 1:
        current = current.flatten()
    
    return scipy_wasserstein(reference, current)


def calculate_hellinger_distance(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """Calculate Hellinger distance.
    
    Hellinger distance measures the similarity between two probability distributions.
    
    Parameters
    ----------
    p : np.ndarray
        First distribution (must sum to 1).
    q : np.ndarray
        Second distribution (must sum to 1).
    eps : float
        Small value for numerical stability.
        
    Returns
    -------
    float
        Hellinger distance (between 0 and 1).
    """
    # Ensure distributions are normalized
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Add epsilon
    p = p + eps
    q = q + eps
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate Hellinger distance
    bc_sum = np.sum(np.sqrt(p * q))  # Bhattacharyya coefficient
    hellinger = np.sqrt(1 - bc_sum)
    
    return hellinger


def calculate_total_variation_distance(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = 1e-10
) -> float:
    """Calculate Total Variation distance.
    
    Maximum difference between probabilities assigned by two distributions.
    
    Parameters
    ----------
    p : np.ndarray
        First distribution (must sum to 1).
    q : np.ndarray
        Second distribution (must sum to 1).
    eps : float
        Small value for numerical stability.
        
    Returns
    -------
    float
        Total variation distance (between 0 and 1).
    """
    # Ensure distributions are normalized
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Add epsilon
    p = p + eps
    q = q + eps
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # TV distance
    return 0.5 * np.sum(np.abs(p - q))


def calculate_chi_squared_statistic(
    observed: np.ndarray,
    expected: np.ndarray,
    eps: float = 1e-10
) -> Tuple[float, int]:
    """Calculate Chi-squared test statistic.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed frequencies.
    expected : np.ndarray
        Expected frequencies.
    eps : float
        Small value to avoid division by zero.
        
    Returns
    -------
    Tuple[float, int]
        Chi-squared statistic and degrees of freedom.
    """
    # Ensure arrays
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    
    # Add epsilon to avoid division by zero
    expected = expected + eps
    
    # Calculate statistic
    chi2 = np.sum((observed - expected) ** 2 / expected)
    df = len(observed) - 1
    
    return chi2, df


def calculate_cramer_von_mises(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """Calculate Cramér-von Mises criterion.
    
    Non-parametric test for comparing two distributions.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference distribution samples.
    current : np.ndarray
        Current distribution samples.
        
    Returns
    -------
    float
        Cramér-von Mises statistic.
    """
    # Flatten if needed
    if len(reference.shape) > 1:
        reference = reference.flatten()
    if len(current.shape) > 1:
        current = current.flatten()
    
    n = len(reference)
    m = len(current)
    
    # Combine and sort
    combined = np.concatenate([reference, current])
    combined_sorted = np.sort(combined)
    
    # Calculate empirical CDFs
    ref_cdf = np.searchsorted(np.sort(reference), combined_sorted, side='right') / n
    curr_cdf = np.searchsorted(np.sort(current), combined_sorted, side='right') / m
    
    # Calculate statistic
    w2 = n * m / (n + m) ** 2 * np.sum((ref_cdf - curr_cdf) ** 2)
    
    return w2


def calculate_anderson_darling(
    reference: np.ndarray,
    current: np.ndarray
) -> float:
    """Calculate Anderson-Darling statistic.
    
    Weighted version of Cramér-von Mises that gives more weight to tails.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference distribution samples.
    current : np.ndarray
        Current distribution samples.
        
    Returns
    -------
    float
        Anderson-Darling statistic.
    """
    # Flatten if needed
    if len(reference.shape) > 1:
        reference = reference.flatten()
    if len(current.shape) > 1:
        current = current.flatten()
    
    n = len(reference)
    m = len(current)
    N = n + m
    
    # Combine and sort
    combined = np.concatenate([reference, current])
    combined_sorted = np.sort(combined)
    
    # Calculate ranks
    ref_ranks = np.searchsorted(combined_sorted, np.sort(reference))
    curr_ranks = np.searchsorted(combined_sorted, np.sort(current))
    
    # Calculate A-D statistic
    i = np.arange(1, N + 1)
    
    # Empirical CDFs
    ref_cdf = np.zeros(N)
    ref_cdf[ref_ranks] = np.arange(1, n + 1) / n
    
    curr_cdf = np.zeros(N)
    curr_cdf[curr_ranks] = np.arange(1, m + 1) / m
    
    # Weight function
    weights = 1.0 / (i * (N - i + 1))
    
    # A-D statistic
    ad2 = n * m / N * np.sum(weights * (ref_cdf - curr_cdf) ** 2)
    
    return ad2