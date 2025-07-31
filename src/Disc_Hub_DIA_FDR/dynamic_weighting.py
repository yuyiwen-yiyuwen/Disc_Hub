import numpy as np

def dynamic_weighting(probs_matrix):
    """
    Applies dynamic weighting to an ensemble of probability predictions based on confidence.

    This function calculates weights for each model's predictions based on the standard deviation
    across models: lower standard deviation (i.e., higher confidence) results in a higher weight.
    The weights are then normalized to sum to 1 across models for each sample.

    Args:
        probs_matrix (ndarray): A 2D NumPy array of shape (n_samples, n_models) representing
                                predicted probabilities from multiple models.

    Returns:
        ndarray: A 2D array of normalized confidence-based weights of shape (n_samples, 1),
                 to be applied to the ensemble predictions.
    """
    confidence_weights = np.std(probs_matrix, axis=0, keepdims=True)
    confidence_weights = 1 / (confidence_weights + 1e-8)
    return confidence_weights / confidence_weights.sum(axis=1, keepdims=True)