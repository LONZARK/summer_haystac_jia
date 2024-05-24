import numpy as np


def estimate_bins(scores: dict, num_bins=30) -> np.ndarray:
    """

    Parameters
    ----------
    scores: dict
        Scores
    num_bins : int
         (defaults to 30)

    Returns
    -------
    type
        data in a score dictionary
    """

    # Reduce scores to a single array
    reduced = np.concatenate([v for v in scores.values()])
    return np.linspace(reduced.min(), reduced.max(), num_bins)
