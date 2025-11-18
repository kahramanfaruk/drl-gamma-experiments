import numpy as np

# =============================================================================
# Utility: One-hot state encoding
# =============================================================================
def one_hot_state(state, state_size=16):
    """
    Convert a scalar state index to a one-hot encoded vector.

    Parameters
    ----------
    state : int
        Scalar representation of the environment state.
    state_size : int, optional
        Number of total states (default is 16 for FrozenLake 4x4).

    Returns
    -------
    np.ndarray
        One-hot encoded vector representation of the state.
    """
    state_vec = np.zeros(state_size, dtype=np.float32)
    state_vec[state] = 1.0
    return state_vec