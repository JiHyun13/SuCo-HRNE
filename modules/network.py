import numpy as np

def enhance_network(sdm, method='gaussian', diffusion=True):
    """
    sdm: self-distance matrix (N x N)
    method: 'gaussian' or other
    """
    if method == 'gaussian':
        W = np.exp(-sdm ** 2)

    # L1 normalization (row-wise)
    W = W / (np.sum(W, axis=1, keepdims=True) + 1e-8)

    if diffusion:
        W_enhanced = W @ W.T  # Diffusion step
        W_enhanced = (W_enhanced + W_enhanced.T) / 2  # Symmetrize
        return W_enhanced
    else:
        return W
