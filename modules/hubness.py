import numpy as np

def reduce_hubness(sdm, method='snn', k=10):
    if method == 'snn':
        # Shared Nearest Neighbors (SNN) Hubness Reduction
        N = sdm.shape[0]
        knn_indices = np.argsort(sdm, axis=1)[:, 1:k+1]  # exclude self (0th)
        snn_matrix = np.zeros_like(sdm)

        for i in range(N):
            for j in range(N):
                shared = len(set(knn_indices[i]) & set(knn_indices[j]))
                snn_matrix[i, j] = k - shared

        return snn_matrix
    elif method == 'local_scaling':
        # Local Scaling: divide by distance to k-th neighbor
        kth = np.sort(sdm, axis=1)[:, k]
        scale = kth[:, np.newaxis] * kth[np.newaxis, :]
        return sdm / (scale + 1e-8)
    elif method == 'mp':
        # Mutual Proximity
        mu = np.mean(sdm)
        sigma = np.std(sdm)
        mp_matrix = 1 - np.exp(-((sdm - mu) ** 2) / (2 * sigma ** 2))
        return mp_matrix
    else:
        return sdm