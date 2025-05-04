import numpy as np
from scipy.sparse import csr_matrix

def reduce_hubness(sdm, k=10):
    N = sdm.shape[0]
    knn_indices = np.argsort(sdm, axis=1)[:, 1:k+1]  # exclude self

    # 1. Sparse binary k-NN matrix (N x N)
    row_idx = np.repeat(np.arange(N), k)
    col_idx = knn_indices.flatten()
    data = np.ones(len(row_idx))
    knn_sparse = csr_matrix((data, (row_idx, col_idx)), shape=(N, N))

    # 2. SNN: shared neighbor count = dot product
    shared_counts = knn_sparse @ knn_sparse.T  # (N x N), fast!

    # 3. 거리로 바꾸기 (공유 이웃 적을수록 거리 큼)
    snn_matrix = k - shared_counts.toarray()

    return snn_matrix
