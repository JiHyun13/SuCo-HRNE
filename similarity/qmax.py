import numpy as np
from scipy.spatial.distance import cdist

def embed_sequence(X, dim=3, tau=1):
    """
    상태 공간 재구성 (delay embedding)
    """
    N = X.shape[0] - (dim - 1) * tau
    embedded = np.zeros((N, X.shape[1] * dim))
    for i in range(N):
        for d in range(dim):
            embedded[i, d*X.shape[1]:(d+1)*X.shape[1]] = X[i + d * tau]
    return embedded

def create_crp(X_emb, Y_emb, threshold=None):
    """
    Cross Recurrence Plot: 거리 기반 이진 행렬
    """
    dist_mat = cdist(X_emb, Y_emb, metric='euclidean')
    if threshold is None:
        threshold = np.percentile(dist_mat, 20)  # 20% 이내를 recurrence로 간주
    crp = (dist_mat < threshold).astype(int)
    return crp

def longest_diagonal(crp):
    """
    가장 긴 대각선 streak의 길이 반환
    """
    max_len = 0
    rows, cols = crp.shape
    for offset in range(-rows + 1, cols):
        diag = np.diagonal(crp, offset=offset)
        curr = 0
        for val in diag:
            if val == 1:
                curr += 1
                max_len = max(max_len, curr)
            else:
                curr = 0
    return max_len

def qmax_similarity(query_seq, reference_seq):
    """
    Qmax 유사도 계산 (repeat 구조 유사도 기반)
    """
    # 상태공간 임베딩
    dim = 3
    tau = 1
    X_emb = embed_sequence(query_seq, dim=dim, tau=tau)
    Y_emb = embed_sequence(reference_seq, dim=dim, tau=tau)

    # 교차 재발현 행렬 생성
    crp = create_crp(X_emb, Y_emb)

    # 최장 대각선 길이 추출
    max_diag = longest_diagonal(crp)

    # 정규화된 유사도 점수
    norm_score = max_diag / min(len(X_emb), len(Y_emb))
    return norm_score
