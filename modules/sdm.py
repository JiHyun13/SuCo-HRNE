import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cosine_sdm(features):
    """
    Cosine-based Self-Distance Matrix (SDM)
    Input: features (T, D) → output: (T, T)
    """
    if features.shape[0] < 2:
        raise ValueError("Too few time frames to compute SDM")

    sim = cosine_similarity(features)  # similarity matrix (T, T)
    dist = 1 - sim  # distance matrix
    return dist


def summarize_features_from_ssm(ssm, method='thumbnail', num_segments=10):
    """
    요약 벡터 추출 (thumbnail 방식)
    Input: ssm (T, T) → Output: (num_segments, T)
    """
    T = ssm.shape[0]
    window = 5
    scores = np.zeros(T)

    for i in range(T - window):
        block = ssm[i:i + window, i:i + window]
        scores[i] = np.mean(block)

    top_indices = np.argsort(scores)[:num_segments]  # 거리 기반이므로 낮은 점수 우선

    summaries = ssm[top_indices]  # shape: (num_segments, T)

    # 정규화
    norms = np.linalg.norm(summaries, axis=1, keepdims=True)
    summaries = summaries / (norms + 1e-8)

    return summaries


def summary_similarity(query_summary, ref_summary):
    """
    요약 벡터 간 row-wise max cosine similarity → 평균값
    Input: (N, D), (M, D)
    """
    sim_matrix = cosine_similarity(query_summary, ref_summary)
    row_max = np.max(sim_matrix, axis=1)
    score = np.mean(row_max)
    return score * 100
