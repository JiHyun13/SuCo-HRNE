import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def compute_cross_similarity(query_summary, ref_summary):
    if len(query_summary) == 0 or len(ref_summary) == 0:
        print("[디버깅] 요약 벡터가 비어 있음 → 유사도: 0.0")
        return 0.0

    q_vec = np.mean(query_summary, axis=0)
    r_vec = np.mean(ref_summary, axis=0)

    q_norm = np.linalg.norm(q_vec)
    r_norm = np.linalg.norm(r_vec)

    print(f"[디버깅] 평균 벡터 차원: 쿼리={len(q_vec)}, 레퍼런스={len(r_vec)}")
    print(f"[디버깅] 평균 벡터 L2 노름: 쿼리={q_norm:.8f}, 레퍼런스={r_norm:.8f}")

    if q_norm < 1e-10 or r_norm < 1e-10:
        print("[디버깅] 벡터의 L2 노름이 매우 작음 → 유사도: 0.0")
        return 0.0

    # 차원 맞추기
    min_len = min(len(q_vec), len(r_vec))
    q_vec = q_vec[:min_len].reshape(1, -1)
    r_vec = r_vec[:min_len].reshape(1, -1)

    q_vec = normalize(q_vec)
    r_vec = normalize(r_vec)

    sim = cosine_similarity(q_vec, r_vec)
    print(f"[디버깅] 최종 코사인 유사도: {sim[0][0]:.4f}")
    return sim[0][0]