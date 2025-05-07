import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def compute_segmentwise_cosine_similarity(query_summary, ref_summary):
    """
    segment 간 코사인 유사도를 평균내어 유사도 점수로 사용.
    두 입력은 (num_segments, 12) 형태여야 함.
    """
    if query_summary.shape[1] != ref_summary.shape[1]:
        print("[경고] 크로마 차원이 일치하지 않음.")
        min_dim = min(query_summary.shape[1], ref_summary.shape[1])
        query_summary = query_summary[:, :min_dim]
        ref_summary = ref_summary[:, :min_dim]

    sim_matrix = cosine_similarity(query_summary, ref_summary)
    sim_score = np.mean(sim_matrix)

    print(f"[디버깅] segment-wise 코사인 유사도 평균: {sim_score:.4f}")
    return sim_score




def compute_cross_similarity(query_summary, ref_summary):
    if len(query_summary) == 0 or len(ref_summary) == 0:
        print("[디버깅] 요약 벡터가 비어 있음 → 유사도: 0.0")
        return 0.0

    q_vec = np.mean(query_summary, axis=0)
    r_vec = np.mean(ref_summary, axis=0)

    q_norm = np.linalg.norm(q_vec)
    r_norm = np.linalg.norm(r_vec)

    print(f"[디버깅] 평균 벡터 차원: 쿼리={len(q_vec)}, 레퍼런스={len(r_vec)}")
    print(f"[디버깅] 평균값 범위: 쿼리=[{np.min(q_vec):.2e}, {np.max(q_vec):.2e}], 레퍼런스=[{np.min(r_vec):.2e}, {np.max(r_vec):.2e}]")
    print(f"[디버깅] 평균 벡터 L2 노름: 쿼리={q_norm:.8f}, 레퍼런스={r_norm:.8f}")

    if q_norm < 1e-10 or r_norm < 1e-10:
        print("[디버깅] 벡터의 L2 노름이 매우 작음 → 유사도: 0.0")
        return 0.0

    # 차원 맞추기
    min_len = min(len(q_vec), len(r_vec))
    q_vec = q_vec[:min_len].reshape(1, -1)
    r_vec = r_vec[:min_len].reshape(1, -1)

# 정규화 전 값 일부 출력
    print(f"[디버깅] 정규화 전 벡터 샘플 (쿼리): {q_vec[0][:5]}")
    print(f"[디버깅] 정규화 전 벡터 샘플 (레퍼런스): {r_vec[0][:5]}")

    q_vec = normalize(q_vec)
    r_vec = normalize(r_vec)

    # 정규화 후 벡터 일부 확인
    print(f"[디버깅] 정규화 후 벡터 샘플 (쿼리): {q_vec[0][:5]}")
    print(f"[디버깅] 정규화 후 벡터 샘플 (레퍼런스): {r_vec[0][:5]}")

    sim = cosine_similarity(q_vec, r_vec)
    print(f"[디버깅] 최종 코사인 유사도: {sim[0][0]:.4f}")
    return sim[0][0]