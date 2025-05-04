def compute_cross_similarity(query_summary, ref_summary):
    scores = []
    for q_vec in query_summary:
        for r_vec in ref_summary:
            if len(q_vec) == 0 or len(r_vec) == 0:
                continue  # 빈 벡터는 건너뛰기
            sim = cosine_similarity(q_vec, r_vec)
            scores.append(sim)
    if not scores:
        return 0.0  # 유사도 계산 실패 시 기본값 반환
    return max(scores)