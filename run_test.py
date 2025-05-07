import os
import numpy as np
from features.extract_features import extract_chroma
from modules.mass import compute_mass_sdm
from modules.hubness import reduce_hubness
from modules.network import enhance_network
from modules.summarizer import summarize_features
from similarity.cross_similarity import compute_cross_similarity
from similarity.cross_similarity import compute_segmentwise_cosine_similarity
from cashe import cache_or_compute
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def summarize_chroma_features(chroma_feat, method='mean', num_segments=5):
    """
    크로마 특징에서 평균 or 세그먼트 기반 요약 벡터 추출
    """
    if chroma_feat is None or len(chroma_feat) == 0:
        return np.zeros((num_segments, 12))

    if method == 'mean':
        return np.mean(chroma_feat, axis=0, keepdims=True)

    elif method == 'segment':
        time_len = chroma_feat.shape[0]
        segment_len = time_len // num_segments
        segments = []

        for i in range(num_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < num_segments - 1 else time_len
            segment = chroma_feat[start:end]
            segments.append(np.mean(segment, axis=0))

        return np.array(segments)

    else:
        raise ValueError("Unknown method for summarizing chroma features.")


def compute_hybrid_similarity(query_feat, ref_feat, alpha=0.5, num_segments=5):
    """
    평균 기반 + 세그먼트 기반 유사도 결합
    """
    # 평균 기반
    q_mean = summarize_chroma_features(query_feat, method='mean')
    r_mean = summarize_chroma_features(ref_feat, method='mean')

    q_mean = normalize(q_mean)
    r_mean = normalize(r_mean)
    sim_mean = cosine_similarity(q_mean, r_mean)[0][0]

    # 세그먼트 기반
    q_seg = summarize_chroma_features(query_feat, method='segment', num_segments=num_segments)
    r_seg = summarize_chroma_features(ref_feat, method='segment', num_segments=num_segments)

    min_len = min(len(q_seg), len(r_seg))
    q_seg = normalize(q_seg[:min_len])
    r_seg = normalize(r_seg[:min_len])
    sim_seg_list = [cosine_similarity(q_seg[i].reshape(1, -1), r_seg[i].reshape(1, -1))[0][0] for i in range(min_len)]
    sim_seg = np.mean(sim_seg_list)

    return alpha * sim_mean + (1 - alpha) * sim_seg


def run_test():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    query_path = "data/query/Similar_Ballade_00001_Org.wav"
    ref_dir = "data/reference"
    query_id = query_path.split("_")[2]

    print(f"[테스트 로그 1/10] ▶ 쿼리 곡 로딩 중: {query_path}")
    query_feat = cache_or_compute(query_path, extract_chroma, f"cache/{query_id}/query_feat.npy")

    print("[테스트 로그 6/10] ▶ 쿼리 크로마 특징 추출 완료")

    ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".wav")]
    sim_scores = []
    ref_list = []

    for idx, ref_file in enumerate(ref_files, start=1):
        print(f"[테스트 로그 7.{idx}/10] ▶ 비교 대상: {ref_file}")
        ref_path = os.path.join(ref_dir, ref_file)
        ref_id = ref_file.split("_")[2]

        ref_feat = cache_or_compute(ref_path, extract_chroma, f"cache/{query_id}/ref_{ref_id}_feat.npy")

        sim_score = compute_hybrid_similarity(query_feat, ref_feat, alpha=0.5, num_segments=5)
        sim_scores.append(sim_score)
        ref_list.append(ref_file)
        print(f"[테스트 로그 8.{idx}/10] ▶ 유사도 점수: {sim_score:.4f}")

    result_path = "data/results/test_similarity_scores_hybrid.txt"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        for i, (ref, score) in enumerate(zip(ref_list, sim_scores), start=1):
            f.write(f"{i}번째 음악 ({ref}) 유사도: {score:.4f}\n")

    print("[테스트 로그 10/10] ▶ 테스트 완료 및 결과 저장 완료 →", result_path)


if __name__ == "__main__":
    run_test()
