import os
import numpy as np
from features.extract_features import extract_combined_features
from modules.mass import compute_mass_sdm
from modules.hubness import reduce_hubness
from modules.network import enhance_network
from modules.summarizer import summarize_features
from similarity.cross_similarity import compute_cross_similarity, rowwise_max, eucli
from other.cache import cache_or_compute, cache_all_wavs
import os
from random import sample
from glob import glob
from modules.chroma_summarizer import summarize_chroma_features, extract_mfcc
from scipy.spatial.distance import cdist
from other.results import evaluate_accuracy_for_all_results


def run_test():

    ts_root = "D:/ListenToMyHeartBeat/sampling/Training/TS"
    cache_all_wavs(ts_root)

    print("[테스트 로그 1/4] ▶ 모든 파일 캐시화 완료")

    wav_paths = glob(os.path.join(ts_root, "*", "*", "*", "*.wav"))

    for query_idx, query_path in enumerate(wav_paths, start=1):
        query_parts = query_path.replace("\\", "/").split("/")
        query_genre = query_parts[-4]
        query_song_id = query_parts[-3]
        query_cover = query_parts[-2]
        query_cache_id = f"{query_genre}_{query_song_id}_{query_cover}"

        query_feat = cache_or_compute(query_path, extract_combined_features, f"SuCo-HRNE/cache/{query_cache_id}/feat.npy")
        query_summary = cache_or_compute(
            None,
            lambda: summarize_chroma_features(query_feat, method='segment_mean_std', num_segments=10),
            f"SuCo-HRNE/cache/{query_cache_id}/summary.pkl",
            is_numpy=False
        )

        print(f"[쿼리 {query_idx}] ▶ {query_path}")
        print("쿼리 summary:", query_summary.shape)

        e_scores = []
        ref_list = []
        log_lines = []
        e = []

        for idx, ref_file in enumerate(wav_paths, start=1):
            if ref_file == query_path:
                continue

            print(f"[테스트 로그 2.{idx}/4] ▶ 비교 대상: {ref_file}")

            ref_parts = ref_file.replace("\\", "/").split("/")
            ref_genre = ref_parts[-4]
            ref_song_id = ref_parts[-3]
            ref_cover = ref_parts[-2]
            ref_cache_id = f"{ref_genre}_{ref_song_id}_{ref_cover}"

            ref_feat = cache_or_compute(ref_file, extract_combined_features, f"SuCo-HRNE/cache/{ref_cache_id}/feat.npy")
            ref_summary = cache_or_compute(
                None,
                lambda: summarize_chroma_features(ref_feat, method='segment_mean_std', num_segments=10),
                f"SuCo-HRNE/cache/{ref_cache_id}/summary.pkl",
                is_numpy=False
            )

            print("레퍼런스 summary:", ref_summary.shape)

            e_score = eucli(query_summary, ref_summary)
            e.append(e_score)
            e_scores.append(e_score)
            ref_list.append(ref_file)

            print(f"[테스트 로그 3.{idx}/4] ▶ 유사도 점수: {e_score:.4f}")

            # 디버깅 로그 기록
            log_lines.append(f"\n[비교 {idx}] ▶ {ref_file}")
            log_lines.append(f"쿼리 summary shape: {query_summary.shape}")
            log_lines.append(f"레퍼런스 summary shape: {ref_summary.shape}")

            # 평균 벡터 계산
            q_vec = np.mean(query_summary, axis=0)
            r_vec = np.mean(ref_summary, axis=0)

            # L2 노름 및 일부 벡터 출력
            q_norm = np.linalg.norm(q_vec)
            r_norm = np.linalg.norm(r_vec)

            log_lines.append(f"[벡터 평균 L2 노름] 쿼리: {q_norm:.6f}, 레퍼런스: {r_norm:.6f}")
            log_lines.append(f"[쿼리 평균 벡터 일부] {q_vec[:5]}")
            log_lines.append(f"[레퍼런스 평균 벡터 일부] {r_vec[:5]}")

            # 평균 벡터 간 유클리드 거리
            euclidean_dist = np.linalg.norm(q_vec - r_vec)
            log_lines.append(f"[평균 벡터 유클리드 거리] {euclidean_dist:.4f}")

            # 유클리드 거리 행렬 계산
            dist_matrix = cdist(query_summary, ref_summary, metric='euclidean')  # shape: (query_segments, ref_segments)
            dist_matrix_mean = np.mean(dist_matrix)
            rowwise_min = np.min(dist_matrix, axis=1)
            rowwise_min_mean = np.mean(rowwise_min)

            log_lines.append(f"[거리 행렬 평균] {dist_matrix_mean:.4f}")
            log_lines.append(f"[rowwise 최소 평균 거리] {rowwise_min_mean:.4f}")

        result_path = f"SuCo-HRNE/output/results/test_similarity_scores_eucli_{query_song_id}.txt"
        e_path = f"SuCo-HRNE/output/results/euclidean_similarity_eucli_{query_song_id}.txt"
        log_path = f"SuCo-HRNE/output/results/similarity_debug_log_eucli_{query_song_id}.txt"

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            for i, (ref, score) in enumerate(zip(ref_list, e_scores), start=1):
                f.write(f"{i}번째 음악 ({os.path.basename(ref)}) 유사도: {score:.4f}\n")

        with open(e_path, "w") as f:
            f.write("\n".join(map(str, e)))

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"[로그 저장 완료] ▶ {log_path}")


if __name__ == "__main__":
    #run_test()
    evaluate_accuracy_for_all_results()