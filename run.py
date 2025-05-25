import os
import numpy as np
from features.extract_features import extract_combined_features
from modules.sdm import compute_cosine_sdm
from modules.hubness import reduce_hubness
from modules.network import enhance_network
from similarity.cross_similarity import compute_cross_similarity, eucli
from other.cache import cache_or_compute, cache_all_wavs
from glob import glob
from modules.chroma_summarizer import summarize_features
from other.results import evaluate_accuracy_for_all_results
import csv



def run():

    ts_root = "D:/ListenToMyHeartBeat/sampling/Training/TS"
    cache_all_wavs(ts_root)

    print("모든 파일 캐시화 완료")

    wav_paths = glob(os.path.join(ts_root, "*", "*", "*", "*.wav"))

    for query_idx, query_path in enumerate(wav_paths, start=1):
        query_parts = query_path.replace("\\", "/").split("/")
        query_genre = query_parts[-4]
        query_song_id = query_parts[-3]
        query_cover = query_parts[-2]
        query_cache_id = f"{query_genre}_{query_song_id}_{query_cover}"

        result_path = f"SuCo-HRNE/output/results/scores_eucli_{query_song_id}.csv"

        query_feat = cache_or_compute(query_path, extract_combined_features, f"SuCo-HRNE/cache/{query_cache_id}/feat.npy")
        query_summary = cache_or_compute(
            None,
            lambda: summarize_features(query_feat, method='segment_mean_std', num_segments=10),
            f"SuCo-HRNE/cache/{query_cache_id}/summary.pkl",
            is_numpy=False
        )

        print(f"=================================[쿼리 {query_idx}] ▶ {query_path}====================================")

        e_scores = []
        ref_list = []

        with open(result_path, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["type", "genre", "song_id", "cover_type", "filename", "similarity"])

            # 쿼리
            writer.writerow(["query", query_genre, query_song_id, query_cover, os.path.basename(query_path), 100.00])


            for idx, ref_file in enumerate(wav_paths, start=1):
                if ref_file == query_path:
                    continue

                ref_parts = ref_file.replace("\\", "/").split("/")
                ref_genre = ref_parts[-4]
                ref_song_id = ref_parts[-3]
                ref_cover = ref_parts[-2]
                ref_cache_id = f"{ref_genre}_{ref_song_id}_{ref_cover}"

                ref_feat = cache_or_compute(ref_file, extract_combined_features, f"SuCo-HRNE/cache/{ref_cache_id}/feat.npy")
                ref_summary = cache_or_compute(
                    None,
                    lambda: summarize_features(ref_feat, method='segment_mean_std', num_segments=10),
                    f"SuCo-HRNE/cache/{ref_cache_id}/summary.pkl",
                    is_numpy=False
                )

                e_score = eucli(query_summary, ref_summary)
                e_scores.append(e_score)
                ref_list.append(ref_file)

                #유사도 측정 로그 필요

            
                writer.writerow(["reference", ref_genre, ref_song_id, ref_cover, os.path.basename(ref_file), round(e_score, 4)])
            # os.makedirs(os.path.dirname(result_path), exist_ok=True)
            # with open(result_path, "w", encoding="utf-8") as f:
            #     for i, (ref, score) in enumerate(zip(ref_list, e_scores), start=1):
            #         f.write(f"{i}번째 음악 ({os.path.basename(ref)}) 유사도: {score:.4f}\n")
            #         if i%200 == 0 :
            #             print(f"{i}/{len(ref_list)}  ▶ 저장중...")


    evaluate_accuracy_for_all_results()

if __name__ == "__main__":
    evaluate_accuracy_for_all_results()
    #run()
    