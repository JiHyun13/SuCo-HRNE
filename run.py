import os
import numpy as np
from features.extract_features import extract_combined_features
from modules.mass import compute_mass_sdm
from modules.hubness import reduce_hubness
from modules.network import enhance_network
from modules.summarizer import summarize_features
from similarity.cross_similarity import compute_cross_similarity, rowwise_max
from cashe import cache_or_compute

from modules.chroma_summarizer import summarize_chroma_features

def run():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    query_path = "C:/Users/zappe/Desktop/창학/SuCo-HRNE/data/query/Similar_Ballade_00001_Org.wav"
    ref_dir = "C:/Users/zappe/Desktop/창학/SuCo-HRNE/data/reference"

    query_id = query_path.split("_")[2]  # e.g., '00001'

    print(f"[로그 1/14] ▶ 쿼리 곡 로딩 중: {query_path}")
    query_feat = cache_or_compute(query_path, extract_combined_features, f"cache/{query_id}/query_feat.npy")
    print("[로그 2/14]    - 크로마 특징 추출 완료")

    query_sdm = cache_or_compute(None, lambda: compute_mass_sdm(query_feat), f"cache/{query_id}/query_sdm.npy")
    print("[로그 3/14]    - SDM 계산 완료")

    query_sdm_hr = cache_or_compute(None, lambda: reduce_hubness(query_sdm), f"cache/{query_id}/query_sdm_hr.npy")
    print("[로그 4/14]    - 허브니스 제거 완료")

    query_ssm = cache_or_compute(None, lambda: enhance_network(query_sdm_hr), f"cache/{query_id}/query_ssm.npy")
    print("[로그 5/14]    - 네트워크 강화 완료")

    query_summary = cache_or_compute(None, lambda: summarize_features(query_ssm), f"cache/{query_id}/query_summary.pkl", is_numpy=False)
    print("[로그 6/14]    - 요약 특징 추출 완료")

    sim_scores = []
    ref_list = []
    total_refs = len([f for f in os.listdir(ref_dir) if f.endswith(".wav")])
    current_idx = 1

    for ref_file in os.listdir(ref_dir):
        if not ref_file.endswith(".wav"):
            continue

        ref_path = os.path.join(ref_dir, ref_file)
        ref_id = ref_file.split("_")[2]

        print(f"[로그 7/14] ▶ ({current_idx}/{total_refs}) 비교 대상: {ref_file}")

        ref_feat = cache_or_compute(ref_path, extract_combined_features, f"cache/{query_id}/ref_{ref_id}_feat.npy")
        print(f"[로그 8/14] ▶ ({current_idx}/{total_refs})   - 특징 추출 완료")

        ref_sdm = cache_or_compute(None, lambda: compute_mass_sdm(ref_feat), f"cache/{query_id}/ref_{ref_id}_sdm.npy")
        print(f"[로그 9/14] ▶ ({current_idx}/{total_refs})   - SDM 계산 완료")

        ref_sdm_hr = cache_or_compute(None, lambda: reduce_hubness(ref_sdm), f"cache/{query_id}/ref_{ref_id}_sdm_hr.npy")
        ref_ssm = cache_or_compute(None, lambda: enhance_network(ref_sdm_hr), f"cache/{query_id}/ref_{ref_id}_ssm.npy")
        ref_summary = cache_or_compute(None, lambda: summarize_features(ref_ssm), f"cache/{query_id}/ref_{ref_id}_summary.pkl", is_numpy=False)

        sim_score = rowwise_max(query_summary, ref_summary)
        print(f"[로그 10/14] ▶ ({current_idx}/{total_refs})  - 유사도 계산 결과 ({ref_file}): {sim_score:.4f}")

        sim_scores.append(sim_score)
        ref_list.append(ref_file)
        current_idx += 1

    # 정확도 계산
    threshold = 0.5
    TP, TN, FP, FN = 0, 0, 0, 0
    for ref, score in zip(ref_list, sim_scores):
        pred = int(score >= threshold)
        ref_id = ref.split("_")[2]
        label = 1 if ref_id == query_id else 0
        if pred == 1 and label == 1: TP += 1
        elif pred == 0 and label == 0: TN += 1
        elif pred == 1 and label == 0: FP += 1
        elif pred == 0 and label == 1: FN += 1

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    print("\n[로그 11/14] --- 최종 평가 결과 ---")
    print(f"정확도: {accuracy:.2%}")
    print(f"TP(정답=1, 예측=1): {TP}")
    print(f"TN(정답=0, 예측=0): {TN}")
    print(f"FP(정답=0, 예측=1): {FP}")
    print(f"FN(정답=1, 예측=0): {FN}")
    print("[로그 12/14] 전체 비교 완료")

    # 유사도 결과 저장
    result_path = "data/results/similarity_scores_이름이나버전.txt"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        for i, (ref, score) in enumerate(zip(ref_list, sim_scores), start=1):
            f.write(f"{i}번째 음악 ({ref}) 유사도: {score:.4f}\n")
    print(f"[로그 13/14] 유사도 결과 저장 완료 → {result_path}")

    print("[로그 14/14] 프로그램 종료 완료")

def run_test(): 

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    query_path = "C:/Users/zappe/Desktop/창학/SuCo-HRNE/data/query/Similar_Ballade_00001_Org.wav"
    ref_dir = "C:/Users/zappe/Desktop/창학/SuCo-HRNE/data/reference"
    query_id = query_path.split("_")[2]

    cos = []
    log_lines = []  # 로그 저장용 리스트

    print(f"[테스트 로그 1/10] ▶ 쿼리 곡 로딩 중: {query_path}")
    log_lines.append(f"[테스트 로그 1/10] ▶ 쿼리 곡 로딩 중: {query_path}")
    query_feat = cache_or_compute(query_path, extract_combined_features, f"cache/{query_id}/query_feat.npy")

    query_summary = cache_or_compute(
        None,
        lambda: summarize_chroma_features(query_feat, method='segment_mean_std', num_segments=5),
        f"cache/{query_id}/query_summary.pkl",
        is_numpy=False
    )

    print("[테스트 로그 6/10] ▶ 쿼리 요약 특징 추출 완료")
    log_lines.append("[테스트 로그 6/10] ▶ 쿼리 요약 특징 추출 완료")
    print("쿼리 summary:", query_summary.shape)
    log_lines.append(f"쿼리 summary: {query_summary.shape}")

    ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".wav")]
    sim_scores = []
    ref_list = []

    for idx, ref_file in enumerate(ref_files, start=1):
        print(f"[테스트 로그 7.{idx}/10] ▶ 비교 대상: {ref_file}")
        log_lines.append(f"[테스트 로그 7.{idx}/10] ▶ 비교 대상: {ref_file}")
        ref_path = os.path.join(ref_dir, ref_file)
        ref_id = ref_file.split("_")[2]

        ref_feat = cache_or_compute(ref_path, extract_combined_features, f"cache/{query_id}/ref_{ref_id}_feat.npy")
        ref_summary = cache_or_compute(
            None,
            lambda: summarize_chroma_features(ref_feat, method='segment_mean_std', num_segments=30),
            f"cache/{query_id}/ref_{ref_id}_summary.pkl",
            is_numpy=False
        )

        print("레퍼런스 summary:", ref_summary.shape)
        log_lines.append(f"레퍼런스 summary: {ref_summary.shape}")

        sim_score = rowwise_max(query_summary, ref_summary)
        cos.append(sim_score)
        sim_scores.append(sim_score)
        ref_list.append(ref_file)

        print(f"[테스트 로그 8.{idx}/10] ▶ 유사도 점수: {sim_score:.4f}")
        log_lines.append(f"[테스트 로그 8.{idx}/10] ▶ 유사도 점수: {sim_score:.4f}")

    result_path = "data/results/test_similarity_scores3.txt"
    cos_path = "data/results/cosine_similarity.txt"
    log_path = "data/results/similarity_debug_log.txt"

    with open(cos_path, "w") as f:
        f.write("\n".join(map(str, cos)))

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        for i, (ref, score) in enumerate(zip(ref_list, sim_scores), start=1):
            f.write(f"{i}번째 음악 ({ref}) 유사도: {score:.4f}\n")

    threshold = 0.5
    TP, TN, FP, FN = 0, 0, 0, 0

    print("\n[디버깅] ▶ 정확도 계산 시작")
    log_lines.append("\n[디버깅] ▶ 정확도 계산 시작")

    for ref, score in zip(ref_list, sim_scores):
        pred = int(score >= threshold)
        ref_id = ref.split("_")[2]
        label = 1 if ref_id == query_id else 0

        log = f"[디버깅] 비교: {ref} | 예측={pred} | 정답={label} | 점수={score:.4f}"
        print(log)
        log_lines.append(log)

        if pred == 1 and label == 1: TP += 1
        elif pred == 0 and label == 0: TN += 1
        elif pred == 1 and label == 0: FP += 1
        elif pred == 0 and label == 1: FN += 1

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    print(f"[디버깅] ▶ 정확도 계산 완료: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    print(f"[디버깅] ▶ 정확도: {accuracy:.2%}")
    log_lines.append(f"[디버깅] ▶ 정확도 계산 완료: TP={TP}, TN={TN}, FP={FP}, FN={FN}")
    log_lines.append(f"[디버깅] ▶ 정확도: {accuracy:.2%}")

    print("[테스트 로그 10/10] ▶ 테스트 완료 및 결과 저장 완료 →", result_path)
    log_lines.append("[테스트 로그 10/10] ▶ 테스트 완료 및 결과 저장 완료 → " + result_path)

    # ✅ 로그 저장
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[로그 저장 완료] ▶ {log_path}")

if __name__ == "__main__":
    run_test()
