import os
import numpy as np
import pickle
from glob import glob
from features.extract_features import extract_combined_features

def cache_or_compute(path_or_input, compute_fn, save_path, is_numpy=True):
    if os.path.exists(save_path):
        if is_numpy:
            return np.load(save_path)
        else:
            with open(save_path, 'rb') as f:
                return pickle.load(f)
    else:
        result = compute_fn(path_or_input) if path_or_input is not None else compute_fn()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if is_numpy:
            np.save(save_path, result)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)
        return result
    

def cache_all_wavs(ts_root):
    wav_paths = glob(os.path.join(ts_root, "*", "*", "*", "*.wav"))
    print(f"[INFO] 총 {len(wav_paths)}개의 wav 파일을 찾았습니다.")

    for idx, wav_path in enumerate(wav_paths, start=1):
        try:
            path_parts = wav_path.replace("\\", "/").split("/")
            genre = path_parts[-4]
            song_id = path_parts[-3]
            cover = path_parts[-2]
            cache_id = f"{genre}_{song_id}_{cover}"

            feat_path = f"SuCo-HRNE/cache/{cache_id}/feat.npy"
            summary_path = f"SuCo-HRNE/cache/{cache_id}/summary.pkl"

            # 둘 다 이미 존재하면 건너뛰기
            if os.path.exists(feat_path) and os.path.exists(summary_path):
                print(f"[{idx}/{len(wav_paths)}] ▶ 스킵 (이미 존재): {wav_path}")
                continue

            print(f"[{idx}/{len(wav_paths)}] ▶ 캐시 중: {wav_path}")

            # Feature 저장
            feat = cache_or_compute(wav_path, extract_combined_features, feat_path)

            # Summary 저장
            _ = cache_or_compute(
                None,
                lambda: summarize_chroma_features(feat, method='segment_mean_std', num_segments=10),
                summary_path,
                is_numpy=False
            )

        except Exception as e:
            print(f"[ERROR] {wav_path} 처리 중 예외 발생: {e}")

    print("[완료] 모든 wav 파일에 대해 캐시 작업이 완료되었습니다.")
