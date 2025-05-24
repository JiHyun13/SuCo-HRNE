import os
import numpy as np
import pickle
from glob import glob
from features.extract_features import extract_combined_features
from modules.chroma_summarizer import summarize_chroma_features

def cache_or_compute(path_or_input, compute_fn, save_path, is_numpy=True):
    """
    path_or_input: 입력 경로 또는 이미 계산된 이전 단계 결과
    compute_fn: 계산 함수 (입력이 필요 없는 경우 lambda로 감쌈)
    save_path: 결과 저장 경로
    is_numpy: numpy 저장인지, pickle 저장인지 선택
    """
    if os.path.exists(save_path):
        print(f"  - [불러옴] {os.path.basename(save_path)}")
        if is_numpy:
            return np.load(save_path)
        else:
            with open(save_path, 'rb') as f:
                return pickle.load(f)
    else:
        result = compute_fn(path_or_input) if path_or_input is not None else compute_fn()
        print(f"  - [계산] {os.path.basename(save_path)}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if is_numpy:
            np.save(save_path, result)
        else:
            with open(save_path, 'wb') as f:
                pickle.dump(result, f)
        return result
    

def cache_all_wavs(ts_root):
    """
    전체 TS 디렉토리 아래의 모든 .wav 파일에 대해 feature와 summary를 캐시하는 함수
    - feat.npy
    - summary.pkl
    """
    wav_paths = glob(os.path.join(ts_root, "*", "*", "*", "*.wav"))
    print(f"[INFO] 총 {len(wav_paths)}개의 wav 파일을 찾았습니다.")
    n=0

    for idx, wav_path in enumerate(wav_paths, start=1):
        try:
            path_parts = wav_path.replace("\\", "/").split("/")
            genre = path_parts[-4]
            song_id = path_parts[-3]
            cover = path_parts[-2]
            cache_id = f"{genre}_{song_id}_{cover}"

            print(f"[{idx}/{len(wav_paths)}] ▶ 캐시 중: {wav_path}")

            # Feature 저장
            feat = cache_or_compute(wav_path, extract_combined_features, f"SuCo-HRNE/cache/{cache_id}/feat.npy")

            # Summary 저장
            _ = cache_or_compute(
                None,
                lambda: summarize_chroma_features(feat, method='segment_mean_std', num_segments=10),
                f"SuCo-HRNE/cache/{cache_id}/summary.pkl",
                is_numpy=False
            )

            n+=1
            if n>100:
                print("[완료] 100개에 대한 wav 파일에 대해 캐시 작업이 완료되었습니다.")
                return 0
        except Exception as e:
            print(f"[ERROR] {wav_path} 처리 중 예외 발생: {e}")

    print("[완료] 모든 wav 파일에 대해 캐시 작업이 완료되었습니다.")


