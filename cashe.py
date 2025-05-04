import os
import numpy as np
import pickle

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
