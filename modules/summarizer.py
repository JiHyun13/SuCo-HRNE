import numpy as np

def find_repeats(ssm, threshold=0.8, min_len=5):
    """
    SSM에서 반복되는 구간 추출
    """
    N = ssm.shape[0]
    repeated = []

    for i in range(N - min_len):
        for j in range(i + 1, N - min_len):
            similarity = np.mean(ssm[i:i+min_len, j:j+min_len])
            if similarity > threshold:
                repeated.append((i, j, similarity))

    return repeated

def summarize_features(ssm, method='repeat', num_segments=5):
    """
    반복 구조 기반 또는 썸네일 기반 요약
    - ssm: self-similarity matrix
    - method: 'repeat' 또는 'thumb'
    - num_segments: 요약 구간 수
    """
    if method == 'repeat':
        repeats = find_repeats(ssm, threshold=0.85, min_len=5)

        if not repeats:
            print("[요약 실패] 반복 구간 없음 → fallback to thumb")
            return summarize_features(ssm, method='thumb', num_segments=num_segments)

        repeats = sorted(repeats, key=lambda x: -x[2])  # 유사도 기준 정렬

        segments = []
        for i, j, sim in repeats[:num_segments]:
            center = (i + j) // 2
            segments.append(ssm[center])

        return np.array(segments)

    elif method == 'thumb':
        scores = np.zeros(ssm.shape[0])
        win = 5
        for i in range(ssm.shape[0] - win):
            scores[i] = np.mean(ssm[i:i+win, i:i+win])

        top_indices = np.argsort(scores)[-num_segments:]
        return ssm[top_indices]

    else:
        raise ValueError("Unknown summarization method")
