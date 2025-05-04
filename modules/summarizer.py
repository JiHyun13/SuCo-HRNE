import numpy as np

def summarize_features(ssm, method='repeat', num_segments=5):
    """
    특징 요약 벡터 생성
    - method: 'repeat' 또는 'thumbnail'
    - fallback 포함
    - 값 정규화 처리 포함
    """
    if method == 'repeat':
        repeats = find_repeats(ssm, threshold=0.85, min_len=5)
        if len(repeats) == 0:
            print("[요약 실패] 반복 구간 없음 → thumbnail 방식으로 대체")
            return summarize_features(ssm, method='thumbnail', num_segments=num_segments)

        # 유사도 기준으로 내림차순 정렬
        repeats = sorted(repeats, key=lambda x: -x[2])

        segments = []
        for i, j, sim in repeats[:num_segments]:
            center = (i + j) // 2
            vec = ssm[center]
            segments.append(vec)

    elif method == 'thumbnail':
        scores = np.zeros(ssm.shape[0])
        win = 5
        for i in range(ssm.shape[0] - win):
            scores[i] = np.mean(ssm[i:i+win, i:i+win])

        top_indices = np.argsort(scores)[-num_segments:]
        segments = [ssm[i] for i in top_indices]

    else:
        raise ValueError("Unknown summarization method")

    # np.array로 변환 + 정규화 (단위 벡터로)
    summary = np.array(segments)
    norms = np.linalg.norm(summary, axis=1, keepdims=True)
    summary = summary / (norms + 1e-8)  # divide by norm, avoid 0-division

    return summary


def find_repeats(ssm, threshold=0.85, min_len=5):
    """
    대각선 위에서 반복 구조 찾기
    - threshold 이상 유사도
    - min_len 이상 길이
    """
    N = ssm.shape[0]
    repeats = []
    for offset in range(1, N):
        diag = np.diag(ssm, k=offset)
        i = 0
        while i < len(diag):
            if diag[i] > threshold:
                start = i
                while i < len(diag) and diag[i] > threshold:
                    i += 1
                end = i
                if end - start >= min_len:
                    repeats.append((start, start + offset, np.mean(diag[start:end])))
            else:
                i += 1
    return repeats
