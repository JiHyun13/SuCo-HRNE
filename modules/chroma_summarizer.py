import numpy as np

def summarize_chroma_features(chroma_feat, method='segment_mean_std', num_segments=5):
    print("크로마 요약 벡터 생성 시작 (method:", method, ")")

    if chroma_feat is None or len(chroma_feat) == 0:
        return np.zeros((num_segments, 24)) if 'mean_std' in method else np.zeros((num_segments, 12))

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

    elif method == 'segment_mean_std':
        time_len = chroma_feat.shape[0]
        segment_len = time_len // num_segments
        summaries = []

        for i in range(num_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < num_segments - 1 else time_len
            segment = chroma_feat[start:end]

            mean_vec = np.mean(segment, axis=0)
            std_vec = np.std(segment, axis=0)

            combined_vec = np.concatenate([mean_vec, std_vec])  # (12 + 12,) → total 24
            summaries.append(combined_vec)

        return np.array(summaries)

    else:
        raise ValueError("Unknown method for summarizing chroma features.")
