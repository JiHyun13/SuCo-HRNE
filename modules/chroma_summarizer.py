import numpy as np
import librosa

def summarize_chroma_features(chroma_feat, method='segment_mean_std', num_segments=5):
    if chroma_feat is None or len(chroma_feat) == 0:
        feature_dim = 12  # 기본값
        return np.zeros((num_segments, feature_dim * 2)) if 'mean_std' in method else np.zeros((num_segments, feature_dim))

    feature_dim = chroma_feat.shape[1]  # 입력 벡터의 실제 차원

    if method == 'mean':
        summary = np.mean(chroma_feat, axis=0, keepdims=True)
        return summary

    elif method == 'segment':
        time_len = chroma_feat.shape[0]
        segment_len = time_len // num_segments
        segments = []

        for i in range(num_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < num_segments - 1 else time_len
            segment = chroma_feat[start:end]
            mean_vec = np.mean(segment, axis=0)
            segments.append(mean_vec)

        summary = np.array(segments)
        return summary

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
            combined_vec = np.concatenate([mean_vec, std_vec])
            summaries.append(combined_vec)

        summary = np.array(summaries)
        return summary

    else:
        raise ValueError("Unknown method for summarizing chroma features.")


def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape: (n_mfcc, T)
    return mfcc.T  # shape: (T, n_mfcc)
