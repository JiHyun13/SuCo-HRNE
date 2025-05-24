import numpy as np
import librosa

def summarize_features(feat, method='segment_mean_std', num_segments=10):
    if feat is None or len(feat) == 0:
        feature_dim = 12  # 기본값
        return np.zeros((num_segments, feature_dim * 2)) if 'mean_std' in method else np.zeros((num_segments, feature_dim))

    feature_dim = feat.shape[1]  # 입력 벡터의 실제 차원

    if method == 'mean': #단순 평균
        summary = np.mean(feat, axis=0, keepdims=True)
        return summary

    elif method == 'segment': #num_segment 개수대로 구간 만들어 구간 별 평균 반환
        time_len = feat.shape[0]
        segment_len = time_len // num_segments
        segments = []

        for i in range(num_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < num_segments - 1 else time_len
            segment = feat[start:end]
            mean_vec = np.mean(segment, axis=0)
            segments.append(mean_vec)

        summary = np.array(segments)
        return summary

    elif method == 'segment_mean_std': #num_segment 개수대로 구간 만들어 구간 별 평균&표준편차 반환
        time_len = feat.shape[0]
        segment_len = time_len // num_segments
        summaries = []

        for i in range(num_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < num_segments - 1 else time_len
            segment = feat[start:end]
            mean_vec = np.mean(segment, axis=0)
            std_vec = np.std(segment, axis=0)
            combined_vec = np.concatenate([mean_vec, std_vec])
            summaries.append(combined_vec)

        summary = np.array(summaries)
        return summary

    else:
        raise ValueError("method가 잘못됨")


def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape: (n_mfcc, T)
    return mfcc.T  # shape: (T, n_mfcc)
