import librosa
import numpy as np

def extract_combined_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    # 1. MFCC (음색)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # (13, T)

    # 2. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)  # (7, T)

    # 3. Chroma (조화, 코드)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)  # (12, T)

    # 4. Tonnetz (조성, 하모니)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)  # (6, T)

    # 5. RMS (볼륨)
    rms = librosa.feature.rms(y=y)  # (1, T)

    # 6. 길이를 맞추기 위해 최소 프레임 수 결정
    min_frames = min(mfcc.shape[1], contrast.shape[1], chroma.shape[1], tonnetz.shape[1], rms.shape[1])

    # 7. 모두 자르고 수직 스택
    features = np.vstack([
        mfcc[:, :min_frames],
        contrast[:, :min_frames],
        chroma[:, :min_frames],
        tonnetz[:, :min_frames],
        rms[:, :min_frames]
    ])  # shape: (총 39, T)

    return features.T  # shape: (T, 39)
