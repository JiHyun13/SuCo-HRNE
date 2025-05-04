#특징 추출

import librosa

def extract_chroma(audio_path):
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    return chroma.T  # shape: (time, 12)
