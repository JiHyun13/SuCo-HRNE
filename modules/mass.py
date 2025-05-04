import numpy as np
from scipy.fftpack import fft, ifft

def sliding_dot_product(q, t):
    m = len(q)
    n = len(t)
    size = 2 ** int(np.ceil(np.log2(m + n)))
    q = np.flipud(q)
    q = np.pad(q, (0, size - m), 'constant')
    t = np.pad(t, (0, size - n), 'constant')
    Q = fft(q)
    T = fft(t)
    return ifft(Q * T).real

def calculate_mean_std(ts, m):
    cumsum = np.cumsum(np.insert(ts, 0, 0))
    cumsum2 = np.cumsum(np.insert(ts**2, 0, 0))
    sum_m = cumsum[m:] - cumsum[:-m]
    sum2_m = cumsum2[m:] - cumsum2[:-m]
    mean = sum_m / m
    std = np.sqrt(np.maximum((sum2_m / m) - (mean ** 2), 0)) + 1e-8
    return mean, std

def mass(query, ts, m, ts_mean, ts_std):
    query = (query - np.mean(query)) / (np.std(query) + 1e-8)
    dot = sliding_dot_product(query, ts)
    dot = dot[m - 1 : m - 1 + len(ts_mean)]

    raw = 2 * (m - (dot - m * np.mean(query) * ts_mean) / (np.std(query) * ts_std + 1e-8))
    dist_profile = np.sqrt(np.maximum(raw, 0))  # 음수 방지
    return dist_profile

def compute_mass_sdm(feature_seq, subseq_len=20):
    time_len, dim = feature_seq.shape
    sdm = np.zeros((time_len - subseq_len + 1, time_len - subseq_len + 1))

    for d in range(dim):
        ts = feature_seq[:, d]
        ts_mean, ts_std = calculate_mean_std(ts, subseq_len)

        for i in range(time_len - subseq_len + 1):
            query = ts[i : i + subseq_len]
            dp = mass(query, ts, subseq_len, ts_mean, ts_std)
            sdm[i, :] += dp ** 2

    sdm = np.sqrt(np.maximum(sdm, 0))  # 안전한 루트
    return sdm
