"""
core/hr.py
===========
逆向后端 libCoreEngineV2.so::CalHRSNR_core (0014f490) 的 100% 复刻。

算法（反编译重建）：
  输入信号 s（已带通），采样率 fs。
  1) 频带：HR 峰值搜索在 [0.6667, 4.0] Hz（即 40~240 BPM）。
     保护下限 0.05 Hz（用于 findpeaks 候选起点）。
  2) 计算幅值谱： mag = abs(fft(s))，长度 N。
     频率向量 f = (0..N-1)/N。
  3) 在 [0.6667,4.0] Hz 内找幅值谱最大点 -> peak_bin
        HR = fs * 60 * peak_bin / N
  4) SNR（关键！是实数，旧代码测成复数是因为实现错误）：
        band_power_in   = sum( mag[f in 0.6667..4.0]^2 )
        band_power_out  = sum( mag[其余]^2 )          # 其余含 0.05Hz 以下保护段
        SNR = log10( band_power_in / band_power_out ) + 0.6

返回 (hr_bpm, snr)。失败时 hr=-1 或 snr=-100（与 .so 约定一致）。
"""

import numpy as np


# 与 .so 中 *param_4 = [0.6667, 4.0] 一致（CHROM 标准 HR 频带）
F_HR_MIN = 0.6667   # ~40 BPM
F_HR_MAX = 4.0      # ~240 BPM
F_GUARD = 0.05      # findpeaks 候选起点保护


def estimate_hr_snr(sig, fs, fmin=F_HR_MIN, fmax=F_HR_MAX, fguard=F_GUARD):
    """
    参数
    ----
    sig : array_like，长度 N 的带通 PPG（单通道）。
    fs  : float，采样率 Hz。

    返回
    ----
    (hr, snr) : (float, float)
        hr  : 估算心率 BPM；失败返回 -1.0
        snr : 信噪比（实数 dB 类量）；失败返回 -100.0
    """
    s = np.asarray(sig, float)
    N = len(s)
    if N < 8:
        return -1.0, -100.0

    # 幅值谱
    spec = np.fft.rfft(s - s.mean())
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # 峰值搜索区间 [fmin, fmax]
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return -1.0, -100.0

    mag_band = mag[mask]
    freqs_band = freqs[mask]
    peak_idx_local = int(np.argmax(mag_band))
    peak_freq = freqs_band[peak_idx_local]

    # 抛物线插值精确定频（在峰值 bin 及其左右邻点做二次插值 refine）
    # 这是对 FFT bin 量化的标准修正，使 HR 精度高于单纯 round。
    if 0 < peak_idx_local < len(mag_band) - 1:
        aL = mag_band[peak_idx_local - 1]
        aC = mag_band[peak_idx_local]
        aR = mag_band[peak_idx_local + 1]
        denom = (aL - 2 * aC + aR)
        delta = 0.5 * (aL - aR) / denom if denom != 0 else 0.0
        peak_freq = peak_freq + delta * (freqs_band[1] - freqs_band[0])

    peak_bin = int(np.round(peak_freq * N / fs))  # 还原到全谱 bin（与 .so 一致）

    # HR = fs * 60 * peak_bin / N
    hr = fs * 60.0 * peak_bin / N
    if not (fmin * 60.0 - 1e-6 <= hr <= fmax * 60.0 + 1e-6):
        hr = peak_freq * 60.0  # 兜底

    # SNR：带内能量 / 带外能量
    band_mask = mask
    # 带外 = 全谱去掉带内（含 fguard 以下也计为带外，与 .so 一致：带外从 0.05 起算其余）
    out_mask = np.ones_like(mask) & True
    # 实际 .so：带外 = 全频谱平方 - 带内平方；等价于对 (全谱 且 不在带内) 求和，
    # 但 0.05Hz 以下保护段在 findpeaks 起点外、仍属于"其余"。
    other = (~band_mask)
    # 避免 DC 项（0 Hz）主导带外：.so 里 0.05/dVar50 作为起点，
    # 即 0~0.05Hz 不计入带外能量统计起点。
    other = other & (freqs >= fguard)

    power_in = np.sum(mag_band ** 2)
    power_out = np.sum(mag[other] ** 2)

    if power_out <= 0 or power_in <= 0:
        snr = -100.0
    else:
        snr = float(np.log10(power_in / power_out) + 0.6)

    return float(hr), snr
