"""
core/ppg.py
============
逆向后端 libCoreEngineV2.so::GenR5 (001f83cc) 的 100% 复刻。

GenR5 算法流程（反编译重建）：
  输入：R/G/B 三通道时间序列（每帧一个均值点），采样率 fs
  1) 重组为 N×3 矩阵，逐通道 zscore 标准化  x' = (x - mean(x)) / std(x)
     （std 使用 MATLAB var(x,1) 即总体标准差；等效 numpy std(ddof=0)）
  2) CHROM 投影： s = 2*R' - G' - B'        （反编译：`((a+a)-b)-c`）
  3) 零相移带通（filtfilt）：
         Wn = [0.3, 3.5] / (fs * 0.5)        （归一化到 Nyquist）
         butter 2 阶 → filtfilt
     得到输出 PPG 波形（用于 IBI 切 cycle、BP 特征）。
  4) 内部还有 60 点滑窗 HR（h_filtfilt 再带通 + 峰值），但实时 HR 实际由
     CalHRSNR_core 路径给出。本模块同时导出 gen_ppg()（波形）与
     sliding_window_hr()（滑窗 HR，与 GenR5 内部一致）。

IBI 缩放：GenPythonPPGIBI 末尾 `*param_9 = *(double*)(param_1+0x10) * 10.0`。
即最终 IBI 序列整体乘以 10（单位由调用方决定）。本模块在 gen_ppg 之外不处理 IBI，
IBI 由上层按此规则缩放。
"""

import numpy as np
from scipy.signal import butter, filtfilt


def _zscore_cols(mat):
    """逐列 zscore（总体 std, ddof=0），与 MATLAB zscore 一致。"""
    m = mat.mean(axis=0, keepdims=True)
    s = mat.std(axis=0, ddof=0, keepdims=True)
    s[s == 0] = 1.0
    return (mat - m) / s


def gen_ppg(R, G, B, fs):
    """
    生成 PPG 波形（与 GenR5 步骤 1-3 一致）。

    参数
    ----
    R, G, B : array_like，长度 N 的每帧均值（已按 0~255 或任意尺度均可，因后续 zscore）。
    fs      : float，采样率（Hz）。

    返回
    ----
    ppg : ndarray，长度 N，带通后的 CHROM 投影信号。
    """
    R = np.asarray(R, float)
    G = np.asarray(G, float)
    B = np.asarray(B, float)
    n = min(len(R), len(G), len(B))
    rgb = np.column_stack([R[:n], G[:n], B[:n]])

    # 1) zscore 逐通道
    z = _zscore_cols(rgb)

    # 2) CHROM 投影 2R - G - B
    proj = 2.0 * z[:, 0] - z[:, 1] - z[:, 2]

    # 3) 零相移带通 [0.3, 3.5] Hz
    nyq = fs * 0.5
    low = 0.3 / nyq
    high = 3.5 / nyq
    low = min(max(low, 1e-4), 1.0 - 1e-4)
    high = min(max(high, low + 1e-4), 1.0 - 1e-4)
    b, a = butter(2, [low, high], btype='band')
    ppg = filtfilt(b, a, proj)
    return ppg


def sliding_window_hr(ppg, fs, win=60):
    """
    复刻 GenR5 内部 60 点滑窗 HR：每个滑窗内再 filtfilt（同带通）+ zscore，
    然后用与 CalHRSNR_core 一致的 FFT 峰值法求 HR（见 core/hr.py）。
    这里直接调用核心 HR 函数。

    返回
    ----
    hrs : list[float]，每个滑窗的 HR（BPM）；不足窗口返回空。
    """
    from .hr import estimate_hr_snr
    hrs = []
    n = len(ppg)
    if n < win:
        return hrs
    for i in range(0, n - win + 1, 1):
        seg = ppg[i:i + win]
        nyq = fs * 0.5
        low = 0.3 / nyq
        high = 3.5 / nyq
        low = min(max(low, 1e-4), 1.0 - 1e-4)
        high = min(max(high, low + 1e-4), 1.0 - 1e-4)
        b, a = butter(2, [low, high], btype='band')
        seg = filtfilt(b, a, seg)
        seg = (seg - seg.mean()) / (seg.std(ddof=0) + 1e-12)
        hr, _ = estimate_hr_snr(seg, fs)
        if hr > 0:
            hrs.append(hr)
    return hrs
