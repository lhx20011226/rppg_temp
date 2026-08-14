"""
core/pipeline.py
=================
端到端处理：每帧 RGB 均值序列 -> PPG 波形 (GenR5) -> 切 cycle -> IBI/HR -> BP 特征。

整合前面复刻的模块：
    roi.gen_ppg        : GenR5 (zscore -> 2R-G-B -> butter 带通)
    hr.estimate_hr_snr : CalHRSNR_core (FFT 峰值 HR + 实数 SNR)
    bp.*               : 74维特征 -> 窗口/组 -> 真实模型

IBI 缩放：GenPythonPPGIBI 末尾 `*param_9 = *(double*)(param_1+0x10) * 10.0`
即 IBI 序列整体 ×10 输出。本模块对外给出 (ibi_raw, ibi_scaled) 与聚合 HR。
"""

import numpy as np
from scipy.signal import find_peaks

from . import roi, hr, bp


def build_rgb_series(frames_rgb):
    """frames_rgb: list of (R,G,B)。返回 (R_arr, G_arr, B_arr)。"""
    R = np.array([c[0] for c in frames_rgb], float)
    G = np.array([c[1] for c in frames_rgb], float)
    B = np.array([c[2] for c in frames_rgb], float)
    return R, G, B


def segment_cycles(ppg, fs, min_peak_dist=None):
    """
    用波谷（PPG 下行过零点/局部极小）切分 cycle。
    返回 list[ndarray]，每个为一个 cycle 波形（用于 BP 特征）。
    同时返回 list[float] 每个 cycle 的 IBI（秒，原始）。
    """
    if min_peak_dist is None:
        min_peak_dist = int(fs * 0.25)   # 最快 ~240bpm
    # PPG 波谷 = 信号负向局部极小
    # 由于已带通，直接取 -ppg 的峰值即为波谷
    neg = -ppg
    peaks, props = find_peaks(neg, distance=min_peak_dist, prominence=np.std(neg) * 0.3)
    cycles = []
    ibis = []
    if len(peaks) < 2:
        return cycles, ibis
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        if b - a < 4:
            continue
        cycles.append(ppg[a:b + 1])
        ibis.append((b - a) / fs)
    return cycles, ibis


def process(frames_rgb, fs, height_cm=None):
    """
    参数
    ----
    frames_rgb : list of (R,G,B) 每帧均值
    fs         : 采样率 Hz（真实摄像头帧率）
    height_cm  : 血压模型身高分桶

    返回
    ----
    dict:
        ppg           : ndarray 带通 PPG 波形
        hr_bpm        : float  FFT 峰值 HR
        snr           : float  实数 SNR
        ibi_raw       : list[float] 每个 cycle IBI（秒）
        ibi_scaled    : list[float] IBI ×10（与 GenPythonPPGIBI 一致）
        hr_from_ibi   : float  由 IBI 均值反算的 HR（60/mean(ibi)）
        bp            : dict   血压模型输出（可能含 error）
        n_cycles      : int
    """
    R, G, B = build_rgb_series(frames_rgb)
    if len(R) < 16:
        return {"error": "not enough frames"}

    ppg = roi.gen_ppg(R, G, B, fs)
    hr_bpm, snr = hr.estimate_hr_snr(ppg, fs)

    cycles, ibis = segment_cycles(ppg, fs)
    ibi_scaled = [x * 10.0 for x in ibis]
    hr_from_ibi = (60.0 / np.mean(ibis)) if ibis else -1.0

    bp_out = {"error": "no cycles"}
    if cycles:
        flag_w, wf, n_valid = bp.get_window_feature(cycles, min_n_valid_cycles=10)
        if n_valid >= 10 and wf:
            gflag, gf = bp.get_window_group_feature([[wf], [wf], [wf]])
            if gflag == 0:
                try:
                    bp_out = bp.BPModel().predict(gf, height_cm=height_cm)
                except Exception as e:
                    bp_out = {"error": f"bp predict: {e}"}

    return {
        "ppg": ppg,
        "hr_bpm": hr_bpm,
        "snr": snr,
        "ibi_raw": ibis,
        "ibi_scaled": ibi_scaled,
        "hr_from_ibi": hr_from_ibi,
        "bp": bp_out,
        "n_cycles": len(cycles),
    }
