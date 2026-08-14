"""
core/bp.py
==========
血压（BP）前端：PPG 波形 -> 切 cycle -> 74 维形态学特征 -> 窗口特征 ->
组特征 -> 真实模型推理。

本模块整合 extracted/repro_rppg_bp.py 与 extracted/bp_inference.py 中【已验证正确】的
纯函数（从 .pyc 逆向得到），作为新项目 rppg_replica 的干净 BP 前端。
不引入任何旧的摄像头/GUI 代码。

调用链（逆向后端 Calculate_PPG_IBI -> extract_ppg_features -> get_window_feature
-> get_window_group_feature -> dm_bp_api）：
    PPG cycles  ->  74 维/cycle  ->  window_feature(74)  ->  group_feature(58)
                ->  BPModel.predict(group_feature, height_cm)
"""

import os
import numpy as np
from scipy.interpolate import CubicSpline

# 与 extracted/features_74.py 一致的 74 维定义（单周期特征）
FEATURES_DELETED = [70, 71, 67, 57, 56, 6, 8, 37, 38, 39, 40, 41, 66, 68, 69, 52]

# 重采样目标长度（原 SDK 内部 900Hz 表示一个周期）
_RESAMPLE_FS = 900
_RESAMPLE_POINTS = 900


def _synth_not_used():
    pass


def extract_single_cycle(y, fs=_RESAMPLE_FS):
    """从单个 PPG cycle（首=左谷，末=右谷）提取 74 维特征。移植自 extract_ppg_features.pyc。"""
    n = len(y)
    valley_left, valley_right = 0, n - 1
    x = np.arange(n, dtype=float)
    peak = int(np.argmax(y[valley_left:valley_right])) + valley_left

    deriv = np.gradient(y, x)
    apg = np.gradient(deriv, x)

    f = [0.0] * 74
    f[0] = (valley_right - valley_left) / fs
    f[1] = (peak - valley_left) / fs
    f[2] = (valley_right - peak) / fs
    f[3] = f[1] / f[0] if f[0] else 0.0
    f[32] = (y[peak] - y[valley_left]) / (x[peak] - x[valley_left]) if (x[peak] - x[valley_left]) else 0.0
    f[33] = (y[valley_right] - y[peak]) / (x[valley_right] - x[peak]) if (x[valley_right] - x[peak]) else 0.0
    f[34] = float(np.max(deriv))
    f[22] = np.trapz(y[valley_left:peak + 1], x[valley_left:peak + 1])
    f[23] = np.trapz(y[peak:valley_right + 1], x[peak:valley_right + 1])
    f[24] = f[22] + f[23]
    a_pos = valley_left + int(np.argmax(apg[valley_left:peak]))
    b_candidates = np.where(apg[valley_left:peak] < 0)[0]
    b_pos = (valley_left + b_candidates[np.argmin(apg[valley_left:peak][b_candidates])]) if len(b_candidates) else peak
    dic_candidates = np.where(y[peak:valley_right] >= 0.5 * y[peak])[0]
    e_pos = peak + (dic_candidates[-1] if len(dic_candidates) else 0)
    f[36] = apg[a_pos]
    f[37] = apg[b_pos]
    f[38] = apg[e_pos]
    f[39] = apg[b_pos] / apg[a_pos] if apg[a_pos] else 0.0
    f[40] = apg[e_pos] / apg[a_pos] if apg[a_pos] else 0.0
    f[41] = f[39] - f[40]
    f[42] = y[e_pos] / y[peak] if y[peak] else 0.0
    return f


def get_window_feature(ppg_cycles_list, min_n_valid_cycles=10):
    """
    输入: list of PPG cycles (每个为 ndarray，首=左谷末=右谷)。
    输出: (flag, window_feature(74维, 原索引), n_valid)。
    """
    if len(ppg_cycles_list) < min_n_valid_cycles:
        return 1, [], 0
    feats = []
    for cyc in ppg_cycles_list:
        cyc = np.asarray(cyc, float)
        if len(cyc) < 4:
            continue
        xs = np.arange(len(cyc))
        cs = CubicSpline(xs, cyc)
        new_x = np.linspace(0, len(cyc) - 1, _RESAMPLE_POINTS)
        new_y = cs(new_x)
        f = extract_single_cycle(new_y, fs=_RESAMPLE_FS)
        if np.any(np.isnan(f)):
            continue
        feats.append(f)
    n_valid = len(feats)
    if n_valid < max(1, min_n_valid_cycles - 2):
        return 2, [], n_valid
    mat = np.array(feats)
    window_feature = list(np.mean(mat, axis=0))
    return 0, window_feature, n_valid


def get_window_group_feature(window_group_list, min_n_groups=1):
    """输入: list of windows（每个 window 是 74 维 list）。输出: (flag, group_feature)。"""
    if len(window_group_list) < min_n_groups:
        return 3, []
    grp = np.array([np.mean(np.array(w), axis=0) for w in window_group_list])
    if len(grp) >= 3:
        from scipy.spatial.distance import pdist, squareform
        d = squareform(pdist(grp, metric="seuclidean"))
        delegate = int(np.argmin(np.mean(d, axis=1)))
        similar = int(np.argsort(d[delegate])[1])
        group_feature = list(0.66 * grp[[delegate, similar]].mean(0) + 0.34 * grp.mean(0))
    else:
        group_feature = list(grp.mean(0))
    return 0, group_feature


# 默认血压模型根目录：优先 rppg_replica/bp_models_full（已随仓库提供），
# 否则回退到 ../extracted/bp_models_full（旧逆向产物目录）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BP_ROOT = os.path.join(_HERE, "..", "bp_models_full",
                                 "dmi_b220818_d220919_v220428", "bp_models")
if not os.path.isdir(_DEFAULT_BP_ROOT):
    _DEFAULT_BP_ROOT = os.path.join(_HERE, "..", "..", "extracted", "bp_models_full",
                                     "dmi_b220818_d220919_v220428", "bp_models")


class BPModel:
    """包装真实血压模型（来自 bp_inference_local.py，随 rppg_replica 自带）。"""

    def __init__(self, bp_root=None):
        from bp_inference_local import BPModel as _BP
        root = bp_root or _DEFAULT_BP_ROOT
        self._m = _BP(root)

    def predict(self, group_feature, height_cm=None):
        return self._m.predict(group_feature, height_cm=height_cm)
