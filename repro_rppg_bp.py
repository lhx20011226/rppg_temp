#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小复现示例：rPPG 测心率 + 血压（基于 offline_engine-release.zip 逆向得到的调用链）

调用链（逆向结论）:
  人脸 RGB 视频帧
    -> libCoreEngineV2.so::Calculate_PPG_IBI  (RGB -> PPG 波形, 切成 cycle)
    -> [Python] extract_ppg_features.extract_ppg_features_from_single_ppg_cycle
          (每个 PPG cycle 提取 74 维形态学特征)
    -> get_window_feature   (一个 35s 窗口 -> 窗口特征)
    -> get_window_group_feature (1~3 个窗口 -> 组特征 window_group_feature)
    -> dm_bp_api.get_bp(window_group_feature, ...)  -> 血压 (lbp/hbp)
  心率：PPG cycle 的 IBI(心动周期) -> HR = 60/IBI_avg

说明：
  * 本示例前端（PPG->74维特征->窗口->组）为纯 numpy/scipy 实现，可直接运行。
  * 血压模型 model.pkl 依赖 `imbalanced-ensemble` + scikit-learn<1.2（与训练时一致），
    见文末“运行依赖”。make_bp() 给出一个最小可替换骨架，加载真实模型后即可得数值。
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# 0) 模拟 rPPG 输出：生成一段 PPG 波形（含 a/b/e 波 + 重搏切迹）
#    真实场景里这一段由 libCoreEngineV2.so::Calculate_PPG_IBI 从人脸 RGB 得到
# ---------------------------------------------------------------------------
def synth_ppg_cycle(fs=900, hr_bpm=75.0, noise=0.01):
    """返回一个 PPG cycle (ndarray)，首=左谷 末=右谷，中间一个主峰。"""
    dur = 60.0 / hr_bpm                 # 周期时长(s)
    n = int(dur * fs)
    t = np.linspace(0, dur, n)
    # 主收缩波 (高斯峰) + 重搏波 (dicrotic，靠后的小峰)
    sys_peak = 0.18 * dur
    dic_peak = 0.55 * dur
    y = (np.exp(-((t - sys_peak) ** 2) / (2 * (0.06 * dur) ** 2)) +
         0.35 * np.exp(-((t - dic_peak) ** 2) / (2 * (0.05 * dur) ** 2)))
    y -= y.min(); y /= y.max()
    y += noise * np.random.randn(n)
    return y.astype(float)


def synth_ppg_cycles(n_cycles=60, fs=900, hr_bpm=75.0):
    """返回一个 list，每个元素是一个 PPG cycle (ndarray)。"""
    return [synth_ppg_cycle(fs, hr_bpm) for _ in range(n_cycles)]


# ---------------------------------------------------------------------------
# 1) 从单个 PPG cycle 提取 74 维特征 (移植自 extract_ppg_features.pyc)
#    这里仅实现运行示例所必需的核心子集，完整 74 维见 features_74.py
# ---------------------------------------------------------------------------
def extract_single_cycle(y, fs=900):
    """返回 74 维特征向量 (list)。y: 单周期 PPG, 首=左谷 末=右谷。"""
    n = len(y)
    valley_left, valley_right = 0, n - 1
    x = np.arange(n, dtype=float)
    peak = int(np.argmax(y[valley_left:valley_right])) + valley_left

    # 一阶/二阶导数
    deriv = np.gradient(y, x)
    apg = np.gradient(deriv, x)

    f = [0.0] * 74
    f[0] = (valley_right - valley_left) / fs                       # 周期时长
    f[1] = (peak - valley_left) / fs                              # 收缩期时长
    f[2] = (valley_right - peak) / fs                            # 舒张期时长
    f[3] = f[1] / f[0] if f[0] else 0.0                          # 收缩期占比
    f[32] = (y[peak] - y[valley_left]) / (x[peak] - x[valley_left]) if (x[peak]-x[valley_left]) else 0.0
    f[33] = (y[valley_right] - y[peak]) / (x[valley_right] - x[peak]) if (x[valley_right]-x[peak]) else 0.0
    f[34] = float(np.max(deriv))                                 # 最大上升斜率
    # 面积
    f[22] = np.trapz(y[valley_left:peak + 1], x[valley_left:peak + 1])   # 收缩期面积
    f[23] = np.trapz(y[peak:valley_right + 1], x[peak:valley_right + 1]) # 舒张期面积
    f[24] = f[22] + f[23]
    # apg 波幅 (a: 导数正峰, b: apg 的负谷靠近 a 之后, e: 重搏附近)
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
    f[42] = y[e_pos] / y[peak] if y[peak] else 0.0                 # AI 幅度比
    return f


# ---------------------------------------------------------------------------
# 2) 窗口特征 / 组特征 (移植自 extract_ppg_features.pyc)
# ---------------------------------------------------------------------------
FEATURES_DELETED = [70, 71, 67, 57, 56, 6, 8, 37, 38, 39, 40, 41, 66, 68, 69, 52]

def get_window_feature(ppg_cycles_list, min_n_valid_cycles=10):
    """输入: list of PPG cycles。输出: (flag, window_feature, n_valid)。
    返回 74 维 (保留原索引位置; 被删除索引填0)。模型 features.txt 用原始 f 编号索引。"""
    if len(ppg_cycles_list) < min_n_valid_cycles:
        return 1, [], 0
    feats = []
    for cyc in ppg_cycles_list:
        cyc = np.asarray(cyc, float)
        xs = np.arange(len(cyc))
        cs = CubicSpline(xs, cyc)
        new_x = np.linspace(0, len(cyc) - 1, int(900 / 300.0 * (len(cyc) - 1)) + 1)
        new_y = cs(new_x)
        f = extract_single_cycle(new_y, fs=900)
        if np.any(np.isnan(f)):
            continue
        feats.append(f)
    n_valid = len(feats)
    if n_valid < min_n_valid_cycles - 2:
        return 2, [], n_valid
    mat = np.array(feats)                      # (n_valid, 74)
    window_feature = list(np.mean(mat, axis=0))  # 74 维, 保留原索引
    return 0, window_feature, n_valid


def get_window_group_feature(window_group_list, min_n_groups=1, max_n_groups=9):
    """输入: list of windows (每个 window 是 list)。输出: (flag, group_feature)。"""
    if len(window_group_list) < min_n_groups:
        return 3, []
    grp = np.array([np.mean(np.array(w), axis=0) for w in window_group_list])
    if len(grp) >= 3:
        # 取最相似的两组加权 (0.66/0.34)，与源码一致
        from scipy.spatial.distance import pdist, squareform
        d = squareform(pdist(grp, metric="seuclidean"))
        delegate = int(np.argmin(np.mean(d, axis=1)))
        similar = int(np.argsort(d[delegate])[1])
        group_feature = list(0.66 * grp[[delegate, similar]].mean(0) + 0.34 * grp.mean(0))
    else:
        group_feature = list(grp.mean(0))
    return 0, group_feature


# ---------------------------------------------------------------------------
# 3) 心率：PPG cycle -> IBI -> HR
# ---------------------------------------------------------------------------
def estimate_hr(ppg_cycles_list, fs=900):
    ibis = [len(c) / fs for c in ppg_cycles_list]     # 每个 cycle 的 IBI(s)
    ibi = np.mean(ibis)
    return 60.0 / ibi


# ---------------------------------------------------------------------------
# 4) 血压：calc_bp 最小骨架（加载真实模型）
#    模型: runtime.zip -> dmi_.../bp_models/models_ch{身高}_ie/<hash>/model.pkl + features.txt
#    注意: 真实 calc_bp 会把 cl70~ch160 共 24 个桶模型全部预测再 stacking 融合；
#          这里给出“单桶”最小版本，演示“特征 -> 模型 -> 血压等级”的调用方式。
# ---------------------------------------------------------------------------
def make_bp(window_group_feature, model_pkl, features_txt):
    """
    window_group_feature: get_window_group_feature 的输出 (list, 维度 = 74 - 删除项)
    model_pkl: 真实 model.pkl 路径
    features_txt: 对应 features.txt 路径 (模型实际使用的特征编号列表, 如 ['f43','f75',...])
    返回: 模型 predict 的原始输出 (分类器为 0/1 高血压等级, 回归器为连续值)
    """
    import pickle, ast
    with open(features_txt) as fh:
        used = ast.literal_eval(fh.read().strip())        # e.g. ['f43','f75',...]
    idx = [int(f[1:]) for f in used]                     # f43 -> 43
    x = np.array([window_group_feature[i] for i in idx]).reshape(1, -1)
    with open(model_pkl, 'rb') as fh:
        m = pickle.load(fh, encoding='latin1')           # 需 imbalanced-ensemble + sklearn<1.2
    return m.predict(x)


# ---------------------------------------------------------------------------
# 5) 跑一遍
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fs = 900
    # (A) 模拟 rPPG 输出：60 个 PPG cycle
    cycles = synth_ppg_cycles(n_cycles=60, fs=fs, hr_bpm=75.0)

    # (B) 心率
    hr = estimate_hr(cycles, fs)
    print(f"[心率] 估计 HR ≈ {hr:.1f} bpm")

    # (C) 窗口特征（实际需多个 35s 窗口，这里用全部 60 cycle 作为一个窗口演示）
    flag, wf, nvalid = get_window_feature(cycles, min_n_valid_cycles=10)
    print(f"[窗口特征] flag={flag}, n_valid_cycles={nvalid}, 特征维度={len(wf)}")

    # (D) 组特征（真实需 1~3 个窗口；这里用 3 份同一窗口模拟 3 窗一组）
    gflag, gf = get_window_group_feature([[wf], [wf], [wf]])
    print(f"[组特征] flag={gflag}, 维度={len(gf)}")

    # (E) 血压（真实模型, venv 内需 scikit-learn==1.1.3 + imbalanced-ensemble==0.1.1 + joblib + lightgbm）
    #     血压模型返回的是真实 mmHg 值 (舒张压 lbp / 收缩压 hbp)，不是风险分数。
    #     组特征需补元特征 f74~f79 (元特征: HR/年龄/性别等)，回归模型才拿到真实信号。
    try:
        from bp_inference import BPModel
        bp = BPModel()
        # 注入元特征: f74=HR(bpm), f75=年龄, f79=gender(1=男)
        gf_ext = list(gf) + [0.0] * (80 - len(gf))
        gf_ext[74] = round(hr, 2)      # 平均心率
        gf_ext[75] = 30.0              # 年龄(岁) — 演示用, 真实应填被测者年龄
        gf_ext[79] = 1.0               # 性别 1=男
        res = bp.predict(gf_ext, height_cm=170, age_1_6=3, gender=1)
        if "error" in res:
            print(f"[血压] 推理失败: {res['error']}")
        else:
            print(f"[血压] 收缩压={res['hbp']} mmHg  舒张压={res['lbp']} mmHg  "
                  f"(融合原始值 hbp_raw={res['hbp_raw']}, lbp_raw={res['lbp_raw']})")
    except Exception as e:
        print(f"[血压] 未运行: {type(e).__name__}: {e}")
