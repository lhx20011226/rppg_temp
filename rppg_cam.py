#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rppg_cam.py — 调用本地摄像头，rPPG 测人脸心率 + 血压

完整调用链 (对齐 offline_engine-release.zip 逆向结论):
  摄像头帧
   -> mediapipe FaceMesh 取人脸 ROI (前额/脸颊)
   -> 每帧 ROI 的 RGB 平均 -> RGB 时间序列
   -> CHROM 算法去运动伪影 -> PPG 信号        (对应 libCoreEngineV2.so::Calculate_PPG_IBI)
   -> 心率: PPG 带通滤波 + FFT 峰值           (对应 CalculateHRVIndex_RGB 的 HR 输出)
   -> 切 cycle + 74维特征 + 窗口/组特征        (对应 extract_ppg_features)
   -> bp_inference.BPModel.predict             (对应 dm_bp_api.get_bp, 返回真实 mmHg)
   注: 血压模型返回的是真实收缩压/舒张压(mmHg), 不是风险分数。

用法:
    source rppg_venv/bin/activate   (或 venv 里直接运行)
    python rppg_cam.py --seconds 30 --height 170 --gender 1
参数:
    --camera   摄像头索引 (默认0)
    --seconds  采集时长秒 (默认30)
    --height   身高 cm (用于选血压模型桶, 默认170)
    --gender   性别 0=女 1=男 (默认1)
    --fps      期望采集帧率 (默认30)
依赖见 requirements.txt (venv: numpy1.23.5/scipy1.10.1/sklearn1.1.3/
      imbalanced-ensemble0.1.1/opencv-python-headless4.8.1.78/mediapipe0.10.5)
"""
import argparse, time, sys, os
import numpy as np
import cv2

# ---------------- GUI 中文绘制 + 全屏 ----------------
# OpenCV 默认位图字体不含中文, 用 PIL 绘制中文再叠加, 避免显示成问号。
_PIL_OK = False
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    pass


def put_text_cn(frame, text, pos, color=(0, 255, 0), scale=0.7, thickness=2):
    """在 BGR 帧上绘制中文文本 (无中文环境时退化为 cv2 英文)。"""
    if not _PIL_OK:
        cv2.putText(frame, text.encode("ascii", "ignore").decode(),
                    pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("msyh.ttc", int(20 * scale))
    except Exception:
        try:
            font = ImageFont.truetype("simhei.ttf", int(20 * scale))
        except Exception:
            font = ImageFont.load_default()
    draw.text((pos[0], pos[1]), text, fill=(color[2], color[1], color[0]), font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def make_window_fullscreen(name, w, h):
    """创建可全屏/窗口化切换的窗口。

    默认窗口化(用户可自由缩放); 按 F 键 或 双击窗口 切换全屏。
    注册鼠标回调实现双击 -> 全屏/窗口切换 (对齐用户要求的"全屏/窗口化")。"""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, min(w, 1280), min(h, 720))   # 默认窗口化显示
    _fs_state = {"full": False}

    def _toggle():
        _fs_state["full"] = not _fs_state["full"]
        try:
            cv2.setWindowProperty(
                name, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if _fs_state["full"] else cv2.WINDOW_NORMAL)
        except Exception:
            pass

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            _toggle()
    cv2.setMouseCallback(name, _on_mouse)
    # 把 toggle 暴露给调用方 (F 键复用)
    make_window_fullscreen._toggle = _toggle


# ---------------- rPPG 信号提取 (CHROM) ----------------
def chrom_extract(rgb_ts, fs):
    """CHROM: 输入 (N,3) RGB 时序, 返回 PPG 信号 (N,)。

    对齐 libCoreEngineV2.so::c_CalS3CoreHR_RGB_Profile1_RGB_ 的投影系数:
    反编译末尾确认投影为 2*R - G - B (三个输入通道 a/b/c 的组合 = 2a - b - c,
    其中 a=R, b=G, c=B)。这与经典 CHROM 的 2R-G-B 一致。
    注意: 原生在投影前对每个通道做 b_filtfilt / c_filtfilt (零相移带通),
    这里返回时不做带通(带通在调用方 estimate_hr 前统一做 bandpass,
    与调用链 c_CalS3CoreHR -> ... -> CalHRSNR_core 的顺序一致)。"""
    rgb = np.asarray(rgb_ts, float)
    rgb = rgb - np.mean(rgb, axis=0, keepdims=True)  # 去均值
    std = np.std(rgb, axis=0, keepdims=True)
    std[std == 0] = 1.0
    rgb = rgb / std
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    # 对齐原生 2R - G - B 投影
    ppg = 2.0 * R - G - B
    return ppg


def bandpass(sig, fs, lo=0.7, hi=4.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def _hr_snr_one(ppg, fs):
    """对齐 libCoreEngineV2.so::b_CalHRSNR_core 的单段 HR+SNR 计算。

    逆向确认的核心步骤:
      1) 固定 N=4096 点 FFT (rfft), 频率轴 f = k*fs/N;
      2) 功率谱 P = |FFT|^2;
      3) 在 0.7~4.0Hz 频带内用 findpeaks(高度>全局最大*0.3) 找主频峰,
         取最强峰对应频率 -> HR = f_peak*60 (对齐 dVar32 = peak_freq*60*fs);
      4) SNR = log10(主频峰值功率 / 全频带总功率) + 0.6
         (对齐 b_CalHRSNR_core 中段 *param_5 = log10(peak/total)+0.6,
          量纲与实测 SNR≈-0.2~0.3 完全吻合)。
    返回 (hr_or_None, snr)。"""
    from scipy.signal import find_peaks
    ppg = np.asarray(ppg, float)
    ppg = ppg - ppg.mean()
    if len(ppg) < 8:
        return None, -100.0
    N = 4096
    x = np.zeros(N, dtype=float)
    x[:len(ppg)] = ppg
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    P = np.abs(X) ** 2
    band = (freq >= 0.7) & (freq <= 4.0)
    if not np.any(band):
        return None, -100.0
    fb = freq[band]
    Pb = P[band]
    if len(Pb) == 0 or Pb.max() <= 0:
        return None, -100.0
    # findpeaks: 高度 > 全局最大*0.3 (对齐 b_findpeaks 的 minpeakheight=max*0.3)
    thr = float(Pb.max()) * 0.3
    peaks, _ = find_peaks(Pb, height=thr)
    if len(peaks) == 0:
        best = int(np.argmax(Pb))
    else:
        best = int(peaks[np.argmax(Pb[peaks])])
    hr = float(fb[best] * 60.0)
    # SNR: 对齐 log10(peak/total)+0.6
    total = float(Pb.sum())
    peak_p = float(Pb[best])
    if total <= 0 or peak_p <= 0:
        snr = -100.0
    else:
        snr = float(np.log10(peak_p / total) + 0.6)
    if not (40.0 <= hr <= 200.0):
        return None, snr
    return hr, snr


def estimate_hr(ppg, fs):
    """瞬时 HR (对齐 b_CalHRSNR_core 的 4096点FFT+findpeaks)。返回 hr 或 None。"""
    hr, _ = _hr_snr_one(ppg, fs)
    return hr


def calc_hr_snr(ppg, fs):
    """复刻 libCoreEngineV2.so::CalHRSNR_core: HRSNR = log10(主频峰值/全频总功率)+0.6。"""
    _, snr = _hr_snr_one(ppg, fs)
    return snr


# 心率有效性判定 (对齐 libCoreEngineV2.so::CalHRSNR_core + FilterHRUsingPreviousDataRGB)
# 原生逻辑: 每窗算 SNR=log10(peak/total)+0.6。SNR 低(无明显主频)的窗直接作废,
# 不进历史; 仅用 SNR 达标的窗做历史加权得到平滑 HR (b_CalWeightMean)。
# 量纲实测: 干净 PPG SNR≈-0.4~0.3; 噪声大时趋近 -1.8。故门槛取 -0.5:
# 低于它的窗视为"无信号", 不显示 HR、不计入加权。
HR_WINDOW_SEC = 3.0        # 历史窗口采样间隔(秒)
HR_MIN_VALID_SAMPLES = 3   # 历史中"可用"样本最少个数, 低于则判定信号不足
HR_PHYS_LO, HR_PHYS_HI = 40.0, 200.0   # 心率生理合理区间(原生大量 0<HR 检查)
# 单窗 SNR 可用门槛 (log10(peak/total)+0.6 量纲): 低于则不采纳该窗 HR
# 量纲实测: 干净 PPG SNR≈-0.4~0.3; 噪声大时趋近 -1.8。
# 原生 CalHRSNR_core 用 peak/total 判"有无主频", SNR<-0.8(peak/total<0.04)即视为无有效主频。
# 故门槛取 -0.8: 仅采纳确有主频的窗进历史; 低于它的窗(真噪声)明确不显示/不采纳。
HR_SNR_USABLE = -0.8
# 对齐 CalculateHRVIndex_RGB 的 "High HR change" 判定:
# 若历史 HR 序列 max-min > max*0.1, 视为心率突变不可信 -> 标记 unstable。
HIGH_HR_CHANGE_RATIO = 0.1

# ---------------- ROI 几何 (逆向自 libcmtrack.so::TddFa::parseRoiBoxFrom*) ----------------
# 原生每帧产出 2 个 ROI (CmTrackInterface::track @0x24da0 各调用一次):
#   ROI A = parseRoiBoxFromBbox     -> 对齐 AllFace  (整个人脸区域)
#   ROI B = parseRoiBoxFromLandmark -> 对齐 ForeHead (关键点区域, HR 主信号)
# 常量已从 .so 的 .rodata 段提取 (见 ROI_REVERSE_ENGINEERING.md)。
ROI_A_SCALE = 1.58      # parseRoiBoxFromBbox: size = round((W+H)/2 * 1.58)
ROI_A_YOFF  = 0.14      # parseRoiBoxFromBbox: cy = y2 - H*0.5 + size*0.14
# ROI B: 以关键点集质心为中心、半对角距 radius 为半边长的正方形 (见文档 §4)
# 质量门控 (对齐原生 AssertInputData / c_CalS3CoreHR_RGB_Profile1_RGB_):
ROI_MIN_VALID_PIXEL_PER = 0.30   # VaildPixelPer: ROI 内有效(非全黑)像素占比下限
ROI_FRAME_RGB_CHANGE_TH = 0.06    # FaceRGBStdThresIgnore: 相邻帧 ROI 均值 RGB 变化上限(归一化)


# ---------------- 74 维特征 (移植自 extract_ppg_features.pyc) ----------------
# 关键修正: 原生 extract_ppg_features 对每个 PPG cycle 先用 CubicSpline 重采样到
# 900Hz 网格 (repro_rppg_bp.py:107: new_x = linspace(0,len-1, int(900/300*(len-1))+1)),
# 再调用 extract_single_cycle(new_y, fs=900)。fs 是内部固定 900, 不是摄像头真实 fps!
# 之前用真实 fps 传入使所有时间特征错 ~30 倍 -> 74维特征整体错误 -> BP 塌缩到基线。
FEATURE_FS = 900
_FROM_SCIPY = True
try:
    from scipy.interpolate import CubicSpline
except Exception:
    _FROM_SCIPY = False


def _resample_cycle_to_900(cyc):
    """对齐 repro_rppg_bp.py:107: CubicSpline 重采样到 900Hz 相对网格。"""
    cyc = np.asarray(cyc, float)
    xs = np.arange(len(cyc))
    if _FROM_SCIPY and len(cyc) >= 4:
        cs = CubicSpline(xs, cyc)
        new_x = np.linspace(0, len(cyc) - 1, int(FEATURE_FS / 300.0 * (len(cyc) - 1)) + 1)
        return cs(new_x)
    # 退化: 线性重采样
    new_n = max(4, int(FEATURE_FS / 300.0 * (len(cyc) - 1)) + 1)
    return np.interp(np.linspace(0, len(cyc) - 1, new_n), xs, cyc)


def extract_single_cycle(y, fs=900):
    """返回 74 维特征 (对齐 features_74.py 定义)。y: 单周期 PPG, 首=左谷 末=右谷。"""
    y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        return None
    valley_left, valley_right = 0, n - 1
    x = np.arange(n, dtype=float)
    peak = int(np.argmax(y[valley_left:valley_right])) + valley_left
    if peak <= valley_left or peak >= valley_right:
        return None
    try:
        deriv = np.gradient(y, x)
        apg = np.gradient(deriv, x)
    except Exception:
        return None
    f = [0.0] * 74
    # ---- f0~f3 时长类 ----
    f[0] = (valley_right - valley_left) / fs
    f[1] = (peak - valley_left) / fs
    f[2] = (valley_right - peak) / fs
    f[3] = f[1] / f[0] if f[0] else 0.0
    # ---- 关键点定位 (apg 二阶导) ----
    a_pos = valley_left + int(np.argmax(apg[valley_left:peak])) if peak > valley_left else peak
    b_c = np.where(apg[valley_left:peak] < 0)[0]
    b_pos = (valley_left + b_c[np.argmin(apg[valley_left:peak][b_c])]) if len(b_c) else peak
    dc = np.where(y[peak:valley_right] >= 0.5 * y[peak])[0]
    e_pos = peak + (dc[-1] if len(dc) else 0)
    e_pos = min(e_pos, valley_right)
    # ---- f4~f8 apg/位置比 ----
    if (peak - valley_left) > 0:
        f[4] = (a_pos - valley_left) / (peak - valley_left)
    f[5] = (e_pos - valley_left) / f[0] if f[0] else 0.0
    if (peak - valley_left) > 0:
        f[6] = (a_pos - valley_left) / (peak - valley_left)
        f[7] = (b_pos - a_pos) / (peak - valley_left)
    if (valley_right - e_pos) > 0:
        f[8] = (e_pos - e_pos) / (valley_right - e_pos)  # inflection~e_pos 退化, 占位
    # ---- f9~f21 宽度类 (幅值>=thr*peak 的跨度, 单位 s) ----
    for thr, idx0 in [(0.1, 9), (0.25, 10), (0.33, 11), (0.5, 12), (0.66, 13), (0.75, 14)]:
        m = y[valley_left:peak + 1] >= thr * y[peak]
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    for thr, idx0 in [(0.1, 15), (0.25, 16), (0.33, 17), (0.5, 18), (0.66, 19), (0.75, 20)]:
        m = y[peak:valley_right + 1] >= thr * y[peak]
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    m = deriv >= 0.25 * (np.max(deriv) if len(deriv) else 1)
    idxs = np.where(m)[0]
    f[21] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    # ---- f22~f24 面积 ----
    f[22] = np.trapz(y[valley_left:peak + 1], x[valley_left:peak + 1]) if peak > valley_left else 0.0
    f[23] = np.trapz(y[peak:valley_right + 1], x[peak:valley_right + 1]) if valley_right > peak else 0.0
    f[24] = f[22] + f[23]
    # ---- f25~f28 分段面积 ----
    dp = a_pos if a_pos > valley_left else peak
    f[25] = np.trapz(y[valley_left:dp + 1], x[valley_left:dp + 1]) if dp > valley_left else 0.0
    f[26] = np.trapz(y[dp:peak + 1], x[dp:peak + 1]) if peak > dp else 0.0
    f[27] = np.trapz(y[peak:e_pos + 1], x[peak:e_pos + 1]) if e_pos > peak else 0.0
    f[28] = np.trapz(y[e_pos:valley_right + 1], x[e_pos:valley_right + 1]) if valley_right > e_pos else 0.0
    denom = (f[25] + f[26] + f[27])
    f[29] = f[28] / denom if denom else 0.0
    # ---- f30~f31 平均斜率 ----
    f[30] = float(np.mean(deriv[valley_left:peak + 1])) if peak > valley_left else 0.0
    f[31] = float(np.mean(deriv[peak:valley_right + 1])) if valley_right > peak else 0.0
    # ---- f32~f34 斜率/峰值 ----
    f[32] = (y[peak] - y[valley_left]) / (x[peak] - x[valley_left]) if (x[peak] - x[valley_left]) else 0.0
    f[33] = (y[valley_right] - y[peak]) / (x[valley_right] - x[peak]) if (x[valley_right] - x[peak]) else 0.0
    f[34] = float(np.max(deriv)) if len(deriv) else 0.0
    # ---- f35 曲率 (e_pos 处) ----
    if 0 < e_pos < n - 1:
        f[35] = float(apg[e_pos])
    # ---- f36~f42 apg 幅度/比值 ----
    f[36] = float(apg[a_pos])
    f[37] = float(apg[b_pos])
    f[38] = float(apg[e_pos])
    f[39] = f[37] / f[36] if f[36] else 0.0
    f[40] = f[38] / f[36] if f[36] else 0.0
    f[41] = f[39] - f[40]
    f[42] = y[e_pos] / y[peak] if y[peak] else 0.0
    # ---- f43~f44 delta_t (inflection≈e_pos) ----
    f[43] = (e_pos - peak) / fs
    f[44] = f[43] / f[0] if f[0] else 0.0
    # ---- f45/f55 导数宽度 ----
    m = deriv >= 0.75 * (np.max(deriv) if len(deriv) else 1)
    idxs = np.where(m)[0]
    f[45] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    # ---- f46~f49 导数左右宽度 ----
    for thr, idx0 in [(0.33, 46), (0.66, 47)]:
        m = deriv[valley_left:peak + 1] >= thr * (np.max(deriv[valley_left:peak + 1]) if peak > valley_left else 1)
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    for thr, idx0 in [(0.33, 48), (0.66, 49)]:
        m = deriv[peak:valley_right + 1] >= thr * (np.max(deriv[peak:valley_right + 1]) if valley_right > peak else 1)
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    # ---- f50~f55 导数面积 ----
    f[50] = np.trapz(np.clip(deriv[valley_left:dp + 1], 0, None), x[valley_left:dp + 1]) if dp > valley_left else 0.0
    f[51] = np.trapz(np.clip(deriv[dp:peak + 1], 0, None), x[dp:peak + 1]) if peak > dp else 0.0
    f[52] = np.trapz(np.clip(deriv[peak:e_pos + 1], None, 0), x[peak:e_pos + 1]) if e_pos > peak else 0.0
    f[53] = float(np.mean(deriv[a_pos:dp + 1])) if dp > a_pos else 0.0
    f[54] = float(np.mean(deriv[dp:b_pos + 1])) if b_pos > dp else 0.0
    f[55] = float(np.mean(deriv[peak:e_pos + 1])) if e_pos > peak else 0.0
    # ---- f56~f57 导数曲率 ----
    f[56] = float(deriv[e_pos]) if 0 < e_pos < n else 0.0
    f[57] = f[56] / f[34] if f[34] else 0.0
    # ---- f58~f65 扩展宽度/对称性 (归一化占位, 与原生一致近似) ----
    for thr, idx0 in [(0.33, 58), (0.66, 59)]:
        m = y[valley_left:peak + 1] >= thr * y[peak]
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    for thr, idx0 in [(0.33, 60), (0.66, 61)]:
        m = y[peak:valley_right + 1] >= thr * y[peak]
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    for thr, idx0 in [(0.33, 62), (0.66, 63)]:
        m = deriv[valley_left:peak + 1] >= thr * (np.max(deriv[valley_left:peak + 1]) if peak > valley_left else 1)
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    for thr, idx0 in [(0.33, 64), (0.66, 65)]:
        m = deriv[peak:valley_right + 1] >= thr * (np.max(deriv[peak:valley_right + 1]) if valley_right > peak else 1)
        idxs = np.where(m)[0]
        f[idx0] = (idxs[-1] - idxs[0]) / fs if len(idxs) else 0.0
    # ---- f66~f73 对称性 (usdc/dsdc) ----
    if y[peak] > 0:
        left = y[valley_left:peak + 1] - y[valley_left]
        right = y[peak:valley_right + 1] - y[peak]
        L = min(len(left), len(right))
        # 上侧对称偏差 (peak 两侧相对幅度)
        usdc = np.abs(left[:L] - right[:L][::-1]) if L else np.array([0.0])
        f[66] = float(np.mean(usdc[:max(1, len(usdc)//5)]))
        f[67] = float(np.median(usdc)) if len(usdc) else 0.0
        f[68] = float(np.mean(usdc[max(0, len(usdc)*4//5):])) if len(usdc) else 0.0
        f[69] = float(np.std(usdc)) if len(usdc) else 0.0
        # 下侧对称偏差 (以 valley 为基准)
        dl = y[valley_left:peak + 1] - y[peak]
        dr = y[peak:valley_right + 1] - y[valley_right]
        dsdc = np.abs(dl[:L] - dr[:L][::-1]) if L else np.array([0.0])
        f[70] = float(np.mean(dsdc[:max(1, len(dsdc)//5)]))
        f[71] = float(np.median(dsdc)) if len(dsdc) else 0.0
        f[72] = float(np.mean(dsdc[max(0, len(dsdc)*4//5):])) if len(dsdc) else 0.0
        f[73] = float(np.std(dsdc)) if len(dsdc) else 0.0
    if np.any(np.isnan(f)) or np.any(np.isinf(f)):
        return None
    return f


def get_window_feature(ppg, fs=30, min_cycles=10):
    """把一段连续 PPG 按波谷切成 cycle (谷-峰-谷), 每个 cycle 重采样到 900Hz 后提取 74 维并平均。
    对应 Calculate_PPG_IBI 输出的 ppg_cycles_list + extract_ppg_features。
    注意: fs 参数此处仅用于波谷检测的距离阈值(真实帧率), 特征内部固定用 900Hz。"""
    from scipy.signal import find_peaks
    # 找局部极小 (波谷): 用 -ppg 找峰; 谷间距不小于 0.4s
    min_dist = max(2, int(0.4 * fs))
    valleys, _ = find_peaks(-ppg, distance=min_dist)
    if len(valleys) < 2:
        return 1, [], 0
    cycles = [ppg[valleys[i]:valleys[i + 1] + 1] for i in range(len(valleys) - 1)]
    if len(cycles) < min_cycles:
        return 1, [], len(cycles)
    feats = []
    for c in cycles:
        if len(c) < min_dist * 0.5:
            continue
        rc = _resample_cycle_to_900(c)               # 重采样到 900Hz 网格
        f = extract_single_cycle(rc, fs=FEATURE_FS)   # 内部固定 fs=900
        if f is None:
            continue
        feats.append(f)
    if len(feats) < max(1, min_cycles - 2):
        return 2, [], len(feats)
    return 0, list(np.mean(feats, axis=0)), len(feats)


def get_window_group_feature(window_group_list):
    if not window_group_list:
        return 3, []
    grp = np.array([np.mean(np.array(w), axis=0) for w in window_group_list])
    return 0, list(grp.mean(0))


# ---------------- 摄像头采集 ----------------
def _median_hr(history):
    """对齐 libCoreEngineV2.so::CalculateInstantHRVIndex2_RGB 的最终 HR 聚合逻辑。

    逆向确认 (反编译 @0x18c2d4): 最终 HR 不是算术平均, 而是 **中位数**!
      核心聚合调用 b_vmedian() (vertical median):
        - 当有效 HR 窗数 local_4998>=2 时, 走 else 分支算 RR 间期后取 b_vmedian;
        - 当 local_4998<2 时, 直接 b_vmedian(local_4920, n) 对有效 HR 序列取中位数。
      之后 60.0/median_RR 才得到稳定 HR。
    之前用 mean 聚合导致"完全不对": 偶发错窗会显著拉偏均值, 而中位数对离群窗更鲁棒,
    这与用户实测"有时准有时不准"完全吻合。

    因此这里: 仅用 snr>=HR_SNR_USABLE 且 HR 生理合理的窗, 取中位数 -> 对齐 b_vmedian。
    同时复刻 "High HR change" 保护: 若 max-min > max*0.1 标记 unstable(原生会随机扰动/拒判)。

    返回 (hr, n_valid, unstable)。"""
    pts = [hr for hr, snr in history
           if hr is not None and HR_PHYS_LO <= hr <= HR_PHYS_HI and snr >= HR_SNR_USABLE]
    if len(pts) < 1:
        return None, 0, False
    hrs = np.array(pts, float)
    hr = float(np.median(hrs))                  # 对齐 b_vmedian (中位数)
    unstable = False
    if len(hrs) >= 2:
        mx, mn = float(hrs.max()), float(hrs.min())
        if mx > 0 and (mx - mn) > mx * HIGH_HR_CHANGE_RATIO:
            unstable = True                      # High HR change
    return hr, len(pts), unstable


# 兼容旧调用名
_mean_hr = _median_hr


# MediaPipe FaceMesh 关键点, 对齐原生 parseRoiBoxFromLandmark 的输入:
# 原生 this+0x38 缓存的是 SeetaFace5 的 5 点 = [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]
# (顺序见 TddFa::track 注释 row(0)=第0点=左眼)。ForeHead ROI 即这 5 点包围盒算出。
# 映射到 MediaPipe 478 点: 左眼=33, 右眼=263, 鼻尖=1, 左嘴角=61, 右嘴角=291。
FOREHEAD_LM_IDS = [33, 263, 1, 61, 291]


def _roi_box_from_bbox(x1, y1, x2, y2):
    """对齐 libcmtrack.so::parseRoiBoxFromBbox (AllFace ROI A) @0x22520。

    反编译得到的精确 C++ (注意 (int) 截断与运算顺序):
        W  = (int)x2 - (int)x1
        H  = (int)y2 - (int)y1
        i1 = W + H
        size = (int)(((i1) >> 1) * 1.58)          // 先整数除2, 再乘 SCALE
        cx = (int)x2 - W*0.5 - size/2             // 注意减的是 W, 不是 H
        cy = (int)y2 - H*0.5 + (i1>>1)*0.14 - size/2
        ROI_A = [cx, cy, cx+size, cy+size]
    即: 以人脸框底边中心为基准, 向左上扩展 size×size 正方形。
    (之前版本把 cx 里的 W 错写成 H, 且 (W+H)/2 顺序错, 导致 ROI 偏离人脸 -> 心率错乱)"""
    W = int(x2) - int(x1)
    H = int(y2) - int(y1)
    if W <= 0 or H <= 0:
        return None
    i1 = W + H
    size = int((i1 >> 1) * ROI_A_SCALE)          # (W+H)//2 * 1.58
    if size < 4:
        return None
    cx = x2 - W * 0.5 - size * 0.5
    cy = y2 - H * 0.5 + (i1 >> 1) * ROI_A_YOFF - size * 0.5
    x0 = int(round(cx)); y0 = int(round(cy))
    return (x0, y0, x0 + size, y0 + size)


def _roi_box_from_landmarks(pts_xy):
    """对齐 libcmtrack.so::parseRoiBoxFromLandmark (ForeHead ROI B) @0x233f4。

    反编译精确逻辑 (输入矩阵为 2×N: row0=x, row1=y, 分别求 x/y 的 min/max):
        dx = xmax - xmin ;  dy = ymax - ymin
        d  = max(dx, dy) * 0.5                  // 取 x/y 跨度较大者, 不是 sqrt!
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        half = d
        ROI_B = [cx-half, cy-half, cx+half, cy+half]   // 边长 = 2*d = max(dx,dy)
    (之前版本用 sqrt(dx^2+dy^2)*0.5 是错的, 会让 ROI 偏小且偏圆)"""
    import math
    if len(pts_xy) < 1:
        return None
    xs = [p[0] for p in pts_xy]; ys = [p[1] for p in pts_xy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    d = max(dx, dy) * 0.5                        # 关键: max, 不是 sqrt
    if d < 2:
        return None
    cx = (xmax + xmin) * 0.5
    cy = (ymax + ymin) * 0.5
    half = d
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    s = int(round(half * 2))
    return (x0, y0, x0 + s, y0 + s)


def _roi_mean_and_valid(roi):
    """返回 (mean_rgb[3], valid_pixel_per)。对齐原生 VaildPixelPer: 非全黑像素占比。"""
    if roi.size == 0:
        return None, 0.0
    px = roi.reshape(-1, 3).astype(float)
    # 有效像素: 非全黑 (R+G+B>0)。原生 "too many zeros" 即有效占比过低。
    valid = (px.sum(axis=1) > 1.0)
    vpp = float(valid.mean()) if len(px) else 0.0
    if vpp < ROI_MIN_VALID_PIXEL_PER:
        return None, vpp
    return px[valid].mean(axis=0), vpp


def _draw_hud(frame, h, *, hr_f=None, snr_f=None, hr_a=None, snr_a=None,
              whr=None, nok=0, best_src="-", unst=False, status="采集中...",
              status_color=(0, 255, 0), status_y=30):
    """统一绘制常驻 HUD: 双 ROI 的 SNR/HR + 最终心率 + 状态 + 操作提示。
    所有分支(正常帧/评估窗/跳过帧)都调用本函数, 保证文字始终可见。"""
    # 双 ROI SNR/HR (无论是否达标都显示, 颜色区分)
    snr_fc = (0, 255, 0) if (hr_f is not None and snr_f is not None and snr_f >= HR_SNR_USABLE) else (0, 0, 255)
    snr_ac = (0, 255, 0) if (hr_a is not None and snr_a is not None and snr_a >= HR_SNR_USABLE) else (0, 0, 255)
    cv2.putText(frame, f"前额SNR={snr_f if snr_f is not None else -99:+.2f} HR={hr_f if hr_f else 0:.0f}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, snr_fc, 2)
    cv2.putText(frame, f"全脸SNR={snr_a if snr_a is not None else -99:+.2f} HR={hr_a if hr_a else 0:.0f}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, snr_ac, 2)
    # 最终心率 (历史中位数)
    if whr is not None:
        col = (0, 255, 0) if (nok >= HR_MIN_VALID_SAMPLES and not unst) else (0, 255, 255)
        put_text_cn(frame, f"心率≈{whr:.0f}  (n={nok} 优选:{best_src})", (10, 30), col, 0.9, 2)
        if unst:
            put_text_cn(frame, "心率波动大-数据不稳", (10, 65), (0, 165, 255), 0.8, 2)
        elif nok < HR_MIN_VALID_SAMPLES:
            put_text_cn(frame, "信号弱, 请保持静止", (10, 65), (0, 255, 255), 0.8, 2)
    else:
        put_text_cn(frame, status, (10, status_y), status_color, 1, 2)
    # 常驻操作提示 (全屏/窗口切换)
    put_text_cn(frame, "F键/双击: 全屏切换  Q: 退出", (10, h - 24),
                (200, 200, 200), 0.55, 1)


def collect(camera_idx, seconds, fps):
    import mediapipe as mp
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头索引 {camera_idx}")
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 创建可全屏窗口 (按 F 键切换全屏/窗口; 默认最大化显示)
    try:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    except Exception:
        fw, fh = 1280, 720
    make_window_fullscreen("rPPG", fw, fh)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5)
    rgb_ts = []          # ForeHead(ROI B) 平均 RGB 时序 —— HR 主信号
    rgb_ts_allface = []  # AllFace(ROI A) 平均 RGB 时序 —— 背景/校验参考
    history = []         # 历史 (hr, snr) 窗口样本
    prev_fore = None     # 上一帧 ForeHead 均值, 用于帧间变化门控
    # 缓存最近一次双 ROI 评估结果, 让非评估窗/跳过帧也能显示 HR/SNR
    cache = {"hr_f": None, "snr_f": None, "hr_a": None, "snr_a": None}
    start = time.time()
    real_fs = fps
    last_sample = start
    print(f"[采集] 开始 {seconds}s, 按 q 可提前结束")
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            print("[采集] 帧读取失败, 退出")
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if not (res.multi_face_landmarks):
            # 常驻 HUD (用缓存的上一次结果, 保证文字不消失)
            whr, nok, unst = _mean_hr(history)
            _draw_hud(frame, h, whr=whr, nok=nok, unst=unst,
                      hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                      hr_a=cache["hr_a"], snr_a=cache["snr_a"],
                      status="NO FACE - 等待人脸", status_color=(0, 0, 255), status_y=30)
            cv2.imshow("rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # 无人脸: 跳过该帧, 不污染 PPG 时序
        lm = res.multi_face_landmarks[0].landmark
        # 人脸框 (两点矩形) —— 对齐 FaceBox(x1,y1,x2,y2)
        xs = [p.x for p in lm]; ys = [p.y for p in lm]
        fx0, fx1 = int(min(xs) * w), int(max(xs) * w)
        fy0, fy1 = int(min(ys) * h), int(max(ys) * h)

        # ---- ROI A: AllFace (parseRoiBoxFromBbox) ----
        box_a = _roi_box_from_bbox(fx0, fy0, fx1, fy1)
        fore_mean = allface_mean = None
        if box_a is not None:
            x0, y0, x1, y1 = box_a
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            roi_a = rgb[y0:y1, x0:x1]
            allface_mean, vpp_a = _roi_mean_and_valid(roi_a)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)  # AllFace 绿框
            # 标注: 全脸 ROI 名称 + 具体坐标/尺寸 (对齐 .so 逆向的 2 个 ROI 定位结果)
            put_text_cn(frame, f"全脸ROI AllFace  x={x0} y={y0} w={x1-x0} h={y1-y0}",
                        (x0, max(0, y0 - 6)), (0, 255, 0), 0.45, 1)

        # ---- ROI B: ForeHead (parseRoiBoxFromLandmark) —— HR 主信号 ----
        fore_pts = [(int(lm[i].x * w), int(lm[i].y * h))
                    for i in FOREHEAD_LM_IDS if i < len(lm)]
        box_b = _roi_box_from_landmarks(fore_pts)
        if box_b is not None:
            x0, y0, x1, y1 = box_b
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            roi_b = rgb[y0:y1, x0:x1]
            fore_mean, vpp_b = _roi_mean_and_valid(roi_b)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)  # ForeHead 青框
            # 标注: 前额 ROI 名称 + 具体坐标/尺寸 (HR 主信号)
            put_text_cn(frame, f"前额ROI ForeHead  x={x0} y={y0} w={x1-x0} h={y1-y0}",
                        (x0, max(0, y0 - 6)), (255, 255, 0), 0.45, 1)

        # ---- 质量门控 (对齐原生) ----
        # 1) 至少 ForeHead 有效; 2) 帧间 ForeHead 均值 RGB 变化不超过阈值(否则 skip)
        if fore_mean is None:
            whr, nok, unst = _mean_hr(history)
            _draw_hud(frame, h, whr=whr, nok=nok, unst=unst,
                      hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                      hr_a=cache["hr_a"], snr_a=cache["snr_a"],
                      status="ROI无效-跳过", status_color=(0, 0, 255), status_y=30)
            cv2.imshow("rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        if prev_fore is not None:
            prev = np.asarray(prev_fore, float); cur = np.asarray(fore_mean, float)
            if prev.sum() > 0:
                change = float(np.abs(cur - prev).sum() / (prev.sum() + 1e-6))
                if change > ROI_FRAME_RGB_CHANGE_TH:
                    prev_fore = cur
                    whr, nok, unst = _mean_hr(history)
                    _draw_hud(frame, h, whr=whr, nok=nok, unst=unst,
                              hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                              hr_a=cache["hr_a"], snr_a=cache["snr_a"],
                              status="运动过大-丢帧", status_color=(0, 165, 255), status_y=30)
                    cv2.imshow("rPPG", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue  # 对齐 FaceRGBStdThresIgnore: 变化过大丢弃该帧
        prev_fore = np.asarray(fore_mean, float)

        rgb_ts.append(fore_mean)                 # 主信号: ForeHead
        if allface_mean is not None:
            rgb_ts_allface.append(allface_mean)  # 参考: AllFace

        # 每 HR_WINDOW_SEC 秒评估一次当前窗 HR/SNR (对齐 CalHRSNR_core)
        # 双 ROI: 分别计算 ForeHead(主) 与 AllFace(参考) 的 HR+SNR, 选 SNR 更高者作为本窗结果。
        # 对齐 libCoreEngineV2.so::c_CalS3CoreHR_RGB 的 e_maximum(选最佳 HR 通道/ROI)。
        now = time.time()
        if now - last_sample >= HR_WINDOW_SEC and len(rgb_ts) > 30:
            last_sample = now
            # ForeHead (ROI B) 主信号
            sub_f = bandpass(chrom_extract(rgb_ts, fps), fps)
            hr_f = estimate_hr(sub_f, fps)
            snr_f = calc_hr_snr(sub_f, fps)
            # AllFace (ROI A) 参考信号
            hr_a = snr_a = None
            if len(rgb_ts_allface) > 30:
                sub_a = bandpass(chrom_extract(rgb_ts_allface, fps), fps)
                hr_a = estimate_hr(sub_a, fps)
                snr_a = calc_hr_snr(sub_a, fps)
            # 选 SNR 更高者入历史 (e_maximum: 取最优 ROI 的 HR)
            hr = snr = None
            best_src = "-"
            if hr_f is not None and (hr_a is None or snr_f >= snr_a):
                hr, snr, best_src = hr_f, snr_f, "前额"
            elif hr_a is not None:
                hr, snr, best_src = hr_a, snr_a, "全脸"
            snr_ok = (hr is not None) and (snr >= HR_SNR_USABLE)
            # 仅 SNR 达标的窗才入历史 (低 SNR = 无主频 = 不显示/不采纳)
            if snr_ok:
                history.append((hr, snr))
            # 缓存到 cache, 供非评估窗/跳过帧持续显示
            cache.update({"hr_f": hr_f, "snr_f": snr_f, "hr_a": hr_a, "snr_a": snr_a})
        # 统一绘制 HUD (每帧都画, 文字常驻)
        whr, nok, unst = _mean_hr(history)
        best_src = cache.get("best_src", "-")
        if not history:
            # 还没攒够一个评估窗: 显示"采集中"但双ROI SNR 仍显示
            _draw_hud(frame, h, whr=None,
                      hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                      hr_a=cache["hr_a"], snr_a=cache["snr_a"],
                      status="采集中...", status_color=(0, 255, 0), status_y=30)
        else:
            # 评估窗已产生: 若最新窗 SNR 不达标, 顶部显示信号差提示(但心率数值仍画)
            last_snr = history[-1][1]
            if last_snr < HR_SNR_USABLE:
                _draw_hud(frame, h, whr=whr, nok=nok, unst=unst,
                          hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                          hr_a=cache["hr_a"], snr_a=cache["snr_a"],
                          status="信号差-SNR过低", status_color=(0, 0, 255), status_y=150)
            else:
                _draw_hud(frame, h, whr=whr, nok=nok, unst=unst, best_src=best_src,
                          hr_f=cache["hr_f"], snr_f=cache["snr_f"],
                          hr_a=cache["hr_a"], snr_a=cache["snr_a"])
        cv2.imshow("rPPG", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('f'):   # F 键切换全屏/窗口 (复用鼠标双击同一逻辑)
            try:
                make_window_fullscreen._toggle()
            except Exception:
                pass
    cap.release()
    cv2.destroyAllWindows()
    n = len(rgb_ts)
    real_fs = n / (time.time() - start) if (time.time() - start) > 0 else fps
    print(f"[采集] 获得 {n} 帧(ForeHead主信号), 实际帧率≈{real_fs:.1f}fps; 有效HR样本={len(history)}")
    return np.array(rgb_ts), real_fs, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--height", type=float, default=170.0)
    ap.add_argument("--gender", type=int, default=1)
    ap.add_argument("--age", type=int, default=30)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    # 1) 采集
    try:
        rgb_ts, fs, history = collect(args.camera, args.seconds, args.fps)
    except Exception as e:
        print(f"[错误] 摄像头采集失败: {e}")
        print("  本环境无摄像头属正常; 在有摄像头的机器用此 venv 运行即可。")
        sys.exit(1)
    if len(rgb_ts) < 30:
        print("[错误] 采集样本过少")
        sys.exit(1)

    # 2) CHROM -> PPG
    ppg = chrom_extract(rgb_ts, fs)
    ppg = bandpass(ppg, fs)
    # 最终 HR = 历史有效窗(仅 SNR>=门槛)的**中位数**, 对齐 CalculateInstantHRVIndex2_RGB
    # 的 b_vmedian (反编译 @0x18c2d4 实测: 核心聚合是 b_vmedian 而非均值; 均值会显著拉偏)。
    # SNR 加权(b_CalWeightMean)是另一条 HRV 子路径, 不参与最终 HR 输出。
    hr, n_valid, hr_unstable = _mean_hr(history)
    snr_all = [s for _, s in history] if history else [-100]
    snr_med = float(np.median(snr_all))
    if hr is None:
        # 退回: 用全段 FFT 兜底(仅当确有样本但全被 SNR 门控挡掉时)
        hr = estimate_hr(ppg, fs)
    print(f"[心率·最终] HR ≈ {hr:.1f} bpm  (有效样本 {n_valid}/{len(history)})"
          f"{'  [波动大-不稳]' if hr_unstable else ''}")
    print(f"[心率SNR] 中位 {snr_med:.3f}  (单窗可用门槛 {HR_SNR_USABLE:.2f})")

    # 有效性: 中位 SNR 必须达标 + 有效样本足够 + HR 生理合理; 否则不显示 HR/不跑 BP
    snr_ok = snr_med >= HR_SNR_USABLE
    hr_valid = (snr_ok and (hr is not None) and (HR_PHYS_LO <= hr <= HR_PHYS_HI)
                and (n_valid >= HR_MIN_VALID_SAMPLES) and (not hr_unstable))
    if not snr_ok:
        print(f"[心率] SNR 过低({snr_med:.2f} < {HR_SNR_USABLE:.2f}), 信号不可信, 不输出 HR/血压")
    elif hr_unstable:
        print(f"[心率] 历史 HR 波动过大(max-min>max*{HIGH_HR_CHANGE_RATIO:.1f}), 视为不稳, 不输出 HR/血压")

    # 3) PPG -> 74维特征 -> 窗口/组 -> 血压
    # 关键: get_window_feature 内部把每个 cycle 重采样到 900Hz 再提特征(fs 固定900),
    # 这里传入的 fs 仅用于波谷检测距离阈值(真实帧率)。
    flag, wf, nvalid = get_window_feature(ppg, fs=fs, min_cycles=10)
    print(f"[窗口特征] flag={flag}, n_valid_cycles={nvalid}")
    if flag != 0 or not wf:
        print("[血压] 特征不足(cycle 数不够或切割失败), 需更长的稳定采集, 本次跳过血压")
    elif not hr_valid:
        print("[血压] 有效心率样本不足或心率非生理值, 信号质量不足以给出可信血压, 本次跳过")
    else:
        gflag, gf = get_window_group_feature([[wf]])
        try:
            from bp_inference import BPModel
            bp = BPModel()
            # 注入元特征 f74=HR f75=年龄 f79=性别, 回归模型才能拿到真实信号
            gf_ext = list(gf) + [0.0] * (80 - len(gf))
            gf_ext[74] = round(hr, 2)     # 平均心率(历史加权)
            gf_ext[75] = float(args.age)  # 年龄
            gf_ext[79] = float(args.gender)
            res = bp.predict(gf_ext, height_cm=args.height, age_1_6=args.age // 10 + 1,
                             gender=args.gender)
            if "error" in res:
                print(f"[血压] 推理错误: {res['error']}")
            else:
                print(f"[血压·最终结果] 收缩压={res['hbp']}mmHg  舒张压={res['lbp']}mmHg "
                      f"(融合原始值 hbp_raw={res['hbp_raw']}, lbp_raw={res['lbp_raw']}; "
                      f"高压分段={'偏高' if res['hbp_is_high'] else '偏低'}, "
                      f"低压分段={'偏高' if res['lbp_is_high'] else '偏低'}; "
                      f"身高={int(args.height)} 性别={args.gender})")
        except Exception as e:
            print(f"[血压] 推理未运行: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
