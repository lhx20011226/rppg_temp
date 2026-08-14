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

# ---------------- rPPG 信号提取 (CHROM) ----------------
def chrom_extract(rgb_ts, fs):
    """CHROM: 输入 (N,3) RGB 时序, 返回 PPG 信号 (N,)"""
    rgb = np.asarray(rgb_ts, float)
    rgb = rgb - np.mean(rgb, axis=0, keepdims=True)  # 去均值
    std = np.std(rgb, axis=0, keepdims=True)
    std[std == 0] = 1.0
    rgb = rgb / std
    # 投影
    X = rgb[:, 0] - rgb[:, 2]          # R - B
    Y = rgb[:, 0] + rgb[:, 2] - 2 * rgb[:, 1]  # R + B - 2G
    # 带通近似: 用二阶差分模拟 (实际用下面 butter 带通)
    alpha = np.std(X) / (np.std(Y) + 1e-9)
    ppg = X - alpha * Y
    return ppg


def bandpass(sig, fs, lo=0.7, hi=4.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def estimate_hr(ppg, fs):
    from scipy.signal import welch
    f, P = welch(ppg, fs=fs, nperseg=min(len(ppg), 512))
    mask = (f >= 0.7) & (f <= 4.0)
    if not np.any(mask):
        return None
    f = f[mask]; P = P[mask]
    hr = f[np.argmax(P)] * 60.0
    return float(hr)


def calc_hr_snr(ppg, fs):
    """复刻 libCoreEngineV2.so::CalHRSNR_core 末尾:
       HRSNR = log10(主频峰值功率 / 全频谱总功率) + 0.6
    返回 SNR (float)。值越高表示 PPG 信噪比越好, 心率越可信。"""
    from scipy.signal import welch
    ppg = np.asarray(ppg, float)
    if len(ppg) < 8:
        return -100.0
    f, P = welch(ppg, fs=fs, nperseg=min(len(ppg), 512))
    P = np.maximum(P, 0.0)
    total = float(np.sum(P))
    if total <= 0:
        return -100.0
    peak = float(np.max(P))
    snr = np.log10(peak / total) + 0.6
    return float(snr)


# 心率有效性判定 (对齐 libCoreEngineV2.so::FilterHRUsingPreviousDataRGB 意图)
# 原生不对瞬时 SNR 一票否决, 而是:
#   - 维护历史 HR/SNR 窗口, 用高置信 HR 加权得到平滑最终 HR (b_CalWeightMean)
#   - 仅在 SNR 持续极低(无信号)时才退回 refHR / 放弃
# 这里是务实复刻参数:
HR_WINDOW_SEC = 3.0        # 历史窗口采样间隔(秒)
HR_MIN_VALID_SAMPLES = 3   # 历史中"可用"样本最少个数, 低于则判定信号不足
HR_PHYS_LO, HR_PHYS_HI = 40.0, 200.0   # 心率生理合理区间(原生大量 0<HR 检查)
# 单窗 SNR 可用门槛: 原生瞬时 SNR 是 log10(peak/total)+0.6 量纲(约 -0.5~0.5),
# 这里取"基本有主频"的下限, 低于它的窗仅不计入加权、但不否决整体。
HR_SNR_USABLE = -1.0

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
def extract_single_cycle(y, fs=900):
    n = len(y)
    valley_left, valley_right = 0, n - 1
    x = np.arange(n, dtype=float)
    peak = int(np.argmax(y[valley_left:valley_right])) + valley_left
    if peak <= valley_left or peak >= valley_right:
        return None
    deriv = np.gradient(y, x)
    apg = np.gradient(deriv, x)
    f = [0.0] * 74
    f[0] = (valley_right - valley_left) / fs
    f[1] = (peak - valley_left) / fs
    f[2] = (valley_right - peak) / fs
    f[3] = f[1] / f[0] if f[0] else 0.0
    f[32] = (y[peak] - y[valley_left]) / (x[peak] - x[valley_left]) if (x[peak]-x[valley_left]) else 0.0
    f[33] = (y[valley_right] - y[peak]) / (x[valley_right] - x[peak]) if (x[valley_right]-x[peak]) else 0.0
    f[34] = float(np.max(deriv))
    f[22] = np.trapz(y[valley_left:peak + 1], x[valley_left:peak + 1])
    f[23] = np.trapz(y[peak:valley_right + 1], x[peak:valley_right + 1])
    f[24] = f[22] + f[23]
    a_pos = valley_left + int(np.argmax(apg[valley_left:peak]))
    b_c = np.where(apg[valley_left:peak] < 0)[0]
    b_pos = (valley_left + b_c[np.argmin(apg[valley_left:peak][b_c])]) if len(b_c) else peak
    dc = np.where(y[peak:valley_right] >= 0.5 * y[peak])[0]
    e_pos = peak + (dc[-1] if len(dc) else 0)
    f[36] = apg[a_pos]; f[37] = apg[b_pos]; f[38] = apg[e_pos]
    f[39] = apg[b_pos] / apg[a_pos] if apg[a_pos] else 0.0
    f[40] = apg[e_pos] / apg[a_pos] if apg[a_pos] else 0.0
    f[41] = f[39] - f[40]
    f[42] = y[e_pos] / y[peak] if y[peak] else 0.0
    return f


def get_window_feature(ppg, fs=900, min_cycles=10):
    """把一段连续 PPG 按波谷切成 cycle (谷-峰-谷), 提取 74 维并平均。
    对应 Calculate_PPG_IBI 输出的 ppg_cycles_list。"""
    from scipy.signal import find_peaks
    # 找局部极小 (波谷): 用 -ppg 找峰
    neg = -ppg
    # 谷间距不小于 0.4s
    min_dist = int(0.4 * fs)
    valleys, _ = find_peaks(neg, distance=min_dist)
    if len(valleys) < 2:
        return 1, [], 0
    cycles = [ppg[valleys[i]:valleys[i + 1] + 1] for i in range(len(valleys) - 1)]
    if len(cycles) < min_cycles:
        return 1, [], len(cycles)
    feats = []
    for c in cycles:
        if len(c) < min_dist * 0.5:
            continue
        f = extract_single_cycle(np.asarray(c, float), fs=fs)
        if f is None or np.any(np.isnan(f)):
            continue
        feats.append(f)
    if len(feats) < min_cycles - 2:
        return 2, [], len(feats)
    return 0, list(np.mean(feats, axis=0)), len(feats)


def get_window_group_feature(window_group_list):
    if not window_group_list:
        return 3, []
    grp = np.array([np.mean(np.array(w), axis=0) for w in window_group_list])
    return 0, list(grp.mean(0))


# ---------------- 摄像头采集 ----------------
def _weighted_hr(history):
    """对齐原生 b_CalWeightMean: 用 SNR 做权重对历史 HR 加权得到稳定 HR。
    history: list of (hr, snr)。仅用 snr>=HR_SNR_USABLE 的样本, 权重=sigmoid(snr)。"""
    pts = [(hr, snr) for hr, snr in history
           if hr is not None and HR_PHYS_LO <= hr <= HR_PHYS_HI and snr >= HR_SNR_USABLE]
    if len(pts) < 1:
        return None, 0
    hrs = np.array([p[0] for p in pts], float)
    # 权重: SNR 越高权重越大 (sigmoid 映射到 0~1, 避免极端值)
    w = 1.0 / (1.0 + np.exp(-(np.array([p[1] for p in pts], float) + 0.3) * 4.0))
    if w.sum() <= 0:
        w = np.ones_like(hrs)
    hr = float(np.sum(hrs * w) / w.sum())
    return hr, len(pts)


# MediaPipe FaceMesh 前额关键点子集索引 (对齐原生 ForeHead = HR 主信号区域)。
# 取额头/眉心区域的点: 10=眉心, 67/69/104/108 额头, 151/9 上额, 338/337/299 右额, 71/63 左额
FOREHEAD_LM_IDS = [10, 67, 69, 104, 108, 151, 9, 338, 337, 299, 71, 63, 105, 66, 46, 57]


def _roi_box_from_bbox(x1, y1, x2, y2):
    """对齐 libcmtrack.so::parseRoiBoxFromBbox (AllFace ROI A)。
    FaceBox 为两点矩形; size=round((W+H)/2*1.58); 中心在底边中心, 向左上扩展。"""
    W = x2 - x1
    H = y2 - y1
    size = int(round((W + H) / 2 * ROI_A_SCALE))
    if size < 4:
        return None
    cx = x2 - H * 0.5
    cy = y2 - H * 0.5 + size * ROI_A_YOFF
    x0 = int(round(cx - size / 2)); y0 = int(round(cy - size / 2))
    return (x0, y0, x0 + size, y0 + size)


def _roi_box_from_landmarks(pts_xy):
    """对齐 libcmtrack.so::parseRoiBoxFromLandmark (ForeHead ROI B)。
    pts_xy: list of (x,y) 像素坐标; 以质心为中心、半对角距 radius 为半边长的正方形。"""
    if len(pts_xy) < 1:
        return None
    xs = [p[0] for p in pts_xy]; ys = [p[1] for p in pts_xy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx = (xmax - xmin) * 0.5
    dy = (ymax - ymin) * 0.5
    import math
    radius = math.sqrt(dx * dx + dy * dy) * 0.5   # 半对角距 * 0.5 (对齐 fmul 0.5)
    if radius < 2:
        return None
    cx = (xmax + xmin) * 0.5
    cy = (ymax + ymin) * 0.5
    half = radius
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


def collect(camera_idx, seconds, fps):
    import mediapipe as mp
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头索引 {camera_idx}")
    cap.set(cv2.CAP_PROP_FPS, fps)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5)
    rgb_ts = []          # ForeHead(ROI B) 平均 RGB 时序 —— HR 主信号
    rgb_ts_allface = []  # AllFace(ROI A) 平均 RGB 时序 —— 背景/校验参考
    history = []         # 历史 (hr, snr) 窗口样本
    prev_fore = None     # 上一帧 ForeHead 均值, 用于帧间变化门控
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
            cv2.putText(frame, "NO FACE - skip", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
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

        # ---- 质量门控 (对齐原生) ----
        # 1) 至少 ForeHead 有效; 2) 帧间 ForeHead 均值 RGB 变化不超过阈值(否则 skip)
        if fore_mean is None:
            cv2.putText(frame, "ROI invalid - skip", (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("rPPG", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        if prev_fore is not None:
            prev = np.asarray(prev_fore, float); cur = np.asarray(fore_mean, float)
            if prev.sum() > 0:
                change = float(np.abs(cur - prev).sum() / (prev.sum() + 1e-6))
                if change > ROI_FRAME_RGB_CHANGE_TH:
                    cv2.putText(frame, "motion - skip frame", (10, h - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    prev_fore = cur
                    cv2.imshow("rPPG", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue  # 对齐 FaceRGBStdThresIgnore: 变化过大丢弃该帧
        prev_fore = np.asarray(fore_mean, float)

        rgb_ts.append(fore_mean)                 # 主信号: ForeHead
        if allface_mean is not None:
            rgb_ts_allface.append(allface_mean)  # 参考: AllFace

        # 每 HR_WINDOW_SEC 秒产生一个 (HR, SNR) 样本入历史
        now = time.time()
        if now - last_sample >= HR_WINDOW_SEC and len(rgb_ts) > 30:
            last_sample = now
            sub = chrom_extract(rgb_ts, fps)
            sub = bandpass(sub, fps)
            hr = estimate_hr(sub, fps)
            snr = calc_hr_snr(sub, fps)
            if hr is not None:
                history.append((hr, snr))
            whr, nok = _weighted_hr(history)
            if whr is not None:
                col = (0, 255, 0) if nok >= HR_MIN_VALID_SAMPLES else (0, 255, 255)
                cv2.putText(frame, f"HR≈{whr:.0f}  (n={nok})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
                if nok < HR_MIN_VALID_SAMPLES:
                    cv2.putText(frame, "signal weak, keep still", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "collecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            whr, _ = _weighted_hr(history)
            if whr is not None:
                cv2.putText(frame, f"HR≈{whr:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "collecting...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
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
    # 最终 HR 用历史窗口加权 (对齐原生 FilterHRUsingPreviousDataRGB 的 b_CalWeightMean),
    # 不再用瞬时 SNR 一票否决
    hr, n_valid = _weighted_hr(history)
    if hr is None:
        # 退回: 用全段 FFT 兜底
        hr = estimate_hr(ppg, fs)
    snr_all = [s for _, s in history] if history else [-100]
    snr_med = float(np.median(snr_all))
    print(f"[心率·最终] HR ≈ {hr:.1f} bpm  (有效样本 {n_valid}/{len(history)})")
    print(f"[心率SNR] 中位 {snr_med:.3f}  (单窗可用门槛 {HR_SNR_USABLE:.2f})")

    # 有效性: 对齐原生意图 —— 仅在 有效样本过少 或 HR 非生理 时放弃 (不卡瞬时 SNR)
    hr_valid = (hr is not None) and (HR_PHYS_LO <= hr <= HR_PHYS_HI) and (n_valid >= HR_MIN_VALID_SAMPLES)

    # 3) PPG -> 74维特征 -> 窗口/组 -> 血压
    # 注意: 必须用真实帧率 fs(摄像头实际 fps), 不能填算法内部假设的 900
    flag, wf, nvalid = get_window_feature(ppg, fs=fs, min_cycles=6)
    print(f"[窗口特征] flag={flag}, n_valid_cycles={nvalid}")
    if flag != 0 or not wf:
        print("[血压] 特征不足, 需更长的稳定采集")
        sys.exit(0)
    if not hr_valid:
        print("[血压] 有效心率样本不足或心率非生理值, 信号质量不足以给出可信血压, 请重测")
        sys.exit(0)
    gflag, gf = get_window_group_feature([[wf], [wf], [wf]])
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
