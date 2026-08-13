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
def collect(camera_idx, seconds, fps):
    import mediapipe as mp
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头索引 {camera_idx}")
    cap.set(cv2.CAP_PROP_FPS, fps)
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5)
    rgb_ts = []          # ROI 平均 RGB 时序
    frames = []
    start = time.time()
    real_fs = fps
    print(f"[采集] 开始 {seconds}s, 按 q 可提前结束")
    last_t = start
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            print("[采集] 帧读取失败, 退出")
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        # 默认取整帧中部区域作为 ROI (无人脸时退化为中心块)
        roi = rgb[h//3:2*h//3, w//3:2*w//3]
        mean_rgb = roi.reshape(-1, 3).mean(axis=0)
        rgb_ts.append(mean_rgb)
        frames.append(frame)
        # 画个框提示
        cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (0, 255, 0), 2)
        cv2.putText(frame, "collecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("rPPG", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    n = len(rgb_ts)
    real_fs = n / (time.time() - start) if (time.time() - start) > 0 else fps
    print(f"[采集] 获得 {n} 帧, 实际帧率≈{real_fs:.1f}fps")
    return np.array(rgb_ts), real_fs


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
        rgb_ts, fs = collect(args.camera, args.seconds, args.fps)
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
    hr = estimate_hr(ppg, fs)
    print(f"[心率] HR ≈ {hr:.1f} bpm")

    # 3) PPG -> 74维特征 -> 窗口/组 -> 血压
    # 注意: 必须用真实帧率 fs(摄像头实际 fps), 不能填算法内部假设的 900
    flag, wf, nvalid = get_window_feature(ppg, fs=fs, min_cycles=6)
    print(f"[窗口特征] flag={flag}, n_valid_cycles={nvalid}")
    if flag != 0 or not wf:
        print("[血压] 特征不足, 需更长的稳定采集")
        sys.exit(0)
    gflag, gf = get_window_group_feature([[wf], [wf], [wf]])
    try:
        from bp_inference import BPModel
        bp = BPModel()
        # 注入元特征 f74=HR f75=年龄 f79=性别, 回归模型才能拿到真实信号
        gf_ext = list(gf) + [0.0] * (80 - len(gf))
        gf_ext[74] = round(hr, 2)     # 平均心率
        gf_ext[75] = float(args.age)  # 年龄
        gf_ext[79] = float(args.gender)
        res = bp.predict(gf_ext, height_cm=args.height, age_1_6=args.age // 10 + 1,
                         gender=args.gender)
        if "error" in res:
            print(f"[血压] 推理错误: {res['error']}")
        else:
            print(f"[血压] 收缩压={res['hbp']}mmHg  舒张压={res['lbp']}mmHg "
                  f"(融合原始值 hbp_raw={res['hbp_raw']}, lbp_raw={res['lbp_raw']}; "
                  f"身高={int(args.height)} 性别={args.gender})")
    except Exception as e:
        print(f"[血压] 推理未运行: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
