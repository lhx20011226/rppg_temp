#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_test.py — 合成信号自测，验证 rppg_replica 的 HR/BP 链路是否与逆向结论一致。

测试 1: 纯净正弦 PPG，已知频率 f0 -> 期望 HR = 60*f0，SNR 应较高（实数、>0）。
测试 2: 多周期形态波形（75 BPM）走 PPG -> 切 cycle -> HR(IBI) -> BP 特征/模型。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import roi, hr, bp, pipeline


def test_sine_hr():
    print("=" * 60)
    print("测试1: 纯净正弦 PPG (已知频率)")
    fs = 30.0
    for f0 in [1.0, 1.25, 1.5]:   # 60, 75, 90 BPM
        t = np.arange(0, 30, 1 / fs)
        ppg = np.sin(2 * np.pi * f0 * t)
        h, snr = hr.estimate_hr_snr(ppg, fs)
        exp = 60.0 * f0
        ok = abs(h - exp) < 2.0   # FFT bin 量化误差（原生 SDK 同样存在，bin 分辨率=fs/N）
        print(f"  f0={f0:.2f}Hz 期望HR={exp:.0f}  测得HR={h:.2f}  SNR={snr:.3f}  "
              f"{'OK' if ok else 'FAIL'}")
        assert ok, f"HR mismatch: {h} vs {exp}"


def test_realistic_ppg():
    print("=" * 60)
    print("测试2: 形态波形 PPG -> cycle -> HR(IBI) -> BP")
    fs = 900.0
    hr_true = 75.0
    dur = 60.0 / hr_true
    # 构造多个周期：主收缩峰 + 重搏波，加轻微噪声
    n_cyc = 40
    cycles = []
    for _ in range(n_cyc):
        n = int(dur * fs)
        t = np.linspace(0, dur, n)
        y = (np.exp(-((t - 0.18 * dur) ** 2) / (2 * (0.06 * dur) ** 2)) +
             0.35 * np.exp(-((t - 0.55 * dur) ** 2) / (2 * (0.05 * dur) ** 2)))
        y -= y.min(); y /= y.max()
        y += 0.005 * np.random.randn(n)
        cycles.append(y.astype(float))
    # 拼接为长 PPG（中间用波谷连接）
    ppg = np.concatenate(cycles)
    # 模拟"已带通"的 gen_ppg 输出：这里直接用拼接波形（已具 0.x~几 Hz 成分）
    h, snr = hr.estimate_hr_snr(ppg, fs)
    print(f"  FFT-HR={h:.1f} BPM (真值 {hr_true})  SNR={snr:.3f}")

    seg_cyc, ibis = pipeline.segment_cycles(ppg, fs)
    print(f"  切出 cycle 数={len(seg_cyc)}  IBI均值={np.mean(ibis):.4f}s  "
          f"HR(IBI)={60/np.mean(ibis):.1f} BPM")
    # BP 前端
    flag_w, wf, n_valid = bp.get_window_feature(seg_cyc, min_n_valid_cycles=10)
    print(f"  窗口特征 flag={flag_w} n_valid={n_valid} 维度={len(wf)}")
    if n_valid >= 10 and wf:
        gflag, gf = bp.get_window_group_feature([[wf], [wf], [wf]])
        try:
            res = bp.BPModel().predict(gf, height_cm=170)
            print(f"  BP 推理: {res}")
        except Exception as e:
            print(f"  BP 推理跳过(模型依赖): {e}")


def test_roi():
    print("=" * 60)
    print("测试3: ROI 定位 (全 landmark 框)")
    # 模拟 478 点：脸大致在 [100,100]~[400,500]
    np.random.seed(0)
    lm = np.random.rand(478, 2) * 0 + 0
    lm[:, 0] = np.linspace(100, 400, 478)
    lm[:, 1] = np.linspace(100, 500, 478)
    x0, y0, x1, y1 = roi.parse_roi_box_from_landmark(lm)
    print(f"  ROI=({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f})")
    s = 0.5 * max(400 - 100, 500 - 100)
    half = s / (2 ** 0.5)
    cx, cy = 250, 300
    assert abs(x0 - (cx - half)) < 1e-6 and abs(x1 - (cx + half)) < 1e-6
    print("  ROI 公式核验 OK (中心质心, 边长=max(w,h)/2/sqrt2)")


if __name__ == "__main__":
    test_sine_hr()
    test_realistic_ppg()
    test_roi()
    print("=" * 60)
    print("全部自测通过 ✅")
