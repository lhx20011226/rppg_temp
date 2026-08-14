#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成信号自测: 验证 chrom_extract(2R-G-B) + bandpass + _hr_snr_one 能否正确还原心率。
不依赖摄像头, 直接 import rppg_cam 的函数。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rppg_cam as R

def synth_ppg(bpm, fs, n_sec, noise=0.0, amplitude=1.0, hr_motion=0.0):
    """合成一段含心率的 RGB 时序 (N,3), 让 CHROM 2R-G-B 投影后正好是一个正弦。
    我们构造: R = base + a*sin, G = base, B = base, 那么 2R-G-B = 2*(base+a*sin) - base - base = 2a*sin
    这样投影后就是纯正弦, 频率 = bpm/60, 便于验证 FFT 还原。"""
    t = np.arange(int(n_sec * fs)) / fs
    f = bpm / 60.0
    wave = amplitude * np.sin(2 * np.pi * f * t)
    # 加少量运动伪影(低频)模拟真实
    motion = hr_motion * np.sin(2 * np.pi * 0.2 * t)
    R_ = 128 + 2 * wave + motion
    G_ = 128 + 1 * wave * 0.3 + motion
    B_ = 128 + 1 * wave * 0.3 + motion
    rgb = np.stack([R_, G_, B_], axis=1)
    if noise > 0:
        rgb += np.random.randn(*rgb.shape) * noise
    return rgb, fs, t

print("=== 合成信号 HR 还原自测 (chrom_extract=2R-G-B) ===")
for bpm in [60, 72, 75, 90, 110, 130]:
    rgb, fs, _ = synth_ppg(bpm, 30.0, 12.0, noise=0.5, amplitude=1.0)
    ppg = R.chrom_extract(rgb, fs)
    ppg_bp = R.bandpass(ppg, fs)
    hr, snr = R._hr_snr_one(ppg_bp, fs)
    mark = "OK " if (hr is not None and abs(hr - bpm) <= 3) else "FAIL"
    print(f"  [{mark}] 目标 {bpm:3d}bpm -> 测得 hr={hr if hr else None}, snr={snr:+.3f}")

print("=== 低 SNR 场景 (噪声大, 应 SNR<门槛 不达标) ===")
for bpm in [72, 90]:
    rgb, fs, _ = synth_ppg(bpm, 30.0, 6.0, noise=8.0, amplitude=1.0)
    ppg = R.chrom_extract(rgb, fs)
    ppg_bp = R.bandpass(ppg, fs)
    hr, snr = R._hr_snr_one(ppg_bp, fs)
    usable = (hr is not None) and (snr >= R.HR_SNR_USABLE)
    print(f"  目标 {bpm}bpm 噪声大 -> hr={hr if hr else None} snr={snr:+.3f} 达标={usable} (门槛={R.HR_SNR_USABLE})")

print("=== 中位数聚合自测 ===")
# 构造 6 个窗: 5个正确 + 1个离群错窗, 验证中位数比均值更稳
h = [(75, 0.1), (74, 0.1), (76, 0.1), (75, 0.1), (73, 0.1), (140, 0.1)]
hr_med, n, unst = R._median_hr(h)
hr_mean = float(np.mean([x for x, _ in h]))
print(f"  样本含1离群(140): 中位数={hr_med:.1f}  均值={hr_mean:.1f}  真实≈75  有效n={n}")
