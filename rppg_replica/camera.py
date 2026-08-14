#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera.py — 从零重建的实时 rPPG 摄像头程序（rppg_replica）

严格依据 Ghidra 逆向结论（不沿用旧 rppg_cam.py）：
  * ROI 定位：libcmtrack.so::parseRoiBoxFromLandmark —— 用【全部人脸 landmark】求
    包围盒，取最大边长一半为半对角线，中心为质心（修复旧代码"额头几点定位错/框偏角落"）。
  * 取色：ROI 内平均 BGR -> (R,G,B)。
  * PPG 生成：core/ppg.gen_ppg = GenR5（zscore -> 2R-G-B -> butter[0.3,3.5]Hz 带通）。
  * HR/SNR：core/hr.estimate_hr_snr = CalHRSNR_core（FFT 峰值 HR + 实数 SNR=log10+0.6）。
  * 帧率：用每帧真实时间戳算 fs（修复旧代码固定 30fps 导致的整体缩放错误）。
  * 评估窗：每累积 win_sec 秒帧做一次 HR/SNR/BP 评估。
  * GUI：中文用 PIL 绘制（避免 OpenCV 中文乱码）。

运行：
  /workspace/rppg_venv/bin/python camera.py
"""

import sys
import os
import time
import queue

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Mediapipe FaceMesh
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import roi as roi_mod
from core import hr as hr_mod
from core import pipeline


# ------------------------- 字体（中文） -------------------------
def _load_font(size=22):
    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT = _load_font(22)
_FONT_BIG = _load_font(40)


def _put_text_cn(img, text, pos, color=(0, 255, 0), font=_FONT):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ------------------------- 全局状态 -------------------------
class RppgState:
    def __init__(self, win_sec=3.0, height_cm=170):
        self.win_sec = win_sec
        self.height_cm = height_cm
        self.rgb_buf = []          # list of (R,G,B)
        self.t_buf = []            # 帧时间戳
        self.ppg = None
        self.last_hr = -1.0
        self.last_snr = -100.0
        self.last_bp = {}
        self.last_eval_t = 0.0

    def add_frame(self, rgb, t):
        self.rgb_buf.append(rgb)
        self.t_buf.append(t)
        # 仅保留最近 win_sec*2 的内容，避免无限增长
        if self.t_buf:
            cutoff = self.t_buf[-1] - max(self.win_sec * 2, 5.0)
            while self.t_buf and self.t_buf[0] < cutoff:
                self.t_buf.pop(0)
                self.rgb_buf.pop(0)

    def maybe_eval(self, now):
        if now - self.last_eval_t < self.win_sec:
            return False
        if len(self.rgb_buf) < 16:
            return False
        # 真实帧率
        fs = len(self.t_buf) / max(1e-3, (self.t_buf[-1] - self.t_buf[0]))
        fs = min(max(fs, 5.0), 120.0)
        res = pipeline.process(self.rgb_buf, fs, height_cm=self.height_cm)
        if "error" in res:
            return False
        self.ppg = res["ppg"]
        self.last_hr = res["hr_bpm"]
        self.last_snr = res["snr"]
        self.last_bp = res.get("bp", {})
        self.last_eval_t = now
        return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头（/dev/video0）。请检查设备或改用其它索引。")
        return

    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    state = RppgState(win_sec=3.0, height_cm=170)
    win_name = "rPPG Replica"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("[提示] 按 Q 退出。请正对摄像头、光线均匀、尽量保持静止。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 读帧失败")
            break
        frame = cv2.flip(frame, 1)
        t = time.time()
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb_frame)

        box = None
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])  # (478,2) 像素
            box = roi_mod.parse_roi_box_from_landmark(pts)
            r, g, b = roi_mod.roi_mean_rgb(frame, box)
            state.add_frame((r, g, b), t)
            state.maybe_eval(t)

            # 画 ROI 框
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        else:
            _put_text_cn(frame, "未检测到人脸", (20, 30), (0, 0, 255))

        # 叠加 HR / SNR / BP 文本
        hr_text = f"HR: {state.last_hr:.0f} BPM" if state.last_hr > 0 else "HR: --"
        snr_text = f"SNR: {state.last_snr:.2f}" if state.last_snr > -90 else "SNR: --"
        frame = _put_text_cn(frame, hr_text, (20, 30), (0, 255, 0), _FONT_BIG)
        frame = _put_text_cn(frame, snr_text, (20, 90), (0, 200, 255))
        if state.last_bp and "error" not in state.last_bp:
            hbp = state.last_bp.get("hbp_risk", float('nan'))
            lbp = state.last_bp.get("lbp_risk", float('nan'))
            frame = _put_text_cn(frame, f"高血压风险 收缩:{hbp:.2f} 舒张:{lbp:.2f}",
                                 (20, 125), (255, 200, 0))
        frame = _put_text_cn(frame, f"缓冲帧数: {len(state.rgb_buf)}", (20, 160),
                             (180, 180, 180))

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_face.close()


if __name__ == "__main__":
    main()
