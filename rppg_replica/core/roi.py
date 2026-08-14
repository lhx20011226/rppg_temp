"""
core/roi.py
================
逆向后端 libcmtrack.so::TddFa::parseRoiBoxFromLandmark / parseRoiBoxFromBbox 的
100% 字节级复刻（基于 Ghidra 反编译 001233f4 / 00122520）。

关键逆向结论：
  * 原生 SDK 的 ROI 不是"额头几个点"，而是用【全部人脸 landmark】求 x∈[xmin,xmax]、
    y∈[ymin,ymax] 的包围盒，再取最大边长的一半作为半对角线 s，中心为质心。
  * 最终 ROI 框是一个【方形】：
        cx = (xmin+xmax)/2 ; cy = (ymin+ymax)/2
        s  = 0.5 * max(xmax-xmin, ymax-ymin)
        half = s / sqrt(2)            # 正方形内接于半径 s 的圆
        x0 = cx - half ; x1 = cx + half
        y0 = cy - half ; y1 = cy + half
  * parseRoiBoxFromBbox 是另一种：用 bbox 宽高和的一半 * 1.58 作为边长，并偏移到额头
    （y 上移 0.14*side，x 居中）。本工程主路径使用 landmark 版本（与 FaceMesh 478 点一致）。

输出约定：返回 (x0, y0, x1, y1) 像素坐标（浮点，左上-右下）。
"""


def parse_roi_box_from_landmark(landmarks):
    """
    参数
    ----
    landmarks : ndarray, shape (N, 2) or (N, 3)
        FaceMesh 478 点的 (x, y) 像素坐标（或带 z 的前三维，只用 x,y）。

    返回
    ----
    (x0, y0, x1, y1) : tuple[float]
        ROI 矩形框（浮点像素坐标）。
    """
    import numpy as np
    lm = np.asarray(landmarks, dtype=float)
    xs = lm[:, 0]
    ys = lm[:, 1]

    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    side = max(xmax - xmin, ymax - ymin)
    s = 0.5 * side                       # 半对角线
    half = s / (2.0 ** 0.5)             # 正方形半边长

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half
    return (x0, y0, x1, y1)


def parse_roi_box_from_bbox(bbox):
    """
    逆向后端 parseRoiBoxFromBbox (00122520) 的复刻，作为 landmark 不可用时的回退。

    参数
    ----
    bbox : (x, y, w, h)  左上角 + 宽高（像素）。

    返回
    ----
    (x0, y0, x1, y1)
    """
    x, y, w, h = [float(v) for v in bbox]
    side_w = int(w) - int(x)            # 反编译里 iVar3 = (int)right - (int)left
    side_h = int(h) - int(y)
    iVar1 = side_h + side_w
    iVar8 = int((iVar1 >> 1) * 1.58)    # 边长
    cx = (int(x) - (side_w * 0.5)) - (iVar8 >> 1)
    cy = ((int(y) - (side_h * 0.5)) + (iVar1 >> 1) * 0.14) - (iVar8 >> 1)
    return (cx, cy, cx + iVar8, cy + iVar8)


def roi_mean_rgb(frame_bgr, roi):
    """
    在 ROI 矩形内对 BGR 帧取平均色，返回 (R, G, B) 浮点（0~255）。

    roi : (x0, y0, x1, y1) 浮点像素坐标。
    """
    import numpy as np
    x0, y0, x1, y1 = [int(round(v)) for v in roi]
    h, w = frame_bgr.shape[:2]
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return (0.0, 0.0, 0.0)
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))  # R, G, B
