# rppg_replica —— 原生 Android rPPG SDK 字节级复刻（Python）

从零重建的无接触心率(HR)/血压(BP)测量项目。**彻底逆向** `libCoreEngineV2.so` /
`libcmtrack.so`（arm64）后，按逆向结论重新实现，不复用任何旧 `rppg_cam.py` 逻辑。

## 逆向依据（Ghidra 反编译确证）

| 后端符号 | 地址 | 复刻位置 | 关键结论 |
|---|---|---|---|
| `TddFa::parseRoiBoxFromLandmark` | `libcmtrack 001233f4` | `core/roi.py` | ROI = 全部 landmark 包围盒，半对角线 `s=0.5*max(w,h)`，中心质心，`half=s/√2` |
| `GenR5` | `libCoreEngineV2 001f83cc` | `core/ppg.py` | RGB→逐通道 zscore→`2R-G-B` 投影→butter `[0.3,3.5]Hz` 零相移带通 |
| `CalHRSNR_core` | `libCoreEngineV2 0014f490` | `core/hr.py` | HR = `fs*60*peak_bin/N`（FFT 峰值，抛物线插值精修）；**SNR = `log10(带内能量/带外能量)+0.6`（实数）** |
| `GenPythonPPGIBI` | `libCoreEngineV2 001f5c10` | `core/pipeline.py` | IBI 序列整体 `×10` 缩放 |
| `Calculate_PPG_IBI` | `libCoreEngineV2 001ae95c` | — | MATLAB Coder 调度器（按数据质量/长度分支），算法本体在 GenR5 |
| BP 前端 (`extract_ppg_features`/`dm_bp_api`) | `.pyc` 逆向 | `core/bp.py` | PPG cycle→74 维形态学特征→窗口特征→组特征(0.66/0.34)→真实身高桶模型 |

## 文件结构

```
rppg_replica/
  core/
    roi.py        # parseRoiBoxFromLandmark / parseRoiBoxFromBbox / roi_mean_rgb
    ppg.py        # GenR5: zscore -> 2R-G-B -> butter 带通 -> (滑窗HR)
    hr.py         # CalHRSNR_core: FFT 峰值 HR + 实数 SNR
    bp.py         # 74维特征 -> 窗口/组特征 -> 真实血压模型
    pipeline.py   # 端到端: frames_rgb -> PPG -> cycle -> IBI/HR -> BP
  camera.py       # 实时摄像头 + Mediapipe FaceMesh + GUI
  synth_test.py   # 合成信号自测（HR/SNR/BP 链路验证）
```

## 运行

```bash
# 合成信号自测
/workspace/rppg_venv/bin/python synth_test.py

# 实时摄像头（需摄像头 + 显示器）
/workspace/rppg_venv/bin/python camera.py
```

依赖：`numpy==1.23.5 scipy==1.10.1 opencv-python mediapipe`（血压模型桶还需
`scikit-learn==1.1.3` + `imbalanced-ensemble==0.1.1`，lgbm 桶需 `lightgbm`）。

## 与旧项目的根本区别（修复点）

1. **ROI 定位**：旧代码用 ForeHead 几个关键点 → 框偏角落；新代码用**全部 landmark 包围盒**（与 .so 一致）。
2. **HR 缩放**：旧代码用固定 `fps=30` 而非真实帧率；新代码用帧时间戳算真实 `fs`。
3. **SNR 为实数**：旧代码 SNR 是复数（实现错误）；新代码严格 `log10(x)+0.6`，恒为实数。
4. **投影**：严格 `2R-G-B`（CHROM），带通 `[0.3,3.5]Hz`（GenR5 实测 butter 系数）。
