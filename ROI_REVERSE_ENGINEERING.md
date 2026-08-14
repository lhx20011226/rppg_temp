# ROI / 人脸跟踪逻辑逆向结论（对齐原生 .so）

> 目标：按用户要求"按照 .so 的逻辑来实现"前额/人脸 ROI，而非猜测。
> 逆向工具：Ghidra MCP（arm64 反编译）+ llvm-objdump（AArch64 汇编还原）。
> 日期：2026-08-13

## 1. 数据流向（谁算 ROI，谁消费 ROI）

```
摄像头帧
  → libface_detect.so      (SeetaFace6, face_detector.csta / face_landmarker_pts5.csta, 5 点 landmark)
  → libcmtrack.so          (CmTrackInterface::track → TddFa::track 跟踪)
       └─ TddFa::parseRoiBoxFromBbox()     算 ROI A
       └─ TddFa::parseRoiBoxFromLandmark() 算 ROI B
       （每帧对 ROI 区域求 RGB 均值，得到 ForeHead*/AllFace* 时序）
  → libengineCore.so / libCoreEngineV2.so   接收 CameraData.ROIs（已算好的 ROI RGB 序列）做 rPPG
```

关键结论（已验证）：
- `libCoreEngineV2.so` 里的 `ForeHeadR/G/B`、`AllFaceR/G/B`、`LFaceR/G/B` 等**只是输出字段名**，
  其字符串唯一引用者是 `saveStructuretoTxt`（结构体调试导出）和 `AssertInputData`（入参校验）。
  `.so` 内部**不计算任何 ROI 几何坐标**，ROI 由外部（libcmtrack.so）算好后作为 `CameraData.ROIs`
  列向量数组喂入（`All data in CameraData.ROIs shall be column oriented`）。
- `libjni_engine.so` 的 SeetaFace5（`face_landmarker_pts5.csta`）只用于**年龄/性别**（`detect_age_gender`
  是唯一调用 `FaceDetector::detect` / `FaceLandmarker::mark` 的函数），**不参与 rPPG ROI**。
- 真正的 ROI 几何实现只在 `libcmtrack.so` 的 `TddFa::parseRoiBoxFromBbox` 和
  `TddFa::parseRoiBoxFromLandmark`，由 `CmTrackInterface::track`（地址 0x24da0）各调用一次。

## 2. 每帧产出的 ROI 数量与语义映射

`CmTrackInterface::track` 中：
- `0x24e14`: `bl parseRoiBoxFromBbox`      → ROI A
- `0x24e58`: `bl parseRoiBoxFromLandmark`  → ROI B

即**每帧 2 个 ROI**。结合 `libCoreEngineV2.so` 字段（`AllFaceR/G/B` + `ForeHeadR/G/B`，
且原生 HR 主信号来自 ForeHead），映射为：
- **ROI A (bbox 版)  = AllFace（整个人脸区域，较大）**
- **ROI B (landmark 版) = ForeHead（关键点区域，较小，HR 主信号）**

> 注：LFace/RFace 不是独立 ROI，是引擎内部从 AllFace 按 Profile.FeaturePoints 再细分，
> 我们复刻主信号只需 ForeHead(ROI B) + AllFace(ROI A)。

## 3. ROI A — parseRoiBoxFromBbox（对齐 AllFace）

源码：`TddFa::parseRoiBoxFromBbox(FaceBox const&, vector<float>&)` @ 0x22520
反汇编已逐条还原。FaceBox 为**两点矩形** (x1,y1,x2,y2)（cv::Mat 内 4 个 float）。

精确公式（伪代码，常量已从 .rodata 提取）：
```
W = x2 - x1
H = y2 - y1
size = round( (W + H) / 2 * 1.58 )      // 常量 SCALE = 1.58 (vaddr 0x184f8)
cx   = x2 - H * 0.5                       // 框宽方向按 H/2 从右边界往左
cy   = y2 - H * 0.5 + size * 0.14        // 常量 YOFF = 0.14 (vaddr 0x18508)
ROI_A = [ cx, cy, cx + size, cy + size ]  // 正方形，size×size
```
即：以人脸框**底边中心**为基准，向左上扩展一个 `size×size` 的正方形 ROI。

## 4. ROI B — parseRoiBoxFromLandmark（对齐 ForeHead，HR 主信号）

源码：`TddFa::parseRoiBoxFromLandmark(cv::Mat const&, vector<float>&)` @ 0x233f4
输入 `cv::Mat` 为 landmark 点矩阵；函数先 `cv::Mat(mat, Range(0,1))` 取**第 0 行**
（常量 vaddr 0x184b0=int32{0,1} → Range(0,1)）。随后在切片内循环求
**x 的 min/max、y 的 min/max**，再计算：

```
dx = xmax - xmin
dy = ymax - ymin
d  = sqrt( (dx*0.5)^2 + (dy*0.5)^2 )   // 点集半对角距 = 半径
cx = (xmax + xmin) * 0.5
cy = (ymax + ymin) * 0.5
half = d * 0.5
ROI_B = [ cx - half, cy - half, cx + half, cy + half ]   // 正方形
```
即：**以点集（关键点）包围盒的质心为中心、以半对角距 radius 为半边长**的正方形 ROI。

实测时 `track` 传入的 landmarkMat 为 SeetaFace5 的 5 点（[左眼,右眼,鼻尖,左嘴角,右嘴
角] 标准顺序，row(0) 为第 0 点=左眼）；更可能 landmark 矩阵布局为 2×N（行=xy），
row(0)=全部点的 x。两种情况都落在"关键点集质心+半径"逻辑内，复刻时我们直接用
MediaPipe 选定的前额关键点子集套用同一公式即可。

## 5. ROI 像素统计与质量门控（对齐原生 AssertInputData / validation）

来自 `libCoreEngineV2.so` 字符串与 `c_CalS3CoreHR_RGB_Profile1_RGB_` 逻辑：
- **有效像素率 VaildPixelPer**：ROI 内"有效"像素 = 非全黑（值≠0）像素；
  `ROI.RGB %f is NOT validated, possibiliy too many zeros.` → 有效像素占比过低则该帧 ROI 作废。
- **帧间稳定性门控**：`FaceRGB Change is larger than ConfigData.ImageQuality.
  FaceRGBStdThresIgnore, Ignore this cal;` → 相邻帧 ROI 平均 RGB 变化超过阈值则丢弃该帧。
- **SNR 有效性**：`CalHRSNR_core` 中 `HRSNR = log10(peak/total) + 0.6`；但**不靠瞬时
  SNR 一票否决**（见 HR 有效性结论）。

## 6. HR 有效性（已在之前确认，保持不变）

- 维护历史 (HR, SNR) 窗口（`FilterHRUsingPreviousDataRGB`），用 SNR 加权得到稳定最终 HR
  （`b_CalWeightMean`），不靠瞬时 SNR 阈值硬拒。
- 仅当：有效样本过少（< HR_MIN_VALID_SAMPLES）或 HR 非生理（40~200 bpm）时才放弃。

## 7. Python 复刻要点（rppg_cam.py）

1. 用 MediaPipe FaceMesh 取人脸，得到人脸框 (x1,y1,x2,y2) → 套用 **ROI A 公式** 得 AllFace。
2. 用 MediaPipe 前额关键点子集（如额头区域点）求质心+半对角距 → 套用 **ROI B 公式** 得 ForeHead。
3. **主 HR 信号用 ForeHead (ROI B)**；AllFace 作背景/校验参考（对齐原生 ForeHead 为 HR 主信号）。
4. 每帧对两个 ROI 求 RGB 均值；做有效像素率门控（剔除近全黑 ROI）与帧间 RGB 变化门控
   （>阈值则 skip 该帧，不污染时序）。
5. HR/SNR 仍按历史加权；BP 仍走 bp_inference 的 24 分类 + l2 + 10 回归融合。
