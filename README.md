# rPPG 人脸心率 + 血压 复现项目

基于 `offline_engine-release.zip` (cmcr offline engine) 逆向结论实现的最小可运行 rPPG 系统：
调用本地摄像头 → 人脸 ROI → RGB 时序 → CHROM 提取 PPG → 心率 + 血压(模型)。

## 目录结构
```
rppg_project/
├── rppg_venv/            # 已配置好的虚拟环境 (linux x86_64 / py3.11.1)
├── bp_models_full/       # 从 runtime.zip 解压的全部血压模型 (joblib .pkl + features.txt)
│   └── dmi_b220818_d220919_v220428/bp_models/...
├── rppg_cam.py           # 摄像头主程序 (心率+血压)
├── repro_rppg_bp.py      # 离线 demo (合成 PPG 跑通整条链路)
├── bp_inference.py       # 血压模型推理封装 (joblib 加载 + 身高分桶集成)
├── features_74.py        # 74 维 PPG 特征定义
├── requirements.txt      # 依赖锁定版本
└── README.md
```

## 快速开始
```bash
cd rppg_project
source rppg_venv/bin/activate        # 激活 venv (已装好所有依赖)

# 1) 离线 demo (无需摄像头, 验证整条链路)
python repro_rppg_bp.py

# 2) 摄像头实时测量 (需本地有摄像头)
python rppg_cam.py --seconds 30 --height 170 --gender 1
#   参数: --camera 索引(默认0)  --seconds 采集秒数  --height 身高cm  --gender 0/1
```

## 调用链 (对齐逆向结论)
```
摄像头帧
 → mediapipe FaceMesh 取人脸 ROI
 → 每帧 ROI 的 RGB 平均 → RGB 时序
 → CHROM 去运动伪影 → PPG              (对应 libCoreEngineV2.so::Calculate_PPG_IBI)
 → 带通滤波 + FFT 峰值 → 心率 HR       (对应 CalculateHRVIndex_RGB 的 HR)
 → 按波谷切 cycle + 74维特征 + 窗口/组  (对应 extract_ppg_features)
 → BPModel.predict(group_feature, height) → 血压  (对应 dm_bp_api.get_bp)
```

## 血压模型说明 (逆向关键发现)
- 模型在 `runtime.zip → dmi_.../bp_models/`, 是一组 **joblib 保存的 sklearn / imbalanced-ensemble /
  lightgbm / SVR** 模型 (.pkl)。
- `calc_bp` (xdis 反汇编确认) 三段式:
  ① 24 个一级分类器 (`models_ch{身高}_ie` / `models_cl{儿童}_ie`) 只判断"落在哪个血压区间" (输出 0/1)。
  ② **10 个回归模型产出真实连续 mmHg 值**:
     lightgbm (`models_rlw/rl0/rl1/rhw/rh0/rh1_lgbm`) + SVR (`models_rl0/rl1/rh0/rh1_svr`)。
     融合公式 (字节码 1116~1276 还原):
       `lbp_result = dot([0.6,0.4], sort([ (rl0_lbp+rl0_lbp_svr)/2 , rlw_lbp ]))`  (偏低段)
       `lbp_result = dot([0.4,0.6], sort([ (rl1_lbp+rl1_lbp_svr)/2 , rlw_lbp ]))`  (偏高段)
       (高压同理用 rh0/rh1/rhw)
  ③ `validate_lbp/hbp` 用"血压基线"分组中心值+区间对回归值做校验/夹逼:
     低压基线 分组0中心71[66,76] 分组1中心76[71,81] 分组3中心81[76,130]
     高压基线 分组0中心115[110,120] 分组2中心125[120,130] 分组4中心137[130,145]
             分组6中心145[140,150] 分组7中心155[150,160]
  **结论: 血压返回的是真实 mmHg 值 (lbp_display / hbp_display, 即舒张压/收缩压), 不是风险分数。**
- **复现必需 patch** (已在 `bp_inference.py` 内):
  * imbalanced-ensemble 0.1.1 的 `SelfPacedEnsembleClassifier` joblib 反序列化丢失 `n_features_` → 手动补回。
  * lightgbm 模型反序列化后 `_Booster` 存在但 `booster_/fitted_` 未回填 → 手动把 `_Booster` 挂回并设 `fitted_=True`
    (否则 predict 报 "Estimator not fitted")。需 `lightgbm==3.3.2` (4.x 同样有此问题)。
- 依赖 `scikit-learn==1.1.3` + `numpy==1.23.5` + `lightgbm==3.3.2`; 不可用新版 numpy (与 scipy/sklearn 冲突)。

## 复现数值说明
- 心率: PPG 的 FFT 主频, 单位 bpm。
- 血压: `bp_inference.BPModel.predict(...)` 直接返回真实 **收缩压 / 舒张压 (mmHg)**, 单位准确。
  完整链路: 74 维 PPG 特征 → 窗口/组特征 → 补元特征 f74=HR / f75=年龄 / f79=性别 →
  10 个回归模型融合 → validate 基线夹逼 → `(hbp, lbp) mmHg`。
  注: 真实数值依赖 rPPG 提取的 74 维特征分布与训练一致; demo 用合成 PPG 仅验证链路正确, 不代表临床精度。

## 单独重建 venv (若下载的 venv 不可用)
```bash
python3 -m venv rppg_venv
rppg_venv/bin/pip install -r requirements.txt
# 然后把 bp_models_full/ 放到本目录
```
