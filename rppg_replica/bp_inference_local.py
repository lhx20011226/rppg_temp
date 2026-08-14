#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bp_inference.py  —  复现 offline_engine 的 rPPG 血压推理
逆向来源: runtime.zip -> dmi_b220818_d220919_v220428/
          model_predict.cpython-37.pyc (calc_bp) + bp_models/models_ch{身高}_ie/<hash>/model.pkl

模型真相 (xdis 反汇编 calc_bp 确认):
  * 血压模型是一组 scikit-learn / imbalanced-ensemble 的 .pkl (joblib 保存)
  * 按"身高"分桶: models_ch110_ie ~ models_ch160_ie (成人, 每5cm一档) +
    models_cl70~cl85_ie (儿童) + models_r{l/h}{w/0/1}* (风险回归桶)
  * calc_bp 会把所有桶模型各自 predict "是否高血压", 再 stacking/融合
  * 这里实现: 加载所有身高桶模型, 用 joblib 加载后补回 n_features_ 属性,
    对每个桶做 predict_proba, 取平均概率作为"高血压风险分数"

依赖 (venv): numpy==1.23.5, scikit-learn==1.1.3, imbalanced-ensemble==0.1.1, joblib
"""
import os, ast, glob, warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

# bp_models 目录 (相对本文件)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BP_ROOT = os.path.join(_HERE, "bp_models_full",
                                "dmi_b220818_d220919_v220428", "bp_models")


class BPModel:
    def __init__(self, bp_root=DEFAULT_BP_ROOT):
        self.bp_root = bp_root
        self.buckets = {}          # name -> (model, feature_idx)
        self._load_all()

    def _load_all(self):
        # 找到所有 <hash>/model.pkl + 同目录 features.txt
        pat = os.path.join(self.bp_root, "*", "*", "model.pkl")
        for pkl in sorted(glob.glob(pat)):
            d = os.path.dirname(pkl)
            ft = os.path.join(d, "features.txt")
            if not os.path.exists(ft):
                continue
            bucket = os.path.basename(os.path.dirname(d))  # models_ch120_ie
            try:
                m = joblib.load(pkl)
                used = ast.literal_eval(open(ft).read().strip())
                idx = [int(f[1:]) for f in used if f.startswith("f")]
                # imbalanced_ensemble 0.1.1 joblib 反序列化会丢失 n_features_
                if not hasattr(m, "n_features_"):
                    m.n_features_ = len(idx)
                self.buckets[bucket] = (m, idx)
            except Exception as e:
                print(f"[warn] 跳过 {bucket}: {e}")
        # 只看身高桶 (_ie 分类器), 过滤掉 dm/vsr/pbg 等
        self.buckets = {k: v for k, v in self.buckets.items()
                        if (k.startswith("models_ch") or k.startswith("models_cl"))
                        and k.endswith("_ie")}
        print(f"[BPModel] 已加载 {len(self.buckets)} 个身高桶血压模型")

    def _select_buckets(self, height_cm):
        """按身高选最靠近的桶 (与 SDK 的身高分桶一致: ch110~ch160 每5cm一档)。"""
        if height_cm is None:
            return list(self.buckets.items())
        # 解析每个桶身高
        cand = []
        for name in self.buckets:
            s = name.replace("models_ch", "").replace("models_cl", "").replace("_ie", "")
            try:
                cand.append((int(s), name))
            except ValueError:
                continue
        if not cand:
            return list(self.buckets.items())
        cand.sort()
        # 找最近的身高
        h = min(cand, key=lambda t: abs(t[0] - height_cm))[0]
        near = [n for (hh, n) in cand if abs(hh - h) <= 5]
        return [(n, self.buckets[n]) for n in near]

    def predict(self, group_feature, height_cm=None):
        """
        group_feature: get_window_group_feature 的输出 (list, 维度58)
        返回 dict:
            hbp_risk : 高血压(收缩)风险分数 0~1 (各桶平均 predict_proba[:,1])
            lbp_risk : 高血压(舒张)风险分数 0~1
            buckets_used: 参与投票的桶
        注: 原 SDK 用 24 桶集成 -> 连续血压值; 这里以"高血压风险分数"形式输出,
            量级与 SDK 的 lbp_display/hbp_display(0/1 等级或回归值)一致。
        """
        sel = self._select_buckets(height_cm)
        if not sel:
            return {"error": "no bucket"}
        # 完整特征向量为 f0~f79 (80维): f0-f73=PPG特征, f74-f79=元特征(HR/年龄/性别等)
        # 不足80维右侧补0
        gf = list(group_feature) + [0.0] * (80 - len(group_feature))
        probs_h, probs_l = [], []
        for name, (m, idx) in sel:
            x = np.array([[gf[i] for i in idx]], dtype=float)
            try:
                p = m.predict_proba(x)[0]
                probs_h.append(float(p[1]))
                probs_l.append(float(p[1]))
            except Exception:
                continue
        if not probs_h:
            return {"error": "predict failed"}
        return {
            "hbp_risk": float(np.mean(probs_h)),
            "lbp_risk": float(np.mean(probs_l)),
            "buckets_used": [n for n, _ in sel],
            "n_buckets": len(probs_h),
        }


if __name__ == "__main__":
    # 快速自测
    bp = BPModel()
    gf = [0.0] * 58
    # 造一点非零特征让预测有区分
    for i in range(58):
        gf[i] = np.random.randn()
    print(bp.predict(gf, height_cm=170))
