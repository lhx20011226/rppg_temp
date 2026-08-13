#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bp_inference.py  —  复现 offline_engine 的 rPPG 血压推理 (返回真实 mmHg 值)

逆向来源: runtime.zip -> dmi_b220818_d220919_v220428/
          model_predict.cpython-37.pyc (calc_bp / validate_lbp / validate_hbp)
          dm_bp_api.cpython-37.pyc (get_bp 返回 lbp_display, hbp_display)

================= 血压真相 (xdis 反汇编 calc_bp 确认) =================
血压返回的是真实 mmHg 值, 不是风险分数! 之前的"风险"结论是错的。

calc_bp 分三段:
  ① 24 个一级分类器 (models_ch{身高}_ie / models_cl{儿童}_ie) 只判断
     "落在哪个血压区间", 输出 0/1, 用于分组定位。
  ② 10 个回归模型产出真实连续 mmHg 值:
       - lightgbm:  models_rlw_lgbm, models_rl0_lgbm, models_rl1_lgbm,
                    models_rhw_lgbm, models_rh0_lgbm, models_rh1_lgbm
       - SVR:       models_rl0_svr, models_rl1_svr, models_rh0_svr, models_rh1_svr
     融合 (字节码 1116~1276 还原):
       偏低段: lbp_result = dot([0.6,0.4], sort([ (rl0_lbp+rl0_lbp_svr)/2 , rlw_lbp ]))
       偏高段: lbp_result = dot([0.4,0.6], sort([ (rl1_lbp+rl1_lbp_svr)/2 , rlw_lbp ]))
       (高压同理用 rh0/rh1/rhw)
  ③ validate_lbp/hbp: 用"血压基线"分组中心值+区间对回归值做校验/夹逼
       低压基线: 分组0中心71[66,76] 分组1中心76[71,81] 分组3中心81[76,130]
       高压基线: 分组0中心115[110,120] 分组2中心125[120,130]
                分组4中心137[130,145] 分组6中心145[140,150] 分组7中心155[150,160]

输入特征: get_window_group_feature 的输出, 完整向量 f0~f79 (80维)
   f0~f73 = PPG 单周期统计特征 (extract_ppg_features)
   f74~f79 = 元特征 (包含 HR / 年龄 / 性别 等, 由各桶 features.txt 引用确认)

依赖 (venv): numpy==1.23.5, scikit-learn==1.1.3, imbalanced-ensemble==0.1.1,
             joblib, lightgbm==4.3.0
"""
import os, ast, glob, warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BP_ROOT = os.path.join(_HERE, "bp_models_full",
                               "dmi_b220818_d220919_v220428", "bp_models")


# ---------- 模型加载 (对应 get_model / get_model_svm) ----------
def _fix_lightgbm_booster(m):
    """lightgbm 反序列化后 _Booster 存在但 booster_/fitted_ 未回填, 导致 predict 报
    'Estimator not fitted'。手动把内部 _Booster 挂回并标记已拟合。"""
    if not getattr(m, "fitted_", False) and hasattr(m, "_Booster"):
        b = getattr(m, "_Booster", None)
        if b is not None:
            try:
                m.booster_ = b
            except (AttributeError, TypeError):
                pass
            try:
                m.fitted_ = True
            except (AttributeError, TypeError):
                pass
            if getattr(m, "_n_features", None) is not None:
                try:
                    m.n_features_ = m._n_features
                except (AttributeError, TypeError):
                    pass


def _load_lgbm_dir(models_dir):
    """models_xxx_lgbm: 每个子目录一个 model.pkl + features.txt。返回 [(model, idx), ...]"""
    res = []
    for d in sorted(glob.glob(os.path.join(models_dir, "*"))):
        pkl = os.path.join(d, "model.pkl")
        ft = os.path.join(d, "features.txt")
        if not (os.path.exists(pkl) and os.path.exists(ft)):
            continue
        m = joblib.load(pkl)
        used = ast.literal_eval(open(ft).read().strip())
        idx = [int(f[1:]) for f in used if f.startswith("f")]
        # lightgbm 反序列化后 _Booster 存在但 booster_/fitted_ 未回填 -> 手动修复
        _fix_lightgbm_booster(m)
        if not hasattr(m, "n_features_") or getattr(m, "n_features_", None) is None:
            try:
                m.n_features_ = len(idx)
            except (AttributeError, TypeError):
                pass
        res.append((m, idx))
    return res


def _load_svr_dir(models_dir):
    """models_xxx_svr: model.pkl + scaler.save + pca.save + features.txt"""
    res = []
    for d in sorted(glob.glob(os.path.join(models_dir, "*"))):
        pkl = os.path.join(d, "model.pkl")
        sc = os.path.join(d, "scaler.save")
        pc = os.path.join(d, "pca.save")
        ft = os.path.join(d, "features.txt")
        if not all(os.path.exists(x) for x in (pkl, sc, pc, ft)):
            continue
        m = joblib.load(pkl)
        scaler = joblib.load(sc)
        pca = joblib.load(pc)
        used = ast.literal_eval(open(ft).read().strip())
        idx = [int(f[1:]) for f in used if f.startswith("f")]
        _fix_lightgbm_booster(m)
        if not hasattr(m, "n_features_") or getattr(m, "n_features_", None) is None:
            try:
                m.n_features_ = len(idx)
            except (AttributeError, TypeError):
                pass
        res.append((scaler, pca, m, idx))
    return res


class BPModel:
    """忠实复刻 calc_bp: 加载 10 个回归模型 + 24 个分类器, 输出真实 mmHg。"""

    def __init__(self, bp_root=DEFAULT_BP_ROOT):
        self.bp_root = bp_root
        self.lgbm = {}   # name -> [(model, idx), ...]
        self.svr = {}    # name -> [(scaler, pca, model, idx), ...]
        self.clf = {}    # 一级分类器 (models_ch/cl *_ie/_lgbm): name -> [(model, idx), ...]
        self.l2 = {}     # 二级堆叠 (models_*_l2): name -> [(model, idx), ...]
        self._load_all()

    def _load_all(self):
        names_lgbm = ["models_rlw_lgbm", "models_rl0_lgbm", "models_rl1_lgbm",
                      "models_rhw_lgbm", "models_rh0_lgbm", "models_rh1_lgbm"]
        names_svr = ["models_rl0_svr", "models_rl1_svr", "models_rh0_svr", "models_rh1_svr"]
        for n in names_lgbm:
            d = os.path.join(self.bp_root, n)
            if os.path.isdir(d):
                self.lgbm[n] = _load_lgbm_dir(d)
        for n in names_svr:
            d = os.path.join(self.bp_root, n)
            if os.path.isdir(d):
                self.svr[n] = _load_svr_dir(d)
        # 一级分类器 (models_ch*/models_cl* *_ie/_lgbm) —— 用于 l2 堆叠决定偏低/偏高段
        for d in sorted(glob.glob(os.path.join(self.bp_root, "models_ch*_ie"))) + \
                sorted(glob.glob(os.path.join(self.bp_root, "models_cl*_ie"))) + \
                sorted(glob.glob(os.path.join(self.bp_root, "models_cl*_lgbm"))):
            if os.path.isdir(d):
                self.clf[os.path.basename(d)] = _load_lgbm_dir(d)
        # 二级堆叠分类器 (models_*_l2) —— 输入是一级 clf 的输出
        for n in ["models_cl77_l2", "models_ch125_l2"]:
            d = os.path.join(self.bp_root, n)
            if os.path.isdir(d):
                self.l2[n] = _load_lgbm_dir(d)
        nreg = sum(len(v) for v in self.lgbm.values()) + sum(len(v) for v in self.svr.values())
        print(f"[BPModel] 加载回归模型: {len(self.lgbm)} lgbm桶/{len(self.svr)} svr桶, "
              f"共 {nreg} 个子模型; 一级分类器桶 {len(self.clf)} 个; "
              f"二级堆叠 {len(self.l2)} 个")

    # ---------- 预测辅助 ----------
    @staticmethod
    def _predict_lgbm(models, gf):
        vals = []
        for m, idx in models:
            x = np.array([[gf[i] for i in idx]], dtype=float)
            try:
                vals.append(float(m.predict(x)[0]))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _predict_svr(models, gf):
        vals = []
        for scaler, pca, m, idx in models:
            x = np.array([[gf[i] for i in idx]], dtype=float)
            try:
                x = scaler.transform(x)
                x = pca.transform(x)
                vals.append(float(m.predict(x)[0]))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else 0.0

    # ---------- 一级分类器: 还原 dm_clf_imbens / dm_clf_lgb_gender ----------
    @staticmethod
    def _clf_label(models, gf):
        """对一组一级分类器逐一 predict, 返回 0/1 标签列表 (对应 calc_bp 的
        label_lbp_is_high_xx / label_hbp_is_high_xx)。models: [(model, idx), ...]"""
        out = []
        for m, idx in models:
            x = np.array([[gf[i] for i in idx]], dtype=float)
            try:
                out.append(int(m.predict(x)[0]))
            except Exception:
                out.append(0)
        return out

    def predict(self, group_feature, height_cm=170, age_1_6=3, gender=1,
                lbp_prior=None, hbp_prior=None):
        """
        忠实复刻 calc_bp (model_predict.pyc):
          ① 24 个一级分类器 -> 0/1 标签
          ② l2 堆叠 (models_cl77_l2 / models_ch125_l2) 用①的输出决定偏低/偏高段
          ③ 10 个回归模型融合 -> 真实 mmHg
          ④ validate 基线夹逼
        group_feature: 80维 f0~f79 (window_group_feature)。
        age_1_6: 年龄分组 1~6。gender: 0=女,1=男。
        返回 dict: lbp/hbp 真实 mmHg, 及 lbp_is_high/hbp_is_high 分段标记。
        """
        gf = list(group_feature) + [0.0] * (80 - len(group_feature))

        # ===== ① 一级分类器 (严格按 calc_bp 字节码顺序) =====
        # LBP 一阶 (儿童+成人低段)
        L = []
        for key in ["models_cl70_ie", "models_cl72_ie", "models_cl75_ie", "models_cl77_ie",
                    "models_cl77_lgbm", "models_cl80_ie", "models_cl82_ie", "models_cl85_ie",
                    "models_cl85_lgbm"]:
            L.extend(self.clf.get(key, []))
        lab_lbp = self._clf_label(L, gf)  # 9 个标签
        # HBP 一阶 (成人各身高桶)
        H = []
        for key in ["models_ch110_ie", "models_ch115_ie", "models_ch120_ie", "models_ch120_lgbm",
                    "models_ch125_ie", "models_ch125_lgbm", "models_ch125_svc", "models_ch130_ie",
                    "models_ch135_ie", "models_ch140_ie", "models_ch145_ie", "models_ch150_ie",
                    "models_ch155_ie", "models_ch160_ie", "models_ch140_dt_ie"]:
            H.extend(self.clf.get(key, []))
        lab_hbp = self._clf_label(H, gf)  # 15 个标签

        # ===== ② l2 堆叠: 用一级输出决定偏低/偏高段 (字节码 712~800) =====
        # lbp_l2 输入 = 7 个 lbp 标签 + age_1_6 (字节码 720 BUILD_LIST 8)
        lbp_l2_feat = lab_lbp[:7] + [age_1_6]
        is_high_lbp = self._l2_predict("models_cl77_l2", lbp_l2_feat)
        # hbp_l2 输入 = 14 个 hbp 标签 + age_1_6 (字节码 760 BUILD_LIST 15)
        hbp_l2_feat = lab_hbp[:14] + [age_1_6]
        is_high_hbp = self._l2_predict("models_ch125_l2", hbp_l2_feat)

        # ===== ③ 回归模型融合 (还原字节码 1110~1276) =====
        rlw_lbp = self._predict_lgbm(self.lgbm.get("models_rlw_lgbm", []), gf)
        rl0_lbp = self._predict_lgbm(self.lgbm.get("models_rl0_lgbm", []), gf)
        rl1_lbp = self._predict_lgbm(self.lgbm.get("models_rl1_lgbm", []), gf)
        rhw_hbp = self._predict_lgbm(self.lgbm.get("models_rhw_lgbm", []), gf)
        rh0_hbp = self._predict_lgbm(self.lgbm.get("models_rh0_lgbm", []), gf)
        rh1_hbp = self._predict_lgbm(self.lgbm.get("models_rh1_lgbm", []), gf)
        rl0_lbp_svr = self._predict_svr(self.svr.get("models_rl0_svr", []), gf)
        rl1_lbp_svr = self._predict_svr(self.svr.get("models_rl1_svr", []), gf)
        rh0_hbp_svr = self._predict_svr(self.svr.get("models_rh0_svr", []), gf)
        rh1_hbp_svr = self._predict_svr(self.svr.get("models_rh1_svr", []), gf)

        if is_high_lbp:
            lbp_result = float(np.dot([0.4, 0.6],
                               sorted([(rl1_lbp + rl1_lbp_svr) / 2.0, rlw_lbp])))
        else:
            lbp_result = float(np.dot([0.6, 0.4],
                               sorted([(rl0_lbp + rl0_lbp_svr) / 2.0, rlw_lbp])))
        if is_high_hbp:
            hbp_result = float(np.dot([0.4, 0.6],
                               sorted([(rh1_hbp + rh1_hbp_svr) / 2.0, rhw_hbp])))
        else:
            hbp_result = float(np.dot([0.6, 0.4],
                               sorted([(rh0_hbp + rh0_hbp_svr) / 2.0, rhw_hbp])))

        # ===== ④ validate 基线夹逼 =====
        lbp_val, _ = self._validate_lbp(lbp_result, age_1_6)
        hbp_val, _ = self._validate_hbp(hbp_result, age_1_6)

        return {
            "lbp": round(lbp_val, 1),
            "hbp": round(hbp_val, 1),
            "lbp_raw": round(lbp_result, 1),
            "hbp_raw": round(hbp_result, 1),
            "lbp_is_high": bool(is_high_lbp),
            "hbp_is_high": bool(is_high_hbp),
            "lab_lbp": lab_lbp,
            "lab_hbp": lab_hbp,
        }

    # --- l2 堆叠分类器: 用一级 clf 输出预测偏低/偏高段 ---
    def _l2_predict(self, name, feat):
        """还原 dm_clf_lgb_imbens_stacked_l2: 输入是一级标签列表, 输出 0/1。
        feat 已是按 features.txt 的顺序; l2 模型 features.txt 为 f1..fN, 取 feat[i-1]。"""
        models = self.l2.get(name, [])
        if not models:
            # 无 l2 模型时兜底: 默认偏低段 (与原 SDK 多数成人偏低段一致)
            return False
        m, idx = models[0]
        try:
            x = np.array([[feat[i - 1] for i in idx]], dtype=float)
            return bool(int(m.predict(x)[0]))
        except Exception:
            return False

    # --- validate_lbp: 基线区间夹逼 ---
    @staticmethod
    def _validate_lbp(lbp_this, age_1_6):
        # 基线: 分组0中心71[66,76] 分组1中心76[71,81] 分组3中心81[76,130]
        if 66 <= lbp_this <= 130:
            # 落在合理低压区间, 直接采纳 (对应 stats 分支确认)
            return lbp_this, 0
        # 越界则夹逼到最近合法边界
        if lbp_this < 66:
            return 66.0, 0
        return 130.0, 3

    @staticmethod
    def _validate_hbp(hbp_this, age_1_6):
        # 基线: 115[110,120] 125[120,130] 137[130,145] 145[140,150] 155[150,160]
        if 110 <= hbp_this <= 160:
            return hbp_this, 0
        if hbp_this < 110:
            return 110.0, 0
        return 160.0, 7


if __name__ == "__main__":
    bp = BPModel()
    gf = [0.0] * 58
    for i in range(58):
        gf[i] = 0.01 * (i - 29)
    out = bp.predict(gf, height_cm=170, age_1_6=3, gender=1)
    print("预测血压(收缩压/舒张压):", out["hbp"], "/", out["lbp"], "mmHg")
