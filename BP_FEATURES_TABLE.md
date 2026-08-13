# rPPG 血压：74 维 PPG 特征定义 与 各血压模型 features.txt 映射

## 一、74 维 PPG 单周期特征定义 (extract_ppg_features_from_single_ppg_cycle)

> 索引 i 对应 `ppg_single_cycle_features[i]`；x=采样点索引, y=PPG幅值, fs=900Hz；
> v_l/v_r=左/右波谷, peak=主峰, a/b/e=apg(a波/b波/e波)波峰, inflection=重搏切迹。

| idx | 含义 | idx | 含义 |
|-----|------|-----|------|
| 0 | 周期总时长(s)=(v_r-v_l)/fs | 1 | 收缩期时长(s)=(peak-v_l)/fs |
| 2 | 舒张期时长(s)=(v_r-peak)/fs | 3 | 收缩期占比=sys/dur |
| 4 | (deriv_peak-v_l)/(peak-v_l) | 5 | (e_pos-v_l)/dur |
| 6 | (a_pos-v_l)/(peak-v_l) | 7 | (b_pos-a_pos)/(peak-v_l) |
| 8 | (inflection-e_pos)/(v_r-e_pos) | 9 | 收缩期宽度@0.1*peak(s) |
| 10 | 收缩期宽度@0.25 | 11 | 收缩期宽度@0.33 |
| 12 | 收缩期宽度@0.5 | 13 | 收缩期宽度@0.66 |
| 14 | 收缩期宽度@0.75 | 15 | 舒张期宽度@0.1 |
| 16 | 舒张期宽度@0.25 | 17 | 舒张期宽度@0.33 |
| 18 | 舒张期宽度@0.5 | 19 | 舒张期宽度@0.66 |
| 20 | 舒张期宽度@0.75 | 21 | 一阶导数宽度@0.25 |
| 22 | 收缩期面积(trapz) | 23 | 舒张期面积(trapz) |
| 24 | 总面积=sys+dias | 25 | v_l~deriv_peak面积 |
| 26 | deriv_peak~peak面积 | 27 | peak~e_pos面积 |
| 28 | e_pos~v_r面积 | 29 | dn面积比 |
| 30 | 收缩期平均斜率 | 31 | 舒张期平均斜率 |
| 32 | 收缩期斜率(y) | 33 | 舒张期斜率(y) |
| 34 | 最大上升斜率=deriv_peak | 35 | e_pos处曲率 |
| 36 | apg[a] | 37 | apg[b] |
| 38 | apg[e] | 39 | apg[b]/apg[a] |
| 40 | apg[e]/apg[a] | 41 | agi=ratio_b_a-ratio_e_a |
| 42 | AI=y[inflection]/y[peak] | 43 | (inflection-peak)/fs |
| 44 | delta_t/dur | 45 | 一阶导数宽度@0.75 |
| 46 | 左宽@0.33 | 47 | 左宽@0.66 |
| 48 | 右宽@0.33 | 49 | 右宽@0.66 |
| 50 | 导数正面积(左) | 51 | 导数正面积(右) |
| 52 | peak~e_pos导数负面积 | 53 | a_pos~deriv_peak斜率 |
| 54 | deriv_peak~b_pos斜率 | 55 | peak~e_pos斜率 |
| 56 | deriv[inflection] | 57 | deriv[inflection]/slope_max |
| 58 | sys宽@0.33(ext) | 59 | sys宽@0.66(ext) |
| 60 | dias宽@0.33(ext) | 61 | dias宽@0.66(ext) |
| 62 | 左宽@0.33(ext) | 63 | 左宽@0.66(ext) |
| 64 | 右宽@0.33(ext) | 65 | 右宽@0.66(ext) |
| 66 | usdc前20%均值 | 67 | usdc中位数 |
| 68 | usdc后20%均值 | 69 | usdc标准差 |
| 70 | dsdc前20%均值 | 71 | dsdc中位数 |
| 72 | dsdc后20%均值 | 73 | dsdc标准差 |

**窗口特征会删除的索引**（get_avg_ppg_features 中 features_idx_deleted）：
`f6, f8, f37, f38, f39, f40, f41, f52, f56, f57, f66, f67, f68, f69, f70, f71`  → 进入模型的窗口特征维度 = 74-16 = 58。

## 二、各命名血压模型的 features.txt 映射（选用哪些 74 维特征）

> 来源：`runtime.zip → dmi_b220818_d220919_v220428/bp_models/<模型>/<hash>/features.txt`
> 命名规则：`models_ch{身高}_算法` / `models_cl{身高}`(儿童) / `models_r{l/h}{w/0/1}`(风险回归)。身高单位 cm，每 5cm 一档。

| 模型 | 用特征数 | 选用的 f 编号 |
|------|--------|--------------|
| models_ch110_ie | 28 | f73, f75, f61, f74, f59, f30, f51, f63, f60, f20, f5, f29, f65, f52, f37, f79, f6, f47, f28, f48, f25, f24, f13, f3, f11, f49, f16, f34 |
| models_ch115_ie | 25 | f62, f75, f61, f59, f21, f51, f60, f63, f29, f65, f66, f79, f47, f54, f26, f15, f45, f14, f12, f11, f49, f18, f55, f17, f16 |
| models_ch120_ie | 23 | f43, f75, f51, f64, f21, f60, f63, f20, f65, f19, f6, f54, f37, f27, f14, f26, f23, f12, f79, f48, f50, f55, f4 |
| models_ch120_lgbm | 25 | f62, f75, f43, f61, f59, f30, f74, f21, f20, f29, f56, f65, f66, f6, f36, f54, f45, f27, f47, f13, f26, f18, f16, f8, f46 |
| models_ch125_ie | 26 | f62, f73, f59, f75, f51, f64, f60, f30, f65, f56, f52, f19, f37, f54, f47, f15, f25, f13, f12, f49, f11, f79, f55, f8, f17, f46 |
| models_ch125_l2 | 13 | f5, f8, f3, f7, f6, f9, f2, f4, f10, f11, f15, f13, f14 |
| models_ch125_lgbm | 26 | f62, f73, f43, f61, f75, f59, f51, f30, f21, f63, f5, f29, f65, f56, f19, f37, f47, f25, f23, f11, f49, f17, f8, f50, f46, f32 |
| models_ch125_svc | 24 | f62, f43, f61, f59, f75, f51, f64, f30, f74, f5, f65, f66, f56, f52, f19, f36, f27, f37, f49, f48, f17, f50, f2, f46 |
| models_ch130_ie | 18 | f62, f43, f61, f75, f59, f51, f21, f29, f66, f6, f19, f25, f47, f49, f48, f10, f50, f28 |
| models_ch135_ie | 39 | f62, f73, f43, f75, f59, f61, f51, f30, f64, f60, f63, f21, f5, f74, f20, f56, f66, f65, f29, f52, f6, f36, f45, f19, f37, f26, f54, f27, f14, f13, f11, f28, f55, f50, f4, f17, f3, f1, f34 |
| models_ch140_dt_ie | 17 | f73, f61, f75, f5, f74, f20, f66, f52, f19, f45, f37, f54, f12, f49, f18, f32, f16 |
| models_ch140_ie | 30 | f73, f75, f30, f51, f60, f64, f21, f74, f5, f20, f29, f66, f6, f19, f45, f37, f47, f54, f24, f25, f27, f15, f48, f14, f44, f11, f49, f10, f18, f17 |
| models_ch145_ie | 27 | f62, f51, f64, f21, f63, f74, f20, f66, f65, f29, f56, f52, f19, f36, f24, f25, f47, f27, f13, f23, f12, f48, f55, f44, f17, f8, f2 |
| models_ch150_ie | 17 | f43, f51, f21, f20, f75, f64, f65, f19, f24, f54, f15, f37, f13, f18, f11, f34, f44 |
| models_ch155_ie | 29 | f61, f62, f73, f43, f59, f51, f60, f64, f20, f5, f75, f29, f56, f65, f19, f36, f24, f37, f25, f47, f18, f48, f12, f11, f49, f55, f32, f4, f3 |
| models_ch160_ie | 27 | f62, f61, f73, f59, f30, f64, f21, f60, f20, f75, f29, f65, f66, f52, f37, f47, f24, f25, f55, f15, f28, f13, f18, f46, f79, f3, f50 |
| models_cl70_ie | 34 | f21, f5, f56, f59, f20, f66, f52, f61, f73, f29, f62, f75, f43, f64, f30, f15, f19, f25, f14, f24, f13, f54, f11, f74, f17, f2, f6, f33, f1, f22, f34, f35, f79, f55 |
| models_cl72_ie | 34 | f21, f60, f5, f20, f66, f65, f73, f51, f59, f56, f52, f62, f30, f75, f19, f24, f25, f63, f27, f15, f14, f8, f23, f18, f11, f17, f50, f2, f33, f31, f26, f28, f44, f55 |
| models_cl75_ie | 35 | f62, f60, f5, f73, f59, f51, f61, f20, f52, f64, f65, f66, f75, f30, f63, f29, f19, f27, f74, f15, f14, f25, f23, f6, f8, f10, f18, f26, f33, f31, f1, f34, f46, f4, f47 |
| models_cl77_ie | 31 | f21, f61, f20, f62, f30, f43, f51, f29, f56, f65, f66, f52, f75, f74, f24, f6, f25, f63, f27, f23, f12, f49, f11, f17, f16, f2, f3, f79, f4, f46, f55 |
| models_cl77_l2 | 8 | f4, f3, f6, f5, f1, f7, f2, f8 |
| models_cl77_lgbm | 32 | f21, f73, f5, f20, f43, f56, f30, f51, f29, f65, f52, f75, f64, f63, f19, f24, f25, f27, f15, f14, f54, f23, f17, f32, f16, f1, f33, f45, f26, f46, f4, f55 |
| models_cl80_ie | 31 | f20, f56, f61, f60, f29, f62, f30, f5, f65, f66, f51, f52, f75, f19, f64, f25, f24, f6, f23, f11, f49, f74, f8, f10, f34, f16, f33, f31, f3, f35, f4 |
| models_cl82_ie | 35 | f56, f20, f60, f30, f29, f61, f5, f62, f73, f51, f59, f66, f65, f52, f75, f6, f25, f24, f27, f64, f14, f13, f23, f63, f10, f74, f17, f50, f79, f35, f45, f46, f48, f26, f55 |
| models_cl85_ie | 34 | f21, f62, f73, f43, f61, f20, f60, f30, f56, f59, f5, f51, f29, f65, f66, f75, f64, f6, f19, f74, f24, f23, f49, f18, f17, f10, f50, f16, f79, f26, f22, f46, f47, f37 |
| models_cl85_lgbm | 30 | f21, f43, f73, f61, f30, f20, f5, f51, f56, f65, f64, f6, f63, f75, f19, f74, f24, f79, f27, f14, f23, f12, f49, f50, f32, f1, f2, f22, f46, f28 |
| models_rh0_lgbm | 23 | f73, f62, f74, f75, f61, f30, f79, f59, f64, f29, f51, f63, f6, f19, f37, f25, f23, f27, f18, f50, f3, f1, f33 |
| models_rh0_svr | 22 | f73, f62, f75, f30, f21, f20, f59, f64, f29, f60, f5, f6, f56, f37, f52, f28, f13, f10, f50, f31, f4, f2 |
| models_rh1_lgbm | 21 | f62, f73, f59, f30, f60, f51, f21, f64, f75, f74, f56, f66, f6, f37, f27, f44, f18, f28, f32, f16, f33 |
| models_rh1_svr | 22 | f62, f61, f59, f30, f51, f63, f20, f75, f56, f65, f52, f19, f45, f24, f47, f27, f23, f12, f32, f4, f8, f50 |
| models_rhw_lgbm | 22 | f73, f61, f59, f75, f30, f51, f21, f74, f63, f5, f20, f29, f65, f66, f36, f15, f14, f44, f55, f79, f17, f2 |
| models_rl0_lgbm | 18 | f19, f24, f29, f51, f5, f15, f61, f54, f23, f12, f56, f52, f22, f34, f75, f55, f4, f37 |
| models_rl0_svr | 20 | f20, f19, f61, f24, f25, f18, f51, f1, f3, f15, f28, f23, f12, f65, f56, f33, f4, f74, f37, f63 |
| models_rl1_lgbm | 18 | f43, f62, f30, f61, f29, f65, f59, f52, f45, f75, f25, f8, f26, f13, f1, f34, f47, f10 |
| models_rl1_svr | 22 | f61, f73, f20, f29, f60, f65, f66, f5, f51, f19, f24, f75, f63, f15, f27, f14, f3, f23, f49, f54, f36, f44 |
| models_rlw_lgbm | 18 | f20, f61, f5, f59, f66, f65, f52, f75, f24, f74, f15, f14, f23, f17, f79, f35, f4, f46 |