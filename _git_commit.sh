#!/bin/bash
set -e
cd /workspace/rppg_project
git init -q 2>/dev/null || true
git add -A
echo "=== staged file count ==="
git diff --cached --name-only | wc -l
echo "=== staged total size (approx) ==="
git diff --cached --numstat | awk '{a+=$1} END{print a" lines added"}'
echo "=== commit ==="
git commit -q -m "rPPG offline engine 复现: 摄像头心率 + 真实血压(mmHg)

- 逆向 calc_bp: 10个回归模型(lightgbm+SVR)融合输出真实收缩压/舒张压 mmHg
- validate_lbp/hbp 血压基线分组中心值+区间夹逼
- bp_inference.py: 加载回归模型 + 融合 + 基线校验
- rppg_cam.py: 摄像头 CHROM+rPPG 心率血压
- repro_rppg_bp.py: 离线 demo
- 修复 lightgbm 反序列化 booster_ 未回填 + 锁 lightgbm==3.3.2" && echo "COMMIT_DONE"
