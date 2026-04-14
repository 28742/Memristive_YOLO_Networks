#!/bin/bash

# ==============================================================================
# MemIntelli YOLOv5 系统评估自动化脚本
# 评估全量 60 个卷积层在量化与 5% 写入噪声下的参数偏差
# ==============================================================================

PROJECT_ROOT="/home/zrc/MemIntelli"
PYTHON_EXEC="/home/zrc/anaconda3/envs/yolov5/bin/python"
EXP_SCRIPT="$PROJECT_ROOT/examples/compare_models_exp.py"

echo "[1/2] 开始执行全量卷积层对比实验..."
$PYTHON_EXEC "$EXP_SCRIPT"

if [ $? -eq 0 ]; then
    echo "--------------------------------------------------------"
    echo "实验完成！"
    echo "导出的模型: yolov5n_orig.pt, yolov5n_quant.pt, yolov5n_noise.pt"
    echo "分析结果图: systematic_impact_comparison.png"
    echo "--------------------------------------------------------"
else
    echo "实验执行失败，请检查 Python 环境或路径。"
    exit 1
fi
