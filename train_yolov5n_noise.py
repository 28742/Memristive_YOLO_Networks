# -*- coding:utf-8 -*-
"""
Experiment 3: Noise-Aware Fine-Tuning (NAT) for YOLOv5
通过劫持/替换 PyTorch 的 Conv2d，在每次前向传播时（仅在训练模式下）为权重注入物理等效噪声。
反向传播将根据带噪声的前向结果计算梯度，从而促使模型寻找到平坦极小值 (Flat Minima)，大幅提升抗噪能力。

基于用户指定的原生训练脚本：/home/zrc/.cache/torch/hub/ultralytics_yolov5_v6.2/train.py
"""

import os
import sys
import math
import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. 核心：带噪声注入的 Conv2d
# ---------------------------------------------------------
class NoisyConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 目标信噪比 10dB
        self.noise_snr_db = 10.0

    def forward(self, input):
        # 仅在训练阶段注入噪声
        if self.training:
            with torch.no_grad():
                # SNR(dB) = 10 * log10( Var(W) / Var(N) )
                # => std(N) = std(W) / (10^(SNR/20))
                factor_linear = 10 ** (self.noise_snr_db / 20.0)
                noise_std = self.weight.std() / factor_linear
                
            # 注入与权重同分布的高斯噪声，且必须保留梯度传播！
            noisy_weight = self.weight + torch.randn_like(self.weight) * noise_std
            return self._conv_forward(input, noisy_weight, self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)

# ---------------------------------------------------------
# 2. 运行时劫持模块
# ---------------------------------------------------------
nn.Conv2d = NoisyConv2d
print(f"[NAT Experiment] PyTorch nn.Conv2d 被成功劫持！前向传播将注入 10dB 物理噪声。")

# ---------------------------------------------------------
# 3. 动态导入并拉起 YOLOv5 训练
# ---------------------------------------------------------
if __name__ == '__main__':
    # YOLOv5 训练脚本路径
    YOLOV5_HUB_PATH = "/home/zrc/.cache/torch/hub/ultralytics_yolov5_v6.2"
    
    if YOLOV5_HUB_PATH not in sys.path:
        sys.path.insert(0, YOLOV5_HUB_PATH)
        
    # 此时内部初始化的所有 Conv2d 都会变成 NoisyConv2d
    import train 
    from utils.general import print_args
    
    opt = train.parse_opt(known=True)
    
    # ==== 强行配置微调参数 ====
    # 指定预训练权重（接力上一轮的权重）
    opt.weights = "/home/zrc/MemIntelli/runs/train/exp_noise_aware/weights/last.pt" 
    # 数据集配置，使用 inference 脚本中的路径
    import yaml
    coco_yaml_path = os.path.join(YOLOV5_HUB_PATH, "data", "coco.yaml")
    custom_yaml_path = "/home/zrc/MemIntelli/custom_coco.yaml"
    with open(coco_yaml_path, "r") as f:
        coco_cfg = yaml.safe_load(f)
    
    coco_cfg["path"] = "/data/dataset/coco"
    coco_cfg["train"] = "images/train2017"
    coco_cfg["val"] = "images/val2017"
    coco_cfg["test"] = "images/test2017"
    if "download" in coco_cfg:
        del coco_cfg["download"]
        
    with open(custom_yaml_path, "w") as f:
        yaml.safe_dump(coco_cfg, f)
        
    opt.data = custom_yaml_path
    
    # 如果用户没有通过 CLI 修改，使用微调友好的配置
    if opt.epochs == 300: # 默认 300
        opt.epochs = 20   # 噪声感知训练一般 20 Epoch 足矣
        
    opt.batch_size = 4
    opt.device = '0'  # 强制指定使用 GPU 0
    
    # ==== 配置基于 COCO 的微调超参数 ====
    # YOLOv5n 在 COCO 上的官方基准是 hyp.scratch-low.yaml
    base_hyp_path = os.path.join(YOLOV5_HUB_PATH, "data", "hyps", "hyp.scratch-low.yaml")
    with open(base_hyp_path, "r") as f:
        custom_hyp = yaml.safe_load(f)
    
    # 针对 10dB 噪声感知微调，我们需要降低基准学习率 (默认是 0.01)
    # 防止高学习率冲刷掉原本收敛好的特征判断能力
    custom_hyp['lr0'] = 0.001      # 初始学习率设为 1e-3 (针对 Fine-tuning)
    custom_hyp['lrf'] = 0.01       # 最终学习率系数也会一同衰减
    
    custom_hyp_path = "/home/zrc/MemIntelli/custom_hyp_finetune.yaml"
    with open(custom_hyp_path, "w") as f:
        yaml.safe_dump(custom_hyp, f)
        
    opt.hyp = custom_hyp_path
    opt.project = "/home/zrc/MemIntelli/runs/train"
    opt.name = "exp_noise_aware_20e"  # 新开一个文件夹，以免覆盖第一轮的结果
    opt.exist_ok = True
    
    print("\n" + "="*50)
    print("🚀 启动 Noise-Aware Fine-Tuning (NAT) 🚀")
    print(f"目标权重: {opt.weights}")
    print(f"数据集: {opt.data}")
    print(f"训练轮数: {opt.epochs}")
    print("="*50 + "\n")
    
    # 拉起训练主进程
    try:
        train.main(opt)
    except Exception as e:
        print(f"训练异常: {e}")
