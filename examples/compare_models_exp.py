# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import sys
import os
from tqdm import tqdm

# Ensure we import the local MemIntelli source tree
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memintelli.NN_models.YOLOv5 import YOLOv5_zoo
from memintelli.pimpy.memmat_tensor import DPETensor
from memintelli.NN_layers import Conv2dMem

def get_model(pt_path, device, mem_enabled=False, write_var=0.0, read_var=0.0):
    # Initialize Engine
    engine = DPETensor(
        write_variation=write_var,
        read_variation=read_var,
        vnoise=0.0,
        rate_stuck_HGS=0.0,
        rate_stuck_LGS=0.0,
        device=device
    )
    
    # 我们不传入 weights_path，让 YOLOv5_zoo 使用默认的 torch.hub pretrain 机制
    # 这样才能保证所有参数完美匹配且三份模型初始权重一致！
    model = YOLOv5_zoo(
        model_name="yolov5n",
        engine=engine,
        input_slice=(1, 1, 2, 4),
        weight_slice=(1, 1, 2, 4),
        device=device,
        mem_enabled=mem_enabled,
        pretrained=True,
        weights_path=None
    )
    model.eval()
    return model

def compare_params(model_orig, model_quant, model_noise):
    print("\n" + "="*50)
    print("全量卷积层 (60层) 系统评估对比实验 (多 Slice 累加恢复对比)")
    print("="*50)
    
    results = []
    
    orig_modules = dict(model_orig.named_modules())
    quant_modules = dict(model_quant.named_modules())
    noise_modules = dict(model_noise.named_modules())
    
    target_layers = [name for name, m in quant_modules.items() if isinstance(m, Conv2dMem)]
    
    print(f"检测到 {len(target_layers)} 个卷积层进行分析...")

    for name in target_layers:
        m_orig = orig_modules[name]
        m_quant = quant_modules[name]
        m_noise = noise_modules[name]
        
        # 1. 原始全精度权重 (FP32)
        orig_w = m_orig.weight.detach().cpu()
        orig_w_flat = orig_w.view(orig_w.shape[0], -1).t()
        
        # 2. 量化累加恢复权重分析 (Reconstructed Total Weight)
        # SlicedData.quantized_data 是内部恢复后的 float 权重 (即所有 slice 累加后的效果)
        # 它是基于量化算法映射后的理想浮点值
        quant_w = m_quant.weight_sliced.quantized_data.detach().cpu()
        if quant_w.shape != orig_w_flat.shape:
             quant_w = quant_w[:orig_w_flat.shape[0], :orig_w_flat.shape[1]]
        
        mse_quant = torch.mean((orig_w_flat - quant_w)**2).item()
        snr_quant = 10 * torch.log10(torch.sum(orig_w_flat**2) / (torch.sum((orig_w_flat - quant_w)**2) + 1e-12)).item()
        
        # 3. 噪声累加恢复权重分析 (Physical Noise Impact on Total Weight)
        # 我们需要从噪声注入后的电导 G 恢复成权重，模拟计算后的结果
        # G shape: (..., num_slice, m, n)
        # slice_weights shape: (num_slice,)
        # max_data: 每个分块的最大值 (用于 Scale)
        
        # 电导恢复权重的物理公式: W_noisy = (Sum(G_noisy[i] * slice_weights[i]) - Offset) * Scale
        G_noisy = m_noise.weight_sliced.G.detach().cpu()
        sliced_weights = m_noise.weight_sliced.sliced_weights.detach().cpu()
        max_data = m_noise.weight_sliced.max_data.detach().cpu()
        
        # 展平 slice weights 以便广播相乘
        sw = sliced_weights.view(1, 1, -1, 1, 1)
        # 这里的 G 对应的是电导，我们需要根据 SlicedData 的逻辑反推回权重数值
        # 简化版：W_from_G = Sum(G_noisy * sw) / Sum(G_ideal_max * sw) * max_data
        
        # 获取理想最大电导层级 (HGS - LGS)
        engine = m_noise.engine
        # 建议在 compare_params 循环中修改：
        G_ideal = (G_noisy - engine.LGS) / engine.Q_G # 对应 level_indices
        # 恢复到 0-1 范围
        G_norm = G_ideal / (engine.g_level - 1) 
        # G_range = engine.HGS - engine.LGS
        
        # # 恢复带噪声的完整权重 (Total Noisy Weight)
        # # 1. 去除电导偏置 LGS 并归一化到 0-1
        # G_norm = (G_noisy - engine.LGS) / G_range
        
        # 模拟恢复过程
        max_weights = m_noise.weight_sliced.sliced_max_weights.detach().cpu().view(1, 1, -1, 1, 1)
        # 每个 slice 的理想数值 = G_norm * max_weights
        val_reconstructed = torch.sum((G_norm * max_weights) * sw, dim=2)
        # 归一化并乘回原量纲
        bits = sum(m_noise.weight_sliced.slice_method)
        total_range = torch.tensor(2**(bits - 1) - 1, dtype=torch.float32, device=val_reconstructed.device)
        
        # 此时 max_data shape 是 (num_r, num_c, 1, 1), 可以通过 squeeze 维度对齐
        # 但是 val_reconstructed shape 是 (num_r, num_c, paral_r, paral_c)
        # max_data 的原有维度是 (batch, num_r, num_c, 1, 1)，如果是 2D 就是 (num_r, num_c, 1, 1)
        # 广播没问题
        noise_w_recovered = (val_reconstructed / total_range) * max_data.view(max_data.shape[0], max_data.shape[1], 1, 1)
        
        # 处理 Padding 与分块拼接: (num_r, num_c, p_r, p_c) -> (num_r * p_r, num_c * p_c)
        noise_w_recovered = noise_w_recovered.transpose(1, 2).reshape(
            noise_w_recovered.shape[0] * noise_w_recovered.shape[2],
            noise_w_recovered.shape[1] * noise_w_recovered.shape[3]
        )
        noise_w_recovered = noise_w_recovered[:orig_w_flat.shape[0], :orig_w_flat.shape[1]]
        
        mse_noise = torch.mean((quant_w - noise_w_recovered)**2).item()
        snr_noise = 10 * torch.log10(torch.sum(quant_w**2) / (torch.sum((quant_w - noise_w_recovered)**2) + 1e-12)).item()
        
        results.append({
            'name': name,
            'snr_quant': snr_quant,
            'snr_noise': snr_noise,
            'sign_flips': torch.sum((quant_w * noise_w_recovered) < 0).item(),
            'total_params': quant_w.numel()
        })

    # 统计平均值
    avg_snr_quant = sum(r['snr_quant'] for r in results) / len(results)
    avg_snr_noise = sum(r['snr_noise'] for r in results) / len(results)
    total_flips = sum(r['sign_flips'] for r in results)
    total_elements = sum(r['total_params'] for r in results)
    flip_rate = (total_flips / total_elements) * 100
    
    print(f"\n[汇总结果 - {len(results)} 层平均]")
    print(f"平均量化信噪比 (SNR Quant): {avg_snr_quant:.2f} dB")
    print(f"平均噪声信噪比 (SNR Noise): {avg_snr_noise:.2f} dB")
    print(f"权重符号翻转数: {total_flips} / {total_elements} ({flip_rate:.4f}%)")

    # 绘图部分
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        x = np.arange(len(results))
        snr_q = [r['snr_quant'] for r in results]
        snr_n = [r['snr_noise'] for r in results]
        
        # SNR 趋势对比
        ax1.plot(x, snr_q, label='Quantization SNR (vs FP32)', marker='o', color='blue')
        ax1.axhline(y=avg_snr_quant, color='blue', linestyle='--', alpha=0.5)
        ax1.set_title('Quantization Impact across 60 Layers')
        ax1.set_ylabel('SNR (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(x, snr_n, label='Noise SNR (vs Ideal Quant)', marker='x', color='orange')
        ax2.axhline(y=avg_snr_noise, color='orange', linestyle='--', alpha=0.5)
        ax2.set_title('Physical Noise Impact across 60 Layers')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_xlabel('Layer Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Systematic Evaluation: Quantization vs Noise (60 Conv Layers)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = "systematic_impact_comparison.png"
        plt.savefig(plot_path)
        print(f"\n全量层实验结果图片已保存至: {plot_path}")
        
    except ImportError:
        print("\n[Warning] 未安装 matplotlib，无法生成图片。")

    print("\n全系统结论分析:")
    if avg_snr_quant < avg_snr_noise:
        print(f"-> 在全部 {len(results)} 层中，量化误差 (平均 {avg_snr_quant:.2f}dB) 依然是主要降级因素。")
    else:
        print(f"-> 物理噪声 (平均 {avg_snr_noise:.2f}dB) 的波动在某些层中可能更加显著。")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt_path = "/home/zrc/yolov5n.pt" # 假设根目录下有权重
    
    if not os.path.exists(pt_path):
        # 尝试从 MemIntelli 目录找
        pt_path = "/home/zrc/MemIntelli/yolov5n.pt"
        
    print(f"使用权重文件: {pt_path}")

    # 1. 原始模型 (Software 模式)
    print("正在加载原始模型...")
    model_orig = get_model(pt_path, device, mem_enabled=False)
    
    # 2. 量化模型 (Mem 模式, 但 variation=0)
    print("正在加载量化模型 (无噪声, 带有方案A微弱权重修剪)...")
    model_quant = get_model(pt_path, device, mem_enabled=True, write_var=0.0)
    
    threshold = 0.005  # 你可以将此值作为超参数进行调节
    with torch.no_grad():
        for name, param in model_quant.named_parameters():
            if 'weight' in name:
                mask = (param < 0) & (param > -threshold)
                param[mask] = 0.0
                
    model_quant.update_weight() # 显式触发量化映射
    
    # 3. 量化+噪声模型 (Mem 模式, variation=0.05)加上方案A截断
    print("正在加载量化+噪声模型 (var=0.05，带有方案A微弱负权重修剪)...")
    model_noise = get_model(pt_path, device, mem_enabled=True, write_var=0.05)
    
    # ==== 方案 A 消融加入点 ====
    print(f"-> [消融 A] 正在将 (-{threshold}, 0) 的微负权重强制截断为 0.0")
    with torch.no_grad():
        pruned_count, total_count = 0, 0
        for name, param in model_noise.named_parameters():
            if 'weight' in name:
                mask = (param < 0) & (param > -threshold)
                pruned_count += mask.sum().item()
                total_count += param.numel()
                param[mask] = 0.0
    print(f"-> [消融 A] 置零参数占比: {pruned_count} / {total_count} ({pruned_count/total_count*100:.4f}%)")
    # ========================

    model_noise.update_weight() # 显式触发噪声注入
    
    # 导出模型权重 (State Dict)
    torch.save(model_orig.state_dict(), "yolov5n_orig.pt")
    torch.save(model_quant.state_dict(), "yolov5n_quant.pt")
    torch.save(model_noise.state_dict(), "yolov5n_noise.pt")
    print("\n已导出三个模型文件: yolov5n_orig.pt, yolov5n_quant.pt, yolov5n_noise.pt")

    # 执行对比实验
    compare_params(model_orig, model_quant, model_noise)

if __name__ == "__main__":
    main()
