# -*- coding:utf-8 -*-
"""
Standalone Demo: Sign-Sparse Offset Representation (SSOR) / Single-Array Offset Encoding
用于您的论文演示：如何用单交叉阵列（结合一列虚拟列/外围加法器）彻底替代开销极高的差分对。

核心思想：
对于卷积层卷积 y = W * X。
如果 W 有负数，差分对需要 W_pos 和 W_neg 两个交叉阵列（增加 100% 面积）。

本方案只需 1 个阵列：
1. 计算每片滤波器的偏置 Z = |min(W)| 
2. 映射正向权重 W_mapped = W + Z （即阵列里全是正数！）
3. 动态极性翻转：如果某个通道绝大多数是负数（导致翻过去的 W_mapped 和 Z 过大），
   直接整体把该通道权重反号 W = -W，降低基座。
4. 在计算阶段跑 y_mapped = W_mapped * X。并在数字外围或者通过一根 Dummy 虚拟列减去 Z * sum(X)。
   
通过该实现，您可以完美拿到理论等效的特征输出，彻底省掉一半的器件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OffsetSingleArrayConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # --- 模拟单交叉阵列 Physical Crossbar Array ---
        # 尺寸：(out_channels, in_channels * kernel_size * kernel_size)
        # 所有写入此阵列的值严格 >= 0，不需要任何额外的负阵列差分对（1x面积）！
        self.weight_mapped = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        
        # --- 阵列补偿位 (Dummy Reference Columns) ---
        # 每个输出滤波器仅需额外维护一个超小开销的值 Z（或者在数字边侧用寄存器存着）
        self.Z_offset = nn.Parameter(torch.zeros(out_channels, 1, 1, 1))
        
        # --- 极性翻转标记 (Polarity Array) ---
        # 1 bit SRAM 记录这个通道是否在部署前为了压低功耗被翻转过正负号
        self.polarity = nn.Parameter(torch.ones(out_channels, 1, 1, 1), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def map_from_standard_weights(self, standard_weight, dynamic_polarity_flip=True):
        """
        在模型部署推演前期进行“权重编译/烧录 (Weight Programming)”
        """
        with torch.no_grad():
            B, C, H, W = standard_weight.shape
            flattened = standard_weight.view(B, -1)
            
            # 计算这一通道的最负面值和最正面值
            min_vals = flattened.min(dim=1)[0]
            max_vals = flattened.max(dim=1)[0]
            
            z_vals = torch.zeros_like(min_vals)
            polarities = torch.ones_like(min_vals)
            final_weights = standard_weight.clone()
            
            for i in range(B):
                # 如果启用动态极性翻转：核心是为了缩小后续被映射进 ReRAM 后，整体的“高电导基准态”
                # 假设分布极端偏负 (-10 到 +1)，如果不翻转，Z=10，全体垫高 10个单位。
                # 翻转后变成 (-1 到 +10)，Z=1，全体只垫高 1个单位，能节省极大的静态交叉电流。
                if dynamic_polarity_flip and abs(min_vals[i]) > abs(max_vals[i]):
                    polarities[i] = -1.0
                    final_weights[i] = -final_weights[i]
                    z_cur = abs(final_weights[i].min()) if final_weights[i].min() < 0 else 0
                else:
                    z_cur = abs(min_vals[i]) if min_vals[i] < 0 else 0
                    
                z_vals[i] = z_cur

            # W_mapped 全部严格被垫到了 >= 0（全部化作正常的导通电阻，没有任何纠结的补码位）
            w_mapped = final_weights + z_vals.view(B, 1, 1, 1)
            
            self.weight_mapped.copy_(w_mapped)
            self.Z_offset.copy_(z_vals.view(B, 1, 1, 1))
            self.polarity.copy_(polarities.view(B, 1, 1, 1))

    def forward(self, x):
        # =========================================================================
        # 步骤 1： 阵列主力运算域 (Analog Compute Array) - 无任何差分对！
        # I_main = W_mapped * X  (所有电阻皆为正向有效电导，电流同向无抵消)
        # =========================================================================
        main_array_out = F.conv2d(x, self.weight_mapped, None, self.stride, self.padding)
        
        # =========================================================================
        # 步骤 2： 旁路补偿/加法器聚合 \sum X
        # 在同一个视野内，由于所有的特征 X 的和是共享的，这只需要求一次即可。
        # 硬件上用一条列全部置 1 的 Dummy Column 轻松积出这个电流。
        # =========================================================================
        sum_kernel = torch.ones_like(self.weight_mapped[0:1]) 
        sum_X = F.conv2d(x, sum_kernel, None, self.stride, self.padding) # Shape: (B, 1, H_out, W_out)
        
        # 抵消电流 = \sum X 乘以各通道对应的垫底值 Z
        # 这可以在 ADC 后使用很便宜的一个数字减法执行。
        # sum_X has shape (batch_size, 1, H_out, W_out)
        # self.Z_offset has shape (1, out_channels, 1, 1) -> wait, it is initialized as (out_channels, 1, 1, 1). We need to reshape it for calculation.
        Z_reshaped = self.Z_offset.view(1, self.out_channels, 1, 1)
        polarity_reshaped = self.polarity.view(1, self.out_channels, 1, 1)
        
        # 抵消电流 = \sum X 乘以各通道对应的垫底值 Z
        # 这可以在 ADC 后使用很便宜的一个数字减法执行。
        offset_current = sum_X * Z_reshaped
        
        # =========================================================================
        # 步骤 3： 输出汇聚与极性还原校正
        # =========================================================================
        # 无修饰等效结果
        out = main_array_out - offset_current
        
        # 利用 polarity 记录（1或-1）来将此前反转的张量纠正回来，这甚至可以用简单的倒相器实现
        out = out * polarity_reshaped
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
            
        return out


# ================================== 测试论证模块 ==================================
def main():
    print("====== 论文仿真论证：混合架构单阵列偏移编码有效性实验 ======")
    torch.manual_seed(42)
    B, in_C, H, W = 2, 64, 14, 14
    out_C = 128
    kernel_size = 3
    
    # 构造假输入和随机带负数的标准卷积核
    x = torch.rand(B, in_C, H, W)
    std_conv = nn.Conv2d(in_C, out_C, kernel_size, padding=1) # 默认初始化权重涵盖正负
    
    # 1. 常规全精度输出 (如果用差分对，也应该是这个理论结果，但要双倍器件)
    target_out = std_conv(x)
    
    # 2. 我们构建 SSOR 偏移全正相架构
    offset_conv = OffsetSingleArrayConv2d(in_C, out_C, kernel_size, padding=1, bias=True)
    offset_conv.bias.data.copy_(std_conv.bias.data)
    
    # 将包含负值的传统权重，映射并硬烧录到这个只有单一合法(>=0)通道的虚拟阵列里
    offset_conv.map_from_standard_weights(std_conv.weight.data, dynamic_polarity_flip=True)
    
    # 3. 在单阵列模拟器中跑特征提取
    ssor_out = offset_conv(x)
    
    # 4. 对比论证（理论误差应该无限接近 0，由于浮点误差可能在 1e-6 级别）
    max_error = torch.max(torch.abs(target_out - ssor_out)).item()
    
    print("\n[对比结果]")
    print(f"最大特征偏差绝对误差 (Max Abs Error) : {max_error}")
    if max_error < 1e-4:
        print("结论: >>> 验证通过！完美在模拟和数字融合中等效了差分对网络。 <<<")
        
    print("\n[面积开销优势评估 (Area overhead)]")
    # 一个输出通道 = 1行
    # diff_pair: 每行有 W_pos 列和 W_neg 列
    cols_per_row = in_C * kernel_size * kernel_size
    diff_pair_cells = out_C * (cols_per_row * 2) 
    
    # SSOR：仅有 1个正向矩阵 + 补偿偏移位(每个通道极端的1bit反转标识+外围小数)
    # 不严格按比特来算，只按核心电阻单元来算：
    ssor_cells = out_C * cols_per_row 
    dummy_cells = out_C * 1   # 用于存Z，极端保守算它挂载阵列旁边
    
    print(f"1. 传统差分对需求 ReRAM 物理单元 : {diff_pair_cells} 个器件")
    print(f"2. SSOR 偏移法需求 ReRAM 物理单元 : {ssor_cells + dummy_cells} 个器件")
    print(f"-> 硬件节约率 : {1 - (ssor_cells + dummy_cells)/diff_pair_cells:.2%}")

if __name__ == "__main__":
    main()
