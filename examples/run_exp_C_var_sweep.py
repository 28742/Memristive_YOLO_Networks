import subprocess
import re
import csv
import os
import sys

def main():
    variations = [0.0, 0.02, 0.05, 0.07, 0.1]
    csv_file = "diff_pair_vs_default_results.csv"

    # 初始化CSV文件并写入表头
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Write Variation", "Architecture", "mAP50-95", "mAP50"])

    print(f"=== 开始对比实验：默认补码 vs 差分对 (变异系数: {variations}) ===")
    print(f"结果将实时写入: {csv_file}\n")

    for var in variations:
        # 分别运行 默认模式(False) 和 差分对模式(True)
        for use_diff in [False, True]:
            arch_name = "Differential Pair (Sign-Magnitude)" if use_diff else "Default (Two's Complement)"
            print(f"[*] 正在运行测试 -> Variation: {var}, 架构: {arch_name}")
            
            cmd = ["python", "11_yolov5_coco_inference.py", "--mem-enabled", "--var", str(var)]
            if use_diff:
                cmd.append("--diff-pair")
            
            # 使用Popen实时捕获输出
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            map_5095 = "Error"
            map_50 = "Error"
            
            for line in process.stdout:
                # 可以取消注释下面这行来在后台日志中实时查看YOLO验证进度条，但这可能会让log变大
                # print(line, end="") 
                
                # 正则匹配 mAP 指标
                match_5095 = re.search(r"mAP50-95:\s+([0-9.]+)", line)
                if match_5095:
                    map_5095 = float(match_5095.group(1))
                
                match_50 = re.search(r"mAP50\s+:\s+([0-9.]+)", line)
                if match_50:
                    map_50 = float(match_50.group(1))
            
            process.wait()
            
            print(f"    => mAP50-95: {map_5095}, mAP50: {map_50}")
            
            # 每跑完一个配置，实时追加到 CSV 中，防止中途中断
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([var, arch_name, map_5095, map_50])

    print(f"\n=== 所有实验运行完毕，结果已保存至 {csv_file} ===")

if __name__ == "__main__":
    main()
