import os
import re

log_dir = "logs"
prefix = "ETTm2_96_336"
best_mse = float("inf")
best_mae = float("inf")
best_mse_file = ""
best_mae_file = ""
best_mse_mae = None
best_mae_mse = None

for filename in os.listdir(log_dir):
    if filename.startswith(prefix) and filename.endswith(".log"):
        filepath = os.path.join(log_dir, filename)
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                # 提取指标
                match = re.search(r"mse:(\S+), mae:(\S+)", content)
                if match:
                    mse = float(match.group(1).rstrip(','))
                    mae = float(match.group(2).rstrip(','))
                    if mse < best_mse:
                        best_mse = mse
                        best_mse_file = filename
                        best_mse_mae = mae  # 同步记录对应 mae
                    if mae < best_mae:
                        best_mae = mae
                        best_mae_file = filename
                        best_mae_mse = mse  # 同步记录对应 mse
        except Exception as e:
            print(f"读取失败：{filename}, 原因：{e}")

print(f"✅ 最小 MSE 文件: {best_mse_file} (MSE = {best_mse}, MAE = {best_mse_mae})")
print(f"✅ 最小 MAE 文件: {best_mae_file} (MAE = {best_mae}, MSE = {best_mae_mse})")
