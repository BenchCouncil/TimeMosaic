import os
import numpy as np
import matplotlib.pyplot as plt

# ======== 配置区域 ========
# model_list = ['AGPT', 'PatchTST_4', 'PatchTST_16', 'PatchTST_32']
model_list = ['PatchTST_4', 'PatchTST_16', 'PatchTST_32']
# pred_len_list = [96, 192, 336, 720]
pred_len_list = [96]
label_len = 96  # 输入序列长度（输入+预测的分界线），请根据实际模型设置调整
data_root = '../results'  # 每个模型文件夹的上层目录
sample_index = 50  # 哪个样本
channel_index = 2  # 哪个通道

save_dir = './figures'
os.makedirs(save_dir, exist_ok=True)

# ======== 主绘图逻辑 ========
for pred_len in pred_len_list:
    plt.figure(figsize=(10, 6))
    true_seq = None
    input_seq = None

    for model in model_list:
        folder_name = f"{model}_{pred_len}"
        pred_path = os.path.join(data_root, folder_name, 'pred.npy')
        true_path = os.path.join(data_root, folder_name, 'true.npy')

        if not os.path.exists(pred_path) or not os.path.exists(true_path):
            print(f"[Warning] Missing: {folder_name}, skipped.")
            continue

        pred = np.load(pred_path)  # [B, pred_len, C]
        true = np.load(true_path)

        pred_y = pred[sample_index, :, channel_index]
        true_y = true[sample_index, :, channel_index]

        if true_seq is None:
            true_seq = true_y
        if input_seq is None:
            # 尝试从 true 推断输入部分：即 true 序列前的 label_len 长度
            # 你可以替换为实际输入数据路径以获得更精确的输入部分
            input_seq = true_y[:label_len]

        plt.plot(range(label_len, label_len + pred_len), pred_y, label=model)

    if true_seq is not None and input_seq is not None:
        # Ground Truth（灰线）
        full_gt = np.concatenate([input_seq, true_seq])
        plt.plot(range(label_len + pred_len), full_gt, '--', color='black', linewidth=1.8, label='Ground Truth')

        # 输入部分（用灰色）
        plt.plot(range(label_len), input_seq, color='gray', linewidth=1.5, label='Input')

        # 分界线
        plt.axvline(x=label_len - 1, color='gray', linestyle=':', linewidth=1)

    plt.title(f'Prediction Length = {pred_len}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'comparison_predlen_{pred_len}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"✅ 所有图已保存至: {save_dir}")