import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_single_attn_map(attns, batch_size, n_vars,
                         layer=0, sample_index=0, var_index=0, head_index=0,
                         save_path=None, show=False):
    """
    仅绘制一张 attention heatmap 图。

    Parameters:
        attns: List[Tensor], 每层是 [B*C, H, L, L]
        batch_size: 实验中的 batch size
        n_vars: 通道数（features）
        layer: 第几层的 attention
        sample_index: 第几个样本（0 ~ batch_size-1）
        var_index: 第几个变量（通道）
        head_index: 第几个 head
        save_path: 保存图片路径（如果为 None 就不保存）
        show: 是否直接显示图（默认否）
    """
    attn_tensor = attns[layer]  # [B*C, H, L, L]
    BxC, H, L, _ = attn_tensor.shape
    assert BxC == batch_size * n_vars, f"attns 的 batch_size 不匹配: got {BxC}, expected {batch_size * n_vars}"

    # 还原成 [B, C, H, L, L]
    attn_tensor = attn_tensor.view(batch_size, n_vars, H, L, L)

    # 取出目标 attention map
    attn_map = attn_tensor[sample_index, var_index, head_index]  # shape: [L, L]

    # 绘图
    plt.figure(figsize=(5, 4))
    sns.heatmap(attn_map.detach().cpu(), cmap="viridis", cbar=True)
    plt.title(f"Layer {layer} | Sample {sample_index} | Var {var_index} | Head {head_index}")
    plt.xlabel("Key Patch")
    plt.ylabel("Query Patch")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved attention map to {save_path}")
        plt.close()
    elif show:
        plt.show()
    else:
        print("Plot not shown or saved. Use save_path or set show=True.")
