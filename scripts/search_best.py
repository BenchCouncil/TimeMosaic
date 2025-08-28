import os
import re
import argparse
import pandas as pd

# ----------------------
# 命令行参数
# ----------------------
parser = argparse.ArgumentParser(description="search")
parser.add_argument("--log_dir", type=str, default="./logs", help="日志目录")
parser.add_argument(
    "--select", type=str, default="min",
    help="{min 96,192,320,512}"
)
args = parser.parse_args()

log_dir = args.log_dir
select_mode = args.select.strip().lower()

allowed_seq_lens = {"96", "192", "320", "512"}
if select_mode != "min" and select_mode not in allowed_seq_lens:
    raise ValueError(f"--select 只能是 'min' 或 {sorted(allowed_seq_lens)} 之一，当前为：{args.select}")

# ----------------------
# 正则与容器
# ----------------------

pattern_metrics = re.compile(r"mse[:=]\s*([0-9.]+)[, ]+mae[:=]\s*([0-9.]+)", re.IGNORECASE)
pattern_filename = re.compile(
    r'(?P<dataset>\w+?)_(?P<seq_len>\d+)_(?P<pred_len>\d+)_(?P<d_model>\d+)_(?P<d_ff>\d+)_e(?P<e_layers>\d+)_ep(?P<epochs>\d+)_lr(?P<lr>[\deE\.-]+)_(?P<model>\w+)\.log$'
)

best_results = {}  # key: (dataset, model, pred_len) -> record dict

# ----------------------
# 遍历日志
# ----------------------
for filename in os.listdir(log_dir):
    if not filename.endswith(".log"):
        continue

    match_file = pattern_filename.match(filename)
    if not match_file:
        continue

    file_info = match_file.groupdict()

    # 如果指定了固定输入长度，仅保留该 seq_len
    if select_mode != "min" and file_info["seq_len"] != select_mode:
        continue

    key = (file_info['dataset'], file_info['model'], int(file_info['pred_len']))
    log_path = os.path.join(log_dir, filename)

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        match_metric = pattern_metrics.search(content)
        if not match_metric:
            continue
        mse = float(match_metric.group(1))
        mae = float(match_metric.group(2))
        avg_score = (mse + mae) / 2.0
        # avg_score = (mse) / 2.0
    except Exception:
        continue

    if (key not in best_results) or (avg_score < best_results[key]['avg_score']):
        # 记录最优（以 avg_score 为准）
        best_results[key] = {
            'dataset': file_info['dataset'],
            'model': file_info['model'],
            'pred_len': int(file_info['pred_len']),
            'seq_len': int(file_info['seq_len']),
            'd_model': int(file_info['d_model']),
            'd_ff': int(file_info['d_ff']),
            'e_layers': int(file_info['e_layers']),
            'epochs': int(file_info['epochs']),
            'lr': float(file_info['lr']),
            'mse': mse,
            'mae': mae,
            'avg_score': avg_score,
            'log_file': filename
        }

# ----------------------
# 输出结果
# ----------------------
if not best_results:
    print("未找到匹配的最优结果。请检查日志目录、文件命名或 --select 设置。")
    raise SystemExit(0)

df = pd.DataFrame(best_results.values())
df = df.sort_values(by=["dataset", "pred_len", "model"])

# 保存 CSV
out_csv = "best_results.csv" if select_mode == "min" else f"best_results_seq{select_mode}.csv"
df.to_csv(out_csv, index=False)

# Markdown 表格（含 avg_score 以便核对）
#  "mse", "mae", "avg_score", , "log_file"
print(f"\n📋 最优参数表格（选择模式: {select_mode}）\n")
print(df[[
    "dataset", "model", "pred_len",
    "seq_len", "d_model", "d_ff", "e_layers", "epochs", "lr"
]].to_markdown(index=False))

print("\n\n📊 模型对比表（逐数据集 & 预测长度）：\n")

datasets = sorted(df['dataset'].unique())
for dataset in datasets:
    subset = df[df["dataset"] == dataset]
    models = sorted(subset["model"].unique())
    pred_lens = sorted(subset["pred_len"].unique())

    print(f"\n### Dataset: {dataset}\n")
    header = ["Pred"] + models
    print(" | ".join(header))
    print("-" * (len(header) * 18))

    # 收集每个模型跨不同 pred_len 的 (mse, mae) 以便结尾求均值
    avg_scores = {model: [] for model in models}

    for pred in pred_lens:
        row = [str(pred)]
        for model in models:
            record = subset[(subset["model"] == model) & (subset["pred_len"] == pred)]
            if not record.empty:
                mse_v = record.iloc[0]["mse"]
                mae_v = record.iloc[0]["mae"]
                row.append(f"{mse_v:.3f}, {mae_v:.3f}")
                avg_scores[model].append((mse_v, mae_v))
            else:
                row.append(" - ")
        print(" | ".join(row))

    # 每个模型在该数据集下所有预测长度的平均 (mse, mae)
    avg_row = ["AVG"]
    for model in models:
        scores = avg_scores[model]
        if scores:
            mse_avg = sum(s[0] for s in scores) / len(scores)
            mae_avg = sum(s[1] for s in scores) / len(scores)
            avg_row.append(f"{mse_avg:.3f}, {mae_avg:.3f}")
        else:
            avg_row.append(" - ")
    print(" | ".join(avg_row))

print(f"\n✅ 已保存 CSV：{out_csv}")
