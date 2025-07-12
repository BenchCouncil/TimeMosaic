import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap, LogNorm



# Set font for scientific paper figures
# mpl.rcParams['font.family'] = 'DejaVu Sans Mono'
mpl.rcParams['font.family'] = 'STIXGeneral'
# mpl.rcParams['font.family'] = 'Arial'  # or 'serif', 'Helvetica', 'CMU Serif'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['pdf.fonttype'] = 42  # Embed fonts in PDF for LaTeX compatibility

# Parameters
base_path = "../dataset"
patch_size = 16
n_clusters = 100
top_k = 20
valid_extensions = [".csv", ".txt"]

def load_file(filepath):
    ext = os.path.splitext(filepath)[-1]
    if ext not in valid_extensions:
        return None
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.dropna()

def extract_patches(base_path, patch_size):
    all_patches = []
    for dirpath, _, filenames in os.walk(base_path):
        for fname in filenames:
            if not any(fname.endswith(ext) for ext in valid_extensions):
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                df = load_file(full_path)
                if df is None or df.shape[1] < 1:
                    continue
                df = pd.DataFrame(MinMaxScaler().fit_transform(df))
                for col in df.columns:
                    values = df[col].values
                    n_patch = len(values) // patch_size
                    if n_patch < 2:
                        continue
                    values = values[:n_patch * patch_size]
                    patch_matrix = values.reshape(n_patch, patch_size)
                    all_patches.append(patch_matrix)
            except Exception as e:
                print(f"Failed on {fname}: {e}")
                continue
    return np.concatenate(all_patches, axis=0) if all_patches else None

def plot_transition_matrix_variants(tokens, K, top_k=20, prefix="transition_matrix"):
    C = np.zeros((K, K), dtype=int)
    for i in range(len(tokens) - 1):
        C[tokens[i], tokens[i + 1]] += 1
    P = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    P[np.isnan(P)] = 0

    total_freq = C.sum(axis=1)
    top_states = np.argsort(total_freq)[-top_k:][::-1]
    P_top = P[np.ix_(top_states, top_states)]

    xticks = np.arange(1, top_k + 1)
    yticks = np.arange(1, top_k + 1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        P_top,
        cmap="YlOrRd", # rocket  YlOrRd crest
        vmin=0, vmax=1.0,
        xticklabels=np.arange(1, top_k + 1),
        yticklabels=np.arange(1, top_k + 1),
        square=True,
        # cbar_kws={"label": "Transition Probability"}
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Next State", fontsize=24)
    plt.ylabel("Current State", fontsize=24)
    plt.title("State Transition Probability Matrix\n(Top 20 Active States)", fontsize=20, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{prefix}.pdf")
    plt.close()


def main():
    print(f"🚀 Loading patches from: {base_path}")
    patches = extract_patches(base_path, patch_size)
    if patches is None or len(patches) < n_clusters + 1:
        print("❌ Not enough patches to cluster.")
        return

    print(f"✅ Total patches: {len(patches)}, patch size: {patch_size}")
    print(f"🤖 Clustering with KMeans (K={n_clusters}) ...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(patches)

    print("📊 Generating transition matrix visualizations ...")
    plot_transition_matrix_variants(
        labels,
        K=n_clusters,
        top_k=top_k,
        prefix=f"transition_matrix_p{patch_size}_k{n_clusters}"
    )

main()
