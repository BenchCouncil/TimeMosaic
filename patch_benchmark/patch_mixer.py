import os 
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

mpl.rcParams['font.family'] = 'STIXGeneral'

# Parameters
base_path = "../dataset/"
p_list = [8, 16, 32, 64]
c_list = list(range(16, 64, 16))
valid_extensions = [".csv", ".txt"]
mixed_dim = 8
mixed_name = "Mixed"
mixed_patch_cfg = {
    mixed_name: [8, 16, 32]
}

def compute_zipf_deviation(freqs):
    sorted_freqs = np.sort(freqs)[::-1]
    ranks = np.arange(1, len(sorted_freqs) + 1).reshape(-1, 1)
    log_ranks = np.log1p(ranks)
    log_freqs = np.log1p(sorted_freqs).reshape(-1, 1)
    model = LinearRegression().fit(log_ranks, log_freqs)
    r2 = model.score(log_ranks, log_freqs)
    return r2

def plot_heatmap_smooth(matrix, title, filename, row_names, c_list):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(matrix, aspect='auto', interpolation='bicubic', cmap='RdYlBu_r')
    ax.set_xticks(np.arange(len(c_list)))
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_xticklabels(c_list, fontsize=10)
    ax.set_yticklabels([f"Patch {p}" for p in row_names], fontsize=10)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Patch Size", fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n(Blue=Worse, Red=Better)", fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"    Heatmap saved: {filename}")

def load_file(filepath):
    ext = os.path.splitext(filepath)[-1]
    if ext not in valid_extensions:
        return None
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.dropna()

def main():
    print(f"🚀 Start loading datasets from: {base_path}")
    all_patches_dict = {p: [] for p in p_list}
    total_files, used_files = 0, 0
    total_timesteps = 0

    for dirpath, _, filenames in os.walk(base_path):
        for fname in filenames:
            if not any(fname.endswith(ext) for ext in valid_extensions):
                continue
            full_path = os.path.join(dirpath, fname)
            total_files += 1
            print(f"  ⏳ Loading file: {full_path}")
            try:
                df = load_file(full_path)
                if df is None or df.shape[1] < 1:
                    print(f"    ⚠️ File is empty or has no valid value columns, skipping.")
                    continue
                used_files += 1
                num_timesteps = df.shape[0] * df.shape[1]
                total_timesteps += num_timesteps

                df = pd.DataFrame(MinMaxScaler().fit_transform(df))
                for p in p_list:
                    patches = []
                    for col in df.columns:
                        values = df[col].values
                        n_patch = len(values) // p
                        if n_patch < 2:
                            continue
                        values = values[:n_patch * p]
                        patch_matrix = values.reshape(n_patch, p)
                        patches.append(patch_matrix)
                    if patches:
                        total = np.concatenate(patches, axis=0)
                        all_patches_dict[p].append(total)
                        print(f"    - Patch size {p} extracted patches: {total.shape[0]}")
            except Exception as e:
                print(f"    ❌ Failed to load: {fname} - {e}")

    print(f"📦 File loading finished. Total files: {total_files}, Successfully processed: {used_files}")
    print(f"🔢 Total time points: {total_timesteps}")

    print(f"\n📊 Constructing mixed patch pool: {mixed_patch_cfg[mixed_name]}")
    mixed_patch_list = []
    for p in mixed_patch_cfg[mixed_name]:
        if not all_patches_dict[p]:
            print(f"  ⚠️ Patch size={p} has no valid patches for mixing, skipping.")
            continue
        global_patches = np.concatenate(all_patches_dict[p], axis=0)
        patches_proj = PCA(n_components=mixed_dim, random_state=0).fit_transform(global_patches)
        mixed_patch_list.append(patches_proj)
        print(f"    - Patch size {p}, PCA reduced: {patches_proj.shape}")
    if mixed_patch_list:
        mixed_patches = np.concatenate(mixed_patch_list, axis=0)
        print(f"  ✅ Mixed Patch total: {mixed_patches.shape[0]}")
    else:
        print(f"  ⚠️ No valid patches for mixed pool.")
        mixed_patches = None

    full_p_list = p_list + [mixed_name]
    zipf_matrix = np.full((len(full_p_list), len(c_list)), np.nan)
    sil_matrix = np.full((len(full_p_list), len(c_list)), np.nan)
    print("📊 Start clustering analysis on the global patch pool including mixer...")

    for i, p in enumerate(full_p_list):
        if p == mixed_name:
            patches = mixed_patches
        else:
            if not all_patches_dict[p]:
                print(f"  ⚠️ Patch size={p} has no valid patches, skipping.")
                continue
            patches = np.concatenate(all_patches_dict[p], axis=0)
        if patches is None or len(patches) == 0:
            print(f"  ⚠️ Patch size={p} has no valid patches, skipping.")
            continue
        print(f"  ✅ Patch size={p}, total global patches={patches.shape[0]}")
        for j, c in enumerate(c_list):
            if c >= patches.shape[0]:
                continue
            try:
                kmeans = KMeans(n_clusters=c, random_state=0, n_init='auto')
                labels = kmeans.fit_predict(patches)
                counts = np.bincount(labels)
                zipf_matrix[i, j] = compute_zipf_deviation(counts)
                if len(np.unique(labels)) > 1:
                    sil_matrix[i, j] = silhouette_score(patches, labels)
                if c % 16 == 0:
                    print(f"      Clustered C={c}, Silhouette={sil_matrix[i, j]:.4f}, Zipf={zipf_matrix[i, j]:.4f}")
            except Exception as e:
                print(f"      ❌ Clustering C={c} failed: {e}")
                continue

    plot_heatmap_smooth(zipf_matrix, "Zipf Deviation Heatmap", "zipf_deviation_with_mixer.pdf", full_p_list, c_list)
    plot_heatmap_smooth(sil_matrix, "Silhouette Score Heatmap", "silhouette_score_with_mixer.pdf", full_p_list, c_list)
    print("🎉 All patches concatenated, clustered, and heatmaps saved. Check the output files in the current directory.")

if __name__ == "__main__":
    main()
