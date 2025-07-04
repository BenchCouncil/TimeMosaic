import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Parameters
base_path = "../dataset"
p_list = [16, 32, 64, 128, 256]
c_list = list(range(16, 512, 16))
valid_extensions = [".csv", ".txt"]

# Zipf Deviation
def compute_zipf_deviation(freqs):
    sorted_freqs = np.sort(freqs)[::-1]
    ranks = np.arange(1, len(sorted_freqs) + 1)
    zipf_expectation = sorted_freqs[0] / ranks
    deviation = np.mean(np.abs(np.log1p(sorted_freqs) - np.log1p(zipf_expectation)))
    return deviation

# Draw heatmap
def plot_heatmap_smooth(matrix, title, filename, p_list, c_list):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(12, 2.5))
    im = ax.imshow(matrix, aspect='auto', interpolation='bicubic', cmap='RdYlBu_r')
    ax.set_xticks(np.arange(len(c_list)))
    ax.set_yticks(np.arange(len(p_list)))
    ax.set_xticklabels(c_list, fontsize=10)
    ax.set_yticklabels([f"Patch {p}" for p in p_list], fontsize=10)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Patch Size", fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\n(Blue=Better, Red=Worse)", fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"    Heatmap saved: {filename}")

# Load a single file
def load_file(filepath):
    ext = os.path.splitext(filepath)[-1]
    if ext not in valid_extensions:
        return None
    df = pd.read_csv(filepath)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    return df.dropna()

# Main process: concatenate global patch pool
def main():
    print(f"🚀 Start loading datasets from: {base_path}")
    all_patches_dict = {p: [] for p in p_list}
    total_files, used_files = 0, 0
    total_timesteps = 0  # NEW: total time points of all datasets & channels

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

                # ====== Count total time points (all datasets & channels) ======
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
                        all_patches_dict[p].append(np.concatenate(patches, axis=0))
                        print(f"    - Patch size {p} extracted patches: {np.concatenate(patches, axis=0).shape[0]}")
            except Exception as e:
                print(f"    ❌ Failed to load: {fname} - {e}")

    print(f"📦 File loading finished. Total files: {total_files}, Successfully processed: {used_files}")
    print(f"🔢 Total time points: {total_timesteps}")
    print("📊 Start clustering analysis on the global patch pool...")

    zipf_matrix = np.full((len(p_list), len(c_list)), np.nan)
    sil_matrix = np.full((len(p_list), len(c_list)), np.nan)
    for i, p in enumerate(p_list):
        if not all_patches_dict[p]:
            print(f"  ⚠️ Patch size={p} has no valid patches, skipping.")
            continue
        global_patches = np.concatenate(all_patches_dict[p], axis=0)
        print(f"  ✅ Patch size={p}, total global patches={global_patches.shape[0]}")
        for j, c in enumerate(c_list):
            if c >= global_patches.shape[0]:
                continue
            try:
                kmeans = KMeans(n_clusters=c, random_state=0, n_init='auto')
                labels = kmeans.fit_predict(global_patches)
                counts = np.bincount(labels)
                zipf_matrix[i, j] = compute_zipf_deviation(counts)
                if len(np.unique(labels)) > 1:
                    sil_matrix[i, j] = silhouette_score(global_patches, labels)
                if c % 64 == 0:
                    print(f"      Clustered C={c}, Silhouette={sil_matrix[i, j]:.4f}, Zipf={zipf_matrix[i, j]:.4f}")
            except Exception as e:
                print(f"      ❌ Clustering C={c} failed: {e}")
                continue

    plot_heatmap_smooth(zipf_matrix, "Global Zipf Deviation Heatmap", "global_zipf_deviation.pdf", p_list, c_list)
    plot_heatmap_smooth(sil_matrix, "Global Silhouette Score Heatmap", "global_silhouette_score.pdf", p_list, c_list)
    print("🎉 All patches concatenated, clustered, and heatmaps saved. Check the output files in the current directory.")

if __name__ == "__main__":
    main()
