import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from matplotlib.colors import LogNorm, PowerNorm


def plot_seaborn_heatmap(file_path, title="Heatmap"):
    result = torch.load(file_path)
    tensor = result['hsic_matrix']
    df = pd.DataFrame(tensor.cpu().numpy())

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="Blues", annot=False, cbar=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.tight_layout()
    plt.show()
    plt.savefig(file_path.replace('.pt', '.png'), dpi=300)


if __name__ == '__main__':
    rootdir = '/ceph/scratch/jlee/llm-robustness/repsim-layers-cka/'
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.pt'):
                model_name = os.path.splitext(file)[0]
                file_path = os.path.join(subdir, file)
                print(file_path)
                plot_seaborn_heatmap(
                    file_path,
                    title=f"CKA {model_name}",
                )
