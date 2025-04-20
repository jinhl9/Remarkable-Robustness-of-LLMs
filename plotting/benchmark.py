import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def aggregate_and_plot_results(root_dir, intervention="swap"):
    model_classes = {
        "openai-community": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        "EleutherAI": [
            "pythia-410m-deduped", "pythia-1.4b-deduped",
            "pythia-2.8b-deduped", "pythia-6.9b-deduped"
        ]
    }
    tasks = ["arc_easy", "lambada_openai", "hellaswag"]

    model_max_depth = {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
        "gpt2-xl": 48,
        "pythia-410m-deduped": 24,
        "pythia-1.4b-deduped": 24,
        "pythia-2.8b-deduped": 32,
        "pythia-6.9b-deduped": 32
    }

    records = []

    for task in tasks:
        for model_class, models in model_classes.items():
            for model in models:
                base_path = Path(
                    root_dir) / task / model_class / model / intervention
                if not base_path.exists():
                    continue
                max_depth = model_max_depth.get(model)
                for layer_dir in base_path.glob("layer_*/__*/results_*.json"):
                    layer_match = re.search(r"layer_(\d+)", str(layer_dir))
                    if not layer_match:
                        continue
                    layer = int(layer_match.group(1))
                    if intervention == "swap":
                        if layer >= int(max_depth - 1):
                            continue  # skip max layer as instructed
                    if intervention == "delete":
                        if layer >= max_depth - 1:
                            continue  # skip max layer as instructed
                    try:
                        with open(layer_dir, "r") as f:
                            data = json.load(f)
                        result = data.get("results", {}).get(task, {})
                        acc = result.get("acc,none")
                        acc_std = result.get("acc_stderr,none")
                        if acc is not None:
                            records.append({
                                "task":
                                task,
                                "model_class":
                                model_class,
                                "model":
                                model,
                                "layer":
                                layer,
                                "normalized_depth":
                                layer / (max_depth - 1),
                                "acc_mean":
                                acc,
                                "acc_std":
                                acc_std
                            })
                            if layer == (max_depth):
                                print('catch')
                    except Exception as e:
                        print(f"Failed to read {layer_dir}: {e}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("No data found. Check paths and directory structure.")
        return

    df = df.sort_values(by=["normalized_depth", "model"])

    sns.set(style="whitegrid")
    g = sns.FacetGrid(df,
                      col="task",
                      hue="model",
                      row="model_class",
                      margin_titles=True,
                      height=4,
                      aspect=1.5)
    g.map(plt.errorbar, "normalized_depth", "acc_mean", "acc_std", fmt='-o')

    g.add_legend()
    g.set_axis_labels("Normalized Depth (Layer / Max Layer)", "Accuracy")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Model Accuracy vs. Normalized Layer-{intervention}\n(per Task, Model Class, Model)"
    )
    plt.show()
    g.savefig(f"plotting/accuracy_vs_layer_{intervention}.png", dpi=300)


def aggregate_and_plot_results_both_intervention(root_dir):
    model_classes = {
        "openai-community": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        "EleutherAI": ["pythia-410m-deduped", "pythia-1.4b-deduped"]
    }
    tasks = ["arc_easy", "lambada_openai", "hellaswag"]

    model_max_depth = {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
        "gpt2-xl": 48,
        "pythia-410m-deduped": 24,
        "pythia-1.4b-deduped": 24
    }

    records = []

    for task in tasks:
        for model_class, models in model_classes.items():
            for model in models:
                for intervention in ['swap', 'delete']:
                    base_path = Path(
                        root_dir) / task / model_class / model / intervention
                    if not base_path.exists():
                        continue
                    max_depth = model_max_depth.get(model)
                    for layer_dir in base_path.glob(
                            "layer_*/__*/results_*.json"):
                        layer_match = re.search(r"layer_(\d+)", str(layer_dir))
                        if not layer_match:
                            continue
                        layer = int(layer_match.group(1))
                        if intervention == "swap":
                            if layer >= int(max_depth - 1):
                                continue  # skip max layer as instructed
                        if intervention == "delete":
                            if layer >= max_depth - 1:
                                continue  # skip max layer as instructed
                        try:
                            with open(layer_dir, "r") as f:
                                data = json.load(f)
                            result = data.get("results", {}).get(task, {})
                            acc = result.get("acc,none")
                            acc_std = result.get("acc_stderr,none")
                            if acc is not None:
                                records.append({
                                    "task":
                                    task,
                                    "intervention":
                                    intervention,
                                    "model_class":
                                    model_class,
                                    "model":
                                    model,
                                    "model_intervention":
                                    f'{model}-{intervention}',
                                    "layer":
                                    layer,
                                    "normalized_depth":
                                    layer / (max_depth - 1),
                                    "acc_mean":
                                    acc,
                                    "acc_std":
                                    acc_std
                                })
                                if layer == (max_depth):
                                    print('catch')
                        except Exception as e:
                            print(f"Failed to read {layer_dir}: {e}")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("No data found. Check paths and directory structure.")
        return

    df = df.sort_values(by=["normalized_depth", "intervention"])

    sns.set(style="whitegrid")

    hue_kws = {
        'color':
        sns.color_palette()[:len(df['model'].unique())] +
        sns.color_palette()[:len(df['model'].unique())],
        "fmt":
        ['-o'] * len(df['model'].unique()) + ['-*'] * len(df['model'].unique())
    }
    g = sns.FacetGrid(df,
                      col="task",
                      hue="model_intervention",
                      row="model_class",
                      margin_titles=True,
                      hue_kws=hue_kws,
                      height=4,
                      aspect=1.5)
    g.map(
        plt.errorbar,
        "normalized_depth",
        "acc_mean",
        "acc_std",
    )

    g.add_legend()
    g.set_axis_labels("Normalized Depth (Layer / Max Layer)", "Accuracy")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Model Accuracy vs. Normalized Layer\n(per Task, Model Class, Model)")
    plt.show()
    g.savefig(f"plotting/accuracy_vs_layer.png", dpi=300)


if __name__ == "__main__":
    root_dir = "results/benchmark_results"
    aggregate_and_plot_results(root_dir, intervention="swap")
    aggregate_and_plot_results(root_dir, intervention="delete")
    aggregate_and_plot_results_both_intervention(root_dir)
