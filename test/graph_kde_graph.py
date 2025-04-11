import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import yaml

def get_user_inputs():
    parser = argparse.ArgumentParser(description="Generate visualizations for sentence lengths.")
    
    parser.add_argument("yaml_file_path", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("generated_file_path", type=str, help="Path to the generated CSV file (model output).")
    parser.add_argument("base_generated_file_path", type=str, help="Path to the base generated CSV file (baseline model output).")
    parser.add_argument("generated_col", type=str, help="Column name for the generated text.")
    parser.add_argument("label_col", type=str, help="Column name for the label text.")
    parser.add_argument("train_legend", type=str, help="Custom legend label for training column.")
    parser.add_argument("label_legend", type=str, help="Custom legend label for label column.")
    parser.add_argument("--base_gen_legend", type=str, default="Base Generated", 
                       help="Custom legend label for base generated column.")

    args = parser.parse_args()

    with open(args.yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    train_file_path = config['train_csv']
    train_col = config['groundtruth_column']

    return (train_file_path, args.generated_file_path, args.base_generated_file_path, 
            train_col, args.generated_col, args.label_col, 
            args.train_legend, args.label_legend, args.base_gen_legend)


def plot_binned_line_graph(train_df, generated_df, base_gen_df, train_col, generated_col, 
                          label_col, train_legend, label_legend, base_gen_legend):
    # Calculate lengths
    train_df["train_len"] = train_df[train_col].astype(str).apply(len)
    generated_df["generated_len"] = generated_df[generated_col].astype(str).apply(len)
    base_gen_df["base_gen_len"] = base_gen_df[generated_col].astype(str).apply(len)
    generated_df["label_len"] = generated_df[label_col].astype(str).apply(len)

    print(f"Avg train_len: {train_df['train_len'].mean()} | gen: {generated_df['generated_len'].mean()} | "
          f"base_gen: {base_gen_df['base_gen_len'].mean()} | label: {generated_df['label_len'].mean()}")

    bin_edges = np.histogram_bin_edges(np.concatenate([
        train_df["train_len"], 
        generated_df["generated_len"], 
        base_gen_df["base_gen_len"],
        generated_df["label_len"]
    ]), bins="auto")

    train_counts, _ = np.histogram(train_df["train_len"], bins=bin_edges)
    generated_counts, _ = np.histogram(generated_df["generated_len"], bins=bin_edges)
    base_gen_counts, _ = np.histogram(base_gen_df["base_gen_len"], bins=bin_edges)
    label_counts, _ = np.histogram(generated_df["label_len"], bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Left y-axis
    ax1.set_xlabel("Character Length", fontsize=12)
    ax1.set_ylabel("Training Sentences", color="blue", fontsize=12)
    ax1.plot(bin_centers, train_counts, marker="o", linestyle="-", color="blue", label=train_legend,
             linewidth=2.0, markersize=6, alpha=0.9)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Generated / Label Sentences", fontsize=12)
    ax2.plot(bin_centers, generated_counts, marker="s", linestyle="--", color="green", label="Generated",
             linewidth=2.0, markersize=5, alpha=0.7)
    ax2.plot(bin_centers, base_gen_counts, marker="D", linestyle=":", color="purple", label=base_gen_legend,
             linewidth=2.0, markersize=5, alpha=0.7)
    # ax2.plot(bin_centers, label_counts, marker="^", linestyle="-.", color="red", label=f"{label_legend} Label",
    #          linewidth=2.0, markersize=5, alpha=0.7)
    ax2.tick_params(axis='y')

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=10)

    plt.title(f"Sentence Length Distribution (Binned)\nTraining: {train_legend} vs Generated vs Base Generated vs Testing: {label_legend}", 
              fontsize=14)
    plt.tight_layout()

    output_path = f"sentence_length_comparison_binned_{train_legend}_{label_legend}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {output_path}")


def plot_density_graph(train_df, generated_df, base_gen_df, train_col, generated_col, 
                      label_col, train_legend, label_legend, base_gen_legend):
    train_df["train_len"] = train_df[train_col].astype(str).apply(len)
    generated_df["generated_len"] = generated_df[generated_col].astype(str).apply(len)
    base_gen_df["base_gen_len"] = base_gen_df[generated_col].astype(str).apply(len)
    generated_df["label_len"] = generated_df[label_col].astype(str).apply(len)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot training data on left axis
    sns.kdeplot(train_df["train_len"], label=f" Training data of {train_legend}", color="blue", linewidth=2.0, ax=ax1)
    ax1.set_xscale("log")
    ax1.set_xlabel("Character Length", fontsize=12)
    ax1.set_ylabel(f"Density ({train_legend})", color="blue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Create right axis for generated and label data
    ax2 = ax1.twinx()
    sns.kdeplot(generated_df["generated_len"], label=f"Model SFT on {train_legend}", color="green", 
                linestyle="--", linewidth=2.0, ax=ax2)
    sns.kdeplot(base_gen_df["base_gen_len"], label=base_gen_legend, color="purple",
                linestyle=":", linewidth=2.0, ax=ax2)
    sns.kdeplot(generated_df["label_len"], label=f"Testing {label_legend} dataset", 
                color="red", linestyle="-.", linewidth=2.0, ax=ax2)
    ax2.set_ylabel("Density (SFT Model/Base Output and Testing Dataset)", fontsize=12)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=10)

    plt.title(f"Sentence Length Distribution (Smoothed KDE) for SFT Model on {train_legend} tested on {label_legend} dataset", 
              fontsize=14)
    plt.tight_layout()

    output_path = f"/work/pi_wenlongzhao_umass_edu/6/chaitali/code/images/sentence_length_density_kde_dual_axis_{train_legend}_{label_legend}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {output_path}")


def main():
    (train_file_path, generated_file_path, base_generated_file_path, 
     train_col, generated_col, label_col, 
     train_legend, label_legend, base_gen_legend) = get_user_inputs()
    
    train_df = pd.read_csv(train_file_path)
    generated_df = pd.read_csv(generated_file_path)
    base_gen_df = pd.read_csv(base_generated_file_path)

    # plot_binned_line_graph(train_df, generated_df, base_gen_df, train_col, generated_col, 
    #                       label_col, train_legend, label_legend, base_gen_legend)
    plot_density_graph(train_df, generated_df, base_gen_df, train_col, generated_col, 
                      label_col, train_legend, label_legend, base_gen_legend)


if __name__ == "__main__":
    main()
    
"""
python graph_kde_graph.py \
"/work/pi_wenlongzhao_umass_edu/6/chaitali/code/test/dataset_configs/goat.yaml" \
"/project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/generated/generated_output_goat_adapter_goat.csv" \
"/project/pi_wenlongzhao_umass_edu/6/outputs/results_3b/generated/generated_output_base_adapter_goat.csv" \
"generated" "answer" "Goat" "Goat" --base_gen_legend "Baseline Model"
"""