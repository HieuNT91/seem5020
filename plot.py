import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Generate performance plots from benchmark data.")
    parser.add_argument('--csv', type=str, default='benchmark_results.csv', help='Path to the CSV file')
    parser.add_argument('--dataset', type=str, required=True, help='Target dataset to plot (e.g., Skewed, Balanced)')
    args = parser.parse_args()

    # Load and filter data
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: {args.csv} not found.")
        return

    
    # df_filtered = df[df['Algorithm'] != "Space-Saving"]
    df_filtered = df[df['Algorithm'] != "MisraGriesAlphaQuarantine"]
    df_filtered = df_filtered[df_filtered['Dataset'] == args.dataset]
    if df_filtered.empty:
        print(f"Error: Dataset '{args.dataset}' not found. Available options: {df['Dataset'].unique()}")
        return

    sns.set_theme(style="whitegrid")

    # Plot 1: Stream Length vs Metrics
    # Fix Alpha to isolate the effect of Stream Length
    fixed_alpha = df_filtered['Alpha'].min()
    df_len = df_filtered[df_filtered['Alpha'] == fixed_alpha]

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle(f"Performance vs. Stream Length ({args.dataset} Dataset, Alpha={fixed_alpha})", fontsize=14, fontweight='bold')

    sns.lineplot(data=df_len, x='Stream Length', y='Mean Relative Error', hue='Algorithm', marker='o', ax=axes1[0])
    axes1[0].set_title("Relative Error vs. Stream Length")
    axes1[0].set_ylim(bottom=0)
    axes1[0].set_ylabel("Mean Relative Error (Log Scale)")
    axes1[0].set_yscale('symlog', linthresh=0.01) # symlog handles 0.0 values without crashing

    sns.lineplot(data=df_len, x='Stream Length', y='Space (Bytes)', hue='Algorithm', marker='s', ax=axes1[1])
    axes1[1].set_title("Space Consumption vs. Stream Length")
    axes1[1].set_ylabel("Space (Bytes)")
    axes1[1].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{args.dataset}_performance_vs_stream_length.png")

    # Plot 2: Alpha vs Metrics
    # Fix Stream Length to isolate the effect of Alpha
    fixed_len = df_filtered['Stream Length'].max()
    df_alpha = df_filtered[df_filtered['Stream Length'] == fixed_len]

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f"Performance vs. Alpha ({args.dataset} Dataset, Stream Length={fixed_len})", fontsize=14, fontweight='bold')

    sns.lineplot(data=df_alpha, x='Alpha', y='Mean Relative Error', hue='Algorithm', marker='o', ax=axes2[0])
    axes2[0].set_title("Relative Error vs. Alpha")
    axes2[0].set_ylabel("Mean Relative Error (Log Scale)")
    axes2[0].set_yscale('symlog', linthresh=0.01)

    sns.lineplot(data=df_alpha, x='Alpha', y='Space (Bytes)', hue='Algorithm', marker='s', ax=axes2[1])
    axes2[1].set_title("Space Consumption vs. Alpha")
    axes2[1].set_ylabel("Space (Bytes)")

    plt.tight_layout()
    plt.savefig(f"{args.dataset}_performance_vs_alpha.png")

if __name__ == "__main__":
    main()