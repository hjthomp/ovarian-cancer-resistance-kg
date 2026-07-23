import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.utils.constants import DATA_DIR

INPUT_DIR = DATA_DIR / "experiments" / "ablation" / "results.csv"
OUTPUT_DIR = DATA_DIR / "experiments" / "figures" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# STYLE
plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
})

COLORS = {
    "cohesion": "#1f77b4",
    "variance": "#ff7f0e",
    "smoothness": "#2ca02c",
    "jump": "#d62728",
}

FIG1_YLABEL = "Batch-level metrics"
FIG2_YLABEL = "Adjacent-node metrics"

# DATA PREP
def prepare_df(df):
    df = df.copy()

    # Define a consistent order for baselines that might appear
    baseline_order = ["default"]
    
    # Separate baselines and agglomerative results
    # Filter only for baselines that are actually present in the results
    existing_baselines = [b for b in baseline_order if b in df['setting'].unique()]
    df_baselines = df[df['setting'].isin(existing_baselines)].copy()
    df_semantic = df[df['setting'] == 'agglomerative'].copy()

    # Sort baselines according to the defined order
    df_baselines['setting'] = pd.Categorical(df_baselines['setting'], categories=existing_baselines, ordered=True)
    df_baselines = df_baselines.sort_values('setting')

    # Sort semantic results by threshold
    df_semantic = df_semantic.sort_values('threshold')
    
    # Recombine and reset index to ensure correct plotting order
    df_sorted = pd.concat([df_baselines, df_semantic], ignore_index=True)

    # --- Create new labels based on the *sorted* dataframe ---
    new_settings = []
    for i, row in df_sorted.iterrows():
        s = row['setting']
        if s == "random":
            new_settings.append("Random\nBaseline")
        elif s == "default":
            new_settings.append("Thematic\n(Baseline)")
        elif s == "agglomerative":
            new_settings.append(row['threshold'])
        else:
            new_settings.append(s) # Fallback

    df_sorted["setting"] = new_settings
    
    return df_sorted

# UTILS
def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()

def safe(x):
    x = np.array(x, dtype=float)
    return np.nan_to_num(x, nan=0.0)

def plot_with_std(ax, x, mean, std, color, label):
    mean = safe(mean)
    std = safe(std)

    ax.plot(x, mean, marker="o", color=color, label=label)

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=0.12
    )

def format_axis(ax, ylabel):
    ax.set_ylabel(ylabel)

    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), linespacing=1.5) # rotation=45

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="best")


# BRACKET BELOW X-AXIS
def add_group_bracket_below(ax, x_start, x_end, text):
    # Do not draw if the range is invalid
    if x_start >= x_end:
        return
        
    y = -0.1  # position below axis

    # horizontal line (main bracket)
    ax.plot(
        [x_start, x_end],
        [y, y],
        transform=ax.get_xaxis_transform(), # Use axis-relative coordinates for y
        color="black",
        linewidth=1,
        clip_on=False
    )

    # left tick
    ax.plot(
        [x_start, x_start],
        [y, y + 0.04],
        transform=ax.get_xaxis_transform(), 
        color="black",
        linewidth=1,
        clip_on=False
    )

    # right tick
    ax.plot(
        [x_end, x_end],
        [y, y + 0.04],
        transform=ax.get_xaxis_transform(),
        color="black",
        linewidth=1,
        clip_on=False
    )

    # text centered below
    ax.text(
        (x_start + x_end) / 2,
        y - 0.025,
        text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9
    )

def plot_batch_metrics_on_ax(ax, df):
    """Helper to plot batch metrics on a given Axes object."""
    x = np.arange(len(df))
    labels = df["setting"].tolist()

    plot_with_std(ax, x, df["avg_cohesion"], df["std_cohesion"], COLORS["cohesion"], "Cohesion")
    plot_with_std(ax, x, df["avg_variance"], df["std_variance"], COLORS["variance"], "Variance")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    format_axis(ax, FIG1_YLABEL)

    # Dynamically find where agglomerative clustering starts
    first_agg_idx = df[df['threshold'] != "none"].index.min()
    if first_agg_idx is not None and not pd.isna(first_agg_idx):
        add_group_bracket_below(ax, first_agg_idx, len(x) - 1, "Agglomerative (Thresholds)")

def plot_adjacent_metrics_on_ax(ax, df):
    """Helper to plot adjacent metrics on a given Axes object."""
    x = np.arange(len(df))
    labels = df["setting"].tolist()

    plot_with_std(ax, x, df["avg_smoothness"], df["std_smoothness"], COLORS["smoothness"], "Cosine similarity")
    plot_with_std(ax, x, df["avg_jump_distance"], df["std_jump_distance"], COLORS["jump"], "L2 distance")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    format_axis(ax, FIG2_YLABEL)

    # Dynamically find where agglomerative clustering starts
    first_agg_idx = df[df['threshold'] != "none"].index.min()
    if first_agg_idx is not None and not pd.isna(first_agg_idx):
        add_group_bracket_below(ax, first_agg_idx, len(x) - 1, "Agglomerative (Thresholds)")

# FIGURE 1
def plot_batch_metrics(df):
    fig, ax = plt.subplots()
    plot_batch_metrics_on_ax(ax, df)
    save_fig("figure_1_batch.png")

# FIGURE 2
def plot_adjacent_metrics(df):
    fig, ax = plt.subplots()
    plot_adjacent_metrics_on_ax(ax, df)
    save_fig("figure_2_adjacent.png")

# FIGURE 3
def plot_multipanel(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_batch_metrics_on_ax(axes[0], df)
    plot_adjacent_metrics_on_ax(axes[1], df)
    plt.tight_layout()
    save_fig("figure_3_multipanel.png")

# MAIN
def run_plot_ablation() -> None:
    df = pd.read_csv(INPUT_DIR)
    df = prepare_df(df)

    plot_batch_metrics(df)
    plot_adjacent_metrics(df)
    plot_multipanel(df)

    print("\nFigures saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    run_plot_ablation()