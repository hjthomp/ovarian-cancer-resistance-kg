"""
Generates visualizations for NER extraction results.
Creates treemaps for category distribution and faceted bar charts for top entities.
Supports custom font families and different scaling methods for treemaps.
"""

import csv
import logging
import datetime
import re
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union, Set

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.font_manager import FontProperties
import numpy as np
import squarify

from src.utils.io import save_json
from src.utils.constants import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_DIR = DATA_DIR / "experiments" / "ner"
OUTPUT_DIR = DATA_DIR / "experiments" / "figures" / "ner"

# Font configuration
PROJECT_ROOT = DATA_DIR.parent
FONTS_DIR = PROJECT_ROOT / "fonts"
FONT_PROPS_REGULAR = None
FONT_PROPS_BOLD = None

# Configuration
DPI = 300
FIG_SIZE_TREEMAP = (12, 8)
ALPHA = 0.85
TEXT_COLOR = "white"
FONT_SIZE_LIST = 8
FONT_SIZE_CATEGORY = 14
FONT_SIZE_TITLE = 18
DEFAULT_TOP_N = 20

# --- Global Color Configuration ---
# A default palette of colors. Categories will be assigned colors from this list in order.
# If there are more categories than colors, the rest will be assigned automatically from a colormap.

DEFAULT_COLOR_PALETTE = [
    "#4E79A7",  # Azul
    "#F28E2B",  # Naranja
    "#E15759",  # Rojo
    "#59A14F",  # Verde
    "#B07AA1",  # Morado
    "#FF7F7F",  # Rosa intenso
    "#D4A017",  # Amarillo oscuro / dorado
    "#A6761D",  # Marrón oscuro
]

def discover_font_families(fonts_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Scans the fonts directory and groups font files by family name."""
    families = defaultdict(dict)
    if not fonts_dir.exists():
        return {}
    
    pattern = re.compile(r"^(.*?)-(Regular|Bold|Italic|BoldItalic)\.ttf$", re.IGNORECASE)
    
    for font_file in fonts_dir.glob("*.ttf"):
        match = pattern.match(font_file.name)
        if match:
            family_name = match.group(1)
            style = match.group(2).title()
            families[family_name][style] = font_file
            
    return dict(families)

def get_user_choice(options: List[Any], prompt_text: str) -> Any:
    """Helper to ask user for a choice from a list."""
    print(f"\n{prompt_text}")
    for i, option in enumerate(options, 1):
        if isinstance(option, Path):
            try:
                display = option.relative_to(INPUT_DIR)
            except ValueError:
                display = option.name
        else:
            display = str(option)

        print(f"[{i}] {display}")
    
    while True:
        try:
            choice = int(input(f"Select an option (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def load_entities(csv_path: Path) -> Tuple[Dict[str, List[Tuple[str, int]]], int]:
    """Loads entities from CSV into a dictionary grouped by category."""
    data = defaultdict(list)
    total_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            cat = row['Category']
            entity = row['Entity']
            freq = int(row['Frequency'])
            data[cat].append((entity, freq))
            total_count += freq
            
    return data, total_count

def generate_category_colors(categories: Union[List[str], Set[str]]) -> Dict[str, Any]:
    """Generates a consistent color palette for the categories."""
    # Use Tab20 for distinct, professional colors for fallback
    cmap = plt.get_cmap('tab20')
    colors = {}
    
    # Sort categories to ensure deterministic color assignment
    sorted_cats = sorted(list(categories))
    
    palette_len = len(DEFAULT_COLOR_PALETTE)
    
    for i, cat in enumerate(sorted_cats):
        if i < palette_len:
            # Assign from the default palette first
            colors[cat] = DEFAULT_COLOR_PALETTE[i]
        else:
            # Assign from the colormap for additional categories
            colors[cat] = cmap((i - palette_len) % 20)
    
    return colors

def generate_category_treemap(
    category_totals: Dict[str, int], 
    palette: Dict[str, Any], 
    scale_method: str, 
    run_dir: Path, 
    input_file: Path, 
    font_props_regular: Optional[FontProperties] = None, 
    font_props_bold: Optional[FontProperties] = None
) -> None:
    """Generates and saves a treemap of NER categories."""
    logging.info(f"Generating category treemap with '{scale_method}' scaling...")
    
    sorted_categories = sorted(category_totals.keys(), key=lambda k: category_totals[k], reverse=True)
    
    # Use original sizes for labels, but scaled sizes for plotting
    original_sizes = [category_totals[cat] for cat in sorted_categories]
    
    if scale_method == 'log':
        # Use log1p for safe log scaling (handles 0)
        plot_sizes = [np.log1p(s) for s in original_sizes]
    elif scale_method == 'sqrt':
        # Use square root for a less aggressive, intermediate scaling
        plot_sizes = [np.sqrt(s) for s in original_sizes]
    else: # linear
        plot_sizes = original_sizes

    colors = [palette[cat] for cat in sorted_categories]
    labels = [f"{cat}\n({size:,})" for cat, size in zip(sorted_categories, original_sizes)]

    fig, ax = plt.subplots(figsize=FIG_SIZE_TREEMAP, dpi=DPI)
    ax.set_axis_off()

    # Use squarify to get coordinates, then draw manually for more control
    norm_sizes = squarify.normalize_sizes(plot_sizes, 100, 100)
    rects = squarify.squarify(norm_sizes, 0, 0, 100, 100)

    for i, rect in enumerate(rects):
        x, y, dx, dy = rect['x'], rect['y'], rect['dx'], rect['dy']
        
        # Draw the rectangle with a border
        ax.add_patch(
            mpatches.Rectangle(
                (x, y), dx, dy,
                facecolor=colors[i],
                alpha=ALPHA,
                edgecolor="white",
                linewidth=2.5
            )
        )
        
        # Add the label if the box is big enough
        if dx > 5 and dy > 5:
             ax.text(x + dx/2, y + dy/2, labels[i],
                    ha="center", va="center",
                    color=TEXT_COLOR, fontsize=12, fontproperties=font_props_bold)

    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0) # Invert Y-axis so largest categories appear at the top
    ax.set_title(f"Distribution of NER Categories by Unique Entities", fontsize=FONT_SIZE_TITLE + 4, fontproperties=font_props_bold)
    
    # Save
    output_filename = f"treemap_categories_{scale_method}.png"
    save_path = run_dir / output_filename
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Category treemap saved to {save_path}")

def generate_entity_barcharts(
    data: Dict[str, List[Tuple[str, int]]], 
    palette: Dict[str, Any], 
    top_n: int, 
    run_dir: Path, 
    input_file: Path, 
    font_props_regular: Optional[FontProperties] = None, 
    font_props_bold: Optional[FontProperties] = None
) -> None:
    """Generates and saves faceted bar charts for top N entities per category."""
    logging.info(f"Generating Top {top_n} entity bar charts...")

    # Sort by number of unique entities to match the Treemap logic (Biggest/Thickest first)
    sorted_categories = sorted(data.keys(), key=lambda k: len(data[k]), reverse=True)
    
    # Determine grid size (e.g., 2 columns)
    n_cats = len(sorted_categories)
    n_cols = 3 # More compact layout
    n_rows = math.ceil(n_cats / n_cols)

    # Adjusted figsize: Increased height per row (3.5 -> 5) to make bars wider and cleaner
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), dpi=DPI, squeeze=False)
    fig.suptitle(f"Top {top_n} Entities per Category", fontsize=FONT_SIZE_TITLE + 4, fontproperties=font_props_bold)

    for i, cat in enumerate(sorted_categories):
        row_idx, col_idx = divmod(i, n_cols)
        ax = axes[row_idx, col_idx]

        # Get top N items for the category
        items = sorted(data[cat], key=lambda x: x[1], reverse=True)[:top_n]
        if not items:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cat, color=palette[cat], fontproperties=font_props_bold)
            ax.set_axis_off()
            continue

        entities = [item[0] for item in items]
        frequencies = [item[1] for item in items]

        # Horizontal bar chart with borders and thinner bars
        # Increased height to 0.8 for thicker bars
        bars = ax.barh(entities, frequencies, color=palette[cat], alpha=ALPHA, height=0.8)
        ax.set_title(cat, color=palette[cat], fontproperties=font_props_bold)
        ax.invert_yaxis()  # Show most frequent on top
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=8)
        ax.set_xlabel("Frequency", fontsize=9, fontproperties=font_props_regular)

        # Add frequency labels to bars
        for bar in bars:
            width = bar.get_width()
            # Increased spacing (0.01 -> 0.02) and font size (8 -> 9)
            ax.text(width + (max(frequencies) * 0.02), bar.get_y() + bar.get_height()/2,
                    f'{width:,}', va='center', ha='left', fontsize=9, fontproperties=font_props_regular)

        # Adjust x-axis limit to make space for labels
        ax.set_xlim(right=max(frequencies) * 1.15)

        if font_props_regular:
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_props_regular)
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_props_regular)

    # Hide unused subplots
    for i in range(n_cats, n_rows * n_cols):
        row_idx, col_idx = divmod(i, n_cols)
        axes[row_idx, col_idx].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # Save
    output_filename = f"barcharts_top_{top_n}_entities.png"
    save_path = run_dir / output_filename
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Entity bar charts saved to {save_path}")


def run_analysis_generator() -> None:
    """Main execution flow for generating visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- NER Distribution Analyzer ---")

    # 1. Select Input File
    # Modified to find both original entities.csv and entities_normalized.csv
    csv_files = sorted(list(INPUT_DIR.rglob("entities*.csv")))
    if not csv_files:
        logging.error(f"No entities.csv found in {INPUT_DIR}")
        return

    input_file = get_user_choice(csv_files, "Select input entities file:")
    
    # 2. Load Data
    data, total_volume = load_entities(input_file)
    logging.info(f"Loaded {sum(len(v) for v in data.values())} entities across {len(data)} categories.")

    # --- New: Select Font Family ---
    global FONT_PROPS_REGULAR, FONT_PROPS_BOLD
    font_families = discover_font_families(FONTS_DIR)
    selected_family_name = "Default (System/Library)"
    
    if font_families:
        font_options = sorted(list(font_families.keys())) + ["Default (System/Library)"]
        selected_family_name = get_user_choice(font_options, "Select Font Family:")

        if selected_family_name != "Default (System/Library)":
            family_files = font_families[selected_family_name]
            
            regular_path = family_files.get("Regular")
            if regular_path and regular_path.exists():
                FONT_PROPS_REGULAR = FontProperties(fname=regular_path)
                logging.info(f"Using regular font: {regular_path.name}")
            else:
                logging.warning(f"Regular font not found for family '{selected_family_name}'. Using default.")
            
            bold_path = family_files.get("Bold")
            if bold_path and bold_path.exists():
                FONT_PROPS_BOLD = FontProperties(fname=bold_path)
                logging.info(f"Using bold font: {bold_path.name}")
            else:
                FONT_PROPS_BOLD = FONT_PROPS_REGULAR
                logging.warning(f"Bold font not found for family '{selected_family_name}'. Using regular font for bold text.")
        else:
            logging.info("Using default system font.")
    else:
        logging.info("No custom fonts found in /fonts. Using default.")

    # 3. User Settings
    try:
        top_n_input = input(f"Enter Top-N entities for bar charts (default {DEFAULT_TOP_N}): ").strip()
        top_n = int(top_n_input) if top_n_input else DEFAULT_TOP_N
    except ValueError:
        top_n = DEFAULT_TOP_N
    
    # Changed default to linear, second option sqrt (accepts 'sqr'), third log
    scale_method = input("Enter treemap scaling (linear/sqrt/log) [default: sqrt]: ").strip().lower() or "sqrt"

    if scale_method not in ["linear", "log", "sqrt"]:
        scale_method = "sqrt"

    # 4. Prepare Data & Palette
    # Modified per user request: Count unique entities instead of summing frequencies for the Treemap
    category_totals = {cat: len(items) for cat, items in data.items()}
    category_totals = {k: v for k, v in category_totals.items() if v > 0} # Filter empty
    
    if not category_totals:
        logging.error("No data to plot after filtering empty categories.")
        return

    palette = generate_category_colors(category_totals.keys())

    # 5. Create output directory and generate figures
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate Figure 1: Category Treemap ---
    generate_category_treemap(category_totals, palette, scale_method, run_dir, input_file, font_props_regular=FONT_PROPS_REGULAR, font_props_bold=FONT_PROPS_BOLD)

    # --- Generate Figure 2: Faceted Bar Charts ---
    generate_entity_barcharts(data, palette, top_n, run_dir, input_file, font_props_regular=FONT_PROPS_REGULAR, font_props_bold=FONT_PROPS_BOLD)

    # 6. Save Metadata
    metadata = {
        "run_timestamp": timestamp,
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "parameters": {
            "top_n_entities": top_n,
            "treemap_scaling": scale_method,
            "font_family": selected_family_name
        },
        "summary": {
            "total_categories_plotted": len(category_totals),
            "total_unique_entities": sum(category_totals.values())
        }
    }
    save_json(metadata, run_dir / "metadata.json")

    logging.info(f"\nAnalysis complete. Figures saved in {run_dir}")

if __name__ == "__main__":
    run_analysis_generator()