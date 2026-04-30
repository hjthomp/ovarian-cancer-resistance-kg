"""
Generates word cloud visualizations from Knowledge Graph releases.
Creates clouds for nodes, relations, and evidence terms, both globally and per theme.
Outputs high-resolution figures in PNG, JPG, and SVG formats with rounded masks.
"""
import json
import logging
import datetime
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS

from src.utils.io import load_json, save_json
from src.utils.constants import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = DATA_DIR / "experiments" / "figures" / "wordclouds"
RELEASES_DIR = DATA_DIR / "releases"

# --- Configuration ---
SEED = 42
DPI = 300
WIDTH = 1600
HEIGHT = 800
MARGIN = 3
MAX_WORDS = 50      # Increased slightly to allow more terms if space permits
MAX_FONT_SIZE = 100  # Reduced significantly to prevent top words from hogging all space
MIN_FONT_SIZE = 10   # Lowered to 10 to ensure small words can fit in the gaps
MIN_FREQ_NODES = 1
MIN_FREQ_EVIDENCE = 2
RELATIVE_SCALING = 0.4  # Slightly flatter scaling so medium words aren't too big either
CORNER_RADIUS = 350

# Font configuration
PROJECT_ROOT = DATA_DIR.parent
FONTS_DIR = PROJECT_ROOT / "fonts"
FONT_PATH_LIST = [
    "GoogleSans-Regular.ttf",
    "GoogleSans-Bold.ttf"
]
FONT_PATH = None


# Theme Color Palette (Can be extended)
# Maps theme keywords to matplotlib colormaps
THEME_COLORMAPS = {
    "default": "#333333",
    "Clinical Outcomes": "#1F77B4",
    "Resistance": "#FF7F0E",
    "Therapeutic Strategies": "#2CA02C",
    "Tumor Features": "#D62728",
}

def get_user_choice(options: List[Any], prompt_text: str) -> Any:
    """Helper to ask user for a choice from a list."""
    print(f"\n{prompt_text}")
    for i, option in enumerate(options, 1):
        if isinstance(option, Path):
            try:
                display = option.relative_to(DATA_DIR)
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

def get_colormap_for_theme(theme_name: str) -> str:
    """Selects a colormap based on the theme name."""
    theme_lower = theme_name.lower()
    for key, cmap in THEME_COLORMAPS.items():
        if key.lower() in theme_lower:
            return cmap
    return THEME_COLORMAPS["default"]

def create_rounded_rectangle_mask(width: int, height: int, radius: int) -> np.ndarray:
    """Creates a boolean mask for a rounded rectangle (0=draw, 255=mask out)."""
    y, x = np.ogrid[:height, :width]
    # Initialize with 0 (draw everywhere)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Mask out the 4 corners (set to 255)
    # Top-left
    mask[(x < radius) & (y < radius) & ((x - radius)**2 + (y - radius)**2 > radius**2)] = 255
    # Top-right
    mask[(x > width - radius - 1) & (y < radius) & ((x - (width - radius - 1))**2 + (y - radius)**2 > radius**2)] = 255
    # Bottom-left
    mask[(x < radius) & (y > height - radius - 1) & ((x - radius)**2 + (y - (height - radius - 1))**2 > radius**2)] = 255
    # Bottom-right
    mask[(x > width - radius - 1) & (y > height - radius - 1) & ((x - (width - radius - 1))**2 + (y - (height - radius - 1))**2 > radius**2)] = 255

    return mask

def generate_wordcloud_from_frequencies(
    freq_dict: Dict[str, int], 
    output_name: str, 
    run_dir: Path, 
    colormap: str = "#333333", 
    min_freq: int = 1, 
    relative_scaling: float = RELATIVE_SCALING
) -> Optional[Dict[str, Any]]:
    """
    Generates and saves a word cloud from a frequency dictionary.
    Saves in PNG, JPG, and SVG formats.
    """
    if not freq_dict:
        logging.warning(f"No data for {output_name}. Skipping.")
        return None

    # Filter by min_freq
    filtered_freq = {k: v for k, v in freq_dict.items() if v >= min_freq}
    
    if not filtered_freq:
        logging.warning(f"Not enough data after filtering (min_freq={min_freq}) for {output_name}. Skipping.")
        return None

    logging.info(f"Generating WordCloud: {output_name} ({len(filtered_freq)} terms)")

    mask = create_rounded_rectangle_mask(WIDTH, HEIGHT, CORNER_RADIUS)

    wc_kwargs = {
        "width": WIDTH,
        "height": HEIGHT,
        "mask": mask,
        "background_color": "white",
        "max_words": MAX_WORDS,
        "random_state": SEED,
        "stopwords": STOPWORDS,
        "prefer_horizontal": 1.0,
        "relative_scaling": relative_scaling,
        "margin": MARGIN,
        "max_font_size": MAX_FONT_SIZE,
        "min_font_size": MIN_FONT_SIZE
    }
    
    if FONT_PATH and Path(FONT_PATH).exists():
        wc_kwargs["font_path"] = str(FONT_PATH)

    if colormap.startswith("#"):
        wc_kwargs["color_func"] = lambda *args, **kwargs: colormap
    else:
        wc_kwargs["colormap"] = colormap

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(filtered_freq)
    
    # Check actual placed words to ensure MAX_WORDS compliance
    placed_count = len(wc.layout_)
    target_count = min(len(filtered_freq), MAX_WORDS)
    
    if placed_count < target_count:
        logging.warning(f"  [Warning] Only {placed_count}/{target_count} words placed. Try reducing MIN_FONT_SIZE or increasing WIDTH/HEIGHT.")
    else:
        logging.info(f"  [OK] Placed {placed_count}/{target_count} words.")

    # Create figure
    plt.figure(figsize=(10, 5), dpi=DPI)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    
    # Save in multiple formats
    formats = ["png", "jpg", "svg"]
    for fmt in formats:
        fmt_dir = run_dir / fmt
        fmt_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = fmt_dir / f"{output_name}.{fmt}"
        plt.savefig(save_path, format=fmt, dpi=DPI, bbox_inches='tight')
        
    plt.close()
    
    return {
        "figure_name": output_name,
        "words_placed": placed_count,
        "words_target": target_count,
        "is_complete": placed_count >= target_count
    }

def process_evidence_text(evidence_list: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """Extracts individual words from evidence sentences, filtering stopwords."""
    words = []
    for ev in evidence_list:
        # Handle both string and dict evidence formats
        text = ev.get("evidence_text", "") if isinstance(ev, dict) else str(ev)
        
        # Simple tokenization
        tokens = re.findall(r'\w+', text.lower())
        
        # Filter stopwords and short words
        # Keep if length > 2 OR if it's a special character (non-ascii, e.g., beta)
        filtered = [w for w in tokens if w not in STOPWORDS and (len(w) > 2 or not w.isascii())]
        words.extend(filtered)
    return words

def run_wordclouds() -> None:
    """Main execution flow for generating word cloud visualizations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- KG Figure Generator (WordClouds) ---")
    
    # 1. Select KG
    json_files = sorted(list(RELEASES_DIR.glob("*.json")))
    if not json_files:
        logging.error(f"No KG files found in {RELEASES_DIR}")
        return

    kg_file = get_user_choice(json_files, "Select Knowledge Graph file:")
    logging.info(f"Loading {kg_file.name}...")
    kg_data = load_json(kg_file)
    graph_data = kg_data.get("graph", kg_data)
    
    # Normalize graph data structure
    if isinstance(graph_data, list):
        graph_data = {"default": graph_data}
    elif not isinstance(graph_data, dict):
        logging.error("Invalid graph data format.")
        return

    # 1.2 Select Font
    global FONT_PATH
    available_fonts = [FONTS_DIR / f for f in FONT_PATH_LIST if (FONTS_DIR / f).exists()]
    if available_fonts:
        font_options = available_fonts + ["Default (System/Library)"]
        selected_font = get_user_choice(font_options, "Select Font:")
        if isinstance(selected_font, Path):
            FONT_PATH = selected_font
        else:
            FONT_PATH = None
    else:
        logging.info("No custom fonts found in /fonts. Using default.")

    # 1.5 Select Target
    target_options = ["All", "Nodes", "Relations", "Evidence"]
    selected_target = get_user_choice(target_options, "Select what to generate:")

    # 2. Prepare Output Directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Aggregate Data
    global_nodes = Counter()
    global_relations = Counter()
    global_evidence_words = Counter()
    
    # Per cluster containers
    cluster_data = defaultdict(lambda: {
        "nodes": Counter(),
        "relations": Counter(),
        "evidence_words": Counter()
    })
    
    logging.info("Processing graph data...")
    for theme, triplets in graph_data.items():
        for t in triplets:
            s = t.get("subject")
            o = t.get("object")
            r = t.get("relation")
            ev_list = t.get("evidence_list", [])
            
            if s and o and r:
                # Nodes
                if selected_target in ["All", "Nodes"]:
                    global_nodes[s] += 1
                    global_nodes[o] += 1
                    cluster_data[theme]["nodes"][s] += 1
                    cluster_data[theme]["nodes"][o] += 1
                
                # Relations
                if selected_target in ["All", "Relations"]:
                    global_relations[r] += 1
                    cluster_data[theme]["relations"][r] += 1
                
                # Evidence
                if selected_target in ["All", "Evidence"]:
                    ev_words = process_evidence_text(ev_list)
                    global_evidence_words.update(ev_words)
                    cluster_data[theme]["evidence_words"].update(ev_words)

    # 4. Generate Figures
    generation_stats = []
    
    # A. Global WordClouds
    if selected_target in ["All", "Nodes"]:
        res = generate_wordcloud_from_frequencies(global_nodes, "global_nodes", run_dir, colormap="#333333", min_freq=MIN_FREQ_NODES)
        if res: generation_stats.append(res)
    if selected_target in ["All", "Relations"]:
        res = generate_wordcloud_from_frequencies(global_relations, "global_relations", run_dir, colormap="#333333", min_freq=MIN_FREQ_NODES)
        if res: generation_stats.append(res)
    if selected_target in ["All", "Evidence"]:
        res = generate_wordcloud_from_frequencies(global_evidence_words, "global_evidence", run_dir, colormap="#333333", min_freq=MIN_FREQ_EVIDENCE)
        if res: generation_stats.append(res)
    
    # B. Per Cluster WordClouds
    for theme, data in cluster_data.items():
        safe_theme = re.sub(r'[^a-zA-Z0-9]', '_', theme)
        cmap = get_colormap_for_theme(theme)
        
        if selected_target in ["All", "Nodes"]:
            res = generate_wordcloud_from_frequencies(
                data["nodes"], 
                f"cluster_{safe_theme}_nodes", 
                run_dir, 
                colormap=cmap, 
                min_freq=MIN_FREQ_NODES
            )
            if res: generation_stats.append(res)
        
        if selected_target in ["All", "Relations"]:
            res = generate_wordcloud_from_frequencies(
                data["relations"], 
                f"cluster_{safe_theme}_relations", 
                run_dir, 
                colormap=cmap, 
                min_freq=MIN_FREQ_NODES
            )
            if res: generation_stats.append(res)
        
        if selected_target in ["All", "Evidence"]:
            res = generate_wordcloud_from_frequencies(
                data["evidence_words"], 
                f"cluster_{safe_theme}_evidence", 
                run_dir, 
                colormap=cmap, 
                min_freq=MIN_FREQ_EVIDENCE
            )
            if res: generation_stats.append(res)

    # 5. Save Metadata
    metadata = {
        "run_timestamp": timestamp,
        "input_kg_file": str(kg_file.relative_to(PROJECT_ROOT)),
        "generation_target": selected_target,
        "wordcloud_parameters": {
            "seed": SEED,
            "dpi": DPI,
            "width": WIDTH,
            "height": HEIGHT,
            "margin": MARGIN,
            "max_words": MAX_WORDS,
            "max_font_size": MAX_FONT_SIZE,
            "min_font_size": MIN_FONT_SIZE,
            "min_freq_nodes": MIN_FREQ_NODES,
            "min_freq_evidence": MIN_FREQ_EVIDENCE,
            "relative_scaling": RELATIVE_SCALING,
            "mask_corner_radius": CORNER_RADIUS,
            "font_path": str(FONT_PATH.name) if isinstance(FONT_PATH, Path) else "Default",
        },
        "theme_colors": THEME_COLORMAPS,
        "input_kg_metadata": kg_data.get("metadata", "Not available"),
        "generation_stats": generation_stats
    }
    metadata_path = run_dir / "metadata.json"
    save_json(metadata, metadata_path)
    logging.info(f"Metadata saved to {metadata_path}")

    logging.info(f"All figures saved to {run_dir}")

if __name__ == "__main__":
    run_wordclouds()
