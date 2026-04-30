import shutil
import logging
from pathlib import Path
from datetime import datetime
from ..utils.constants import RELEASES_DIR, ARTICLE_BUILDER_DIR, ABSTRACT_AUGMENTER_DIR, SUFFIX_FINAL_KG


def get_available_graphs():
    """Scans for candidate graphs in builder and augmenter directories."""
    candidates = []
    
    # 1. Scan Article Builder Unified Graphs
    # Path: data/runs/article_builder/<TIMESTAMP>/unified_graphs/kg_*.json
    if ARTICLE_BUILDER_DIR.exists():
        for path in ARTICLE_BUILDER_DIR.glob("**/unified_graphs/kg_*.json"):
            candidates.append({
                "path": path,
                "source": "Article Builder (Unified)",
                "date": path.parent.parent.name, # Parent is unified_graphs, Grandparent is timestamp
                "type": "unified"
            })

    # 2. Scan Abstract Augmenter Graphs
    # Path: data/runs/abstract_augmenter/TIMESTAMP/kg_*.json
    if ABSTRACT_AUGMENTER_DIR.exists():
        for path in ABSTRACT_AUGMENTER_DIR.glob("**/kg_*.json"):
             candidates.append({
                "path": path,
                "source": "Abstract Augmenter",
                "date": path.parent.name,
                "type": "augmented"
            })
            
    # Sort by date (descending)
    candidates.sort(key=lambda x: x["date"], reverse=True)
    return candidates

def publish_graph(graph_path: Path, target_filename: str = None):
    """Copies the graph to data/releases with versioning."""
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    
    if target_filename:
        if not target_filename.endswith('.json'):
            target_filename += ".json"
    else:
        # Determine next version number automatically
        pattern = f"{SUFFIX_FINAL_KG}_v*.json"
        existing_versions = list(RELEASES_DIR.glob(pattern))
        next_version = 1
        
        if existing_versions:
            versions = []
            for v in existing_versions:
                try:
                    # Extract number from suffix_final_kg_v1.json
                    parts = v.stem.rsplit("_v", 1)
                    if len(parts) == 2 and parts[0] == SUFFIX_FINAL_KG:
                        versions.append(int(parts[1]))
                except ValueError:
                    pass
            if versions:
                next_version = max(versions) + 1
                
        target_filename = f"{SUFFIX_FINAL_KG}_v{next_version}.json"

    target_path = RELEASES_DIR / target_filename
    
    logging.info(f"Publishing {graph_path.name} to {target_path}...")
    shutil.copy2(graph_path, target_path)
    
    print(f"\n Graph published successfully!")
    print(f"  - Versioned: {target_path}")

def run_release_interface():
    """CLI interface for the release manager."""
    while True:
        print("\n" + "="*80)
        print(" RELEASE MANAGER: SELECT GRAPH TO PUBLISH ".center(80))
        print("="*80)
        
        candidates = get_available_graphs()
        
        if not candidates:
            print("No graphs found available for release.")
            input("Press Enter to return...")
            return

        print(f"{'#':<4} | {'Date':<20} | {'Source':<30} | {'File'}")
        print("-" * 80)
        for i, c in enumerate(candidates):
            print(f"{i+1:<4} | {c['date']:<20} | {c['source']:<30} | {c['path'].name}")
            
        print("-" * 80)
        print("[Q] Return to Main Menu")
        
        choice = input("\nSelect graph number to publish: ").strip().upper()
        
        if choice == 'Q':
            return
            
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            selected = candidates[int(choice)-1]
            print(f"\nYou selected: {selected['path'].name} ({selected['date']})")
            
            # Determine next version number for preview
            pattern = f"{SUFFIX_FINAL_KG}_v*.json"
            existing_versions = list(RELEASES_DIR.glob(pattern))
            next_v = 1
            if existing_versions:
                versions = []
                for v in existing_versions:
                    try:
                        parts = v.stem.rsplit("_v", 1)
                        if len(parts) == 2 and parts[0] == SUFFIX_FINAL_KG:
                            versions.append(int(parts[1]))
                    except ValueError: pass
                if versions:
                    next_v = max(versions) + 1

            default_name = f"{SUFFIX_FINAL_KG}_v{next_v}.json"
            print(f"\nProposed filename: {default_name}")
            
            user_input = input("Press [Enter] to accept, type a new name, or 'c' to cancel: ").strip()
            
            if user_input.lower() in ['c', 'cancel']:
                print("Operation cancelled.")
                continue

            final_name = user_input if user_input else default_name
            
            publish_graph(selected['path'], target_filename=final_name)
            input("\nPress Enter to continue...")
        else:
            print("Invalid selection.")
