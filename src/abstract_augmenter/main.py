"""
Main orchestrator for the Abstract Augmenter module.
Manages Knowledge Graph augmentation with PubMed abstracts, including
new runs, session resumption, normalization, and graph refinement.
"""

import logging
import sys

from ..abstract_augmenter.refine_graph import refine_graph
from ..abstract_augmenter.normalize_graph import normalize_completed_graph
from ..abstract_augmenter.augment_graph import create_run_from_graph, resume_run, generate_partial_graph_from_incomplete_run, add_single_abstract
from ..utils.constants import PROJECT_ROOT, ARTICLE_BUILDER_DIR, ABSTRACT_AUGMENTER_DIR


# --- Logging and Path Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root to sys.path to allow imports from other directories
sys.path.append(str(PROJECT_ROOT))

def main() -> None:
    """Main function to orchestrate the graph augmentation process."""
    
    menu_width = 100
    title = " MODULE: AUGMENT_GRAPH "
    while True:
        print("\n" + title.center(menu_width, "="))
        print("  PubMed Augmentation Runs")
        print(f"  [1] Start new run from Article Graph              (Loads from '{ARTICLE_BUILDER_DIR.relative_to(PROJECT_ROOT)}')")
        print(f"  [2] Start new run from Augmented Graph            (Loads from '{ABSTRACT_AUGMENTER_DIR.relative_to(PROJECT_ROOT)}')")
        print("  [3] Resume an incomplete run")
        print("  [4] Generate a partial graph from an incomplete run")
        print("  ---")
        print("  Single-Article Augmentation")
        print(f"  [5] Add a single abstract from Article Graph      (Loads from '{ARTICLE_BUILDER_DIR.relative_to(PROJECT_ROOT)}')")
        print(f"  [6] Add a single abstract from Augmented Graph    (Loads from '{ABSTRACT_AUGMENTER_DIR.relative_to(PROJECT_ROOT)}')")
        print("  ---")
        print("  Post-processing (only for complete runs)")
        print("  [7] Normalize an Augmented Graph")
        print("  [8] Refine a Graph")
        print("  ---")
        print("  [R] Return to main menu")
        print("  [Q] Exit")
        print("="*menu_width)
        
        choice = input("Select an option: ").strip().lower()
        if choice == "1":
            create_run_from_graph(mode="article")
        elif choice == "2":
            create_run_from_graph(mode="augmented")
        elif choice == "3":
            resume_run()
            pass
        elif choice == "4":
            generate_partial_graph_from_incomplete_run()
            pass
        elif choice == "5":
            add_single_abstract(mode="article")
            pass
        elif choice == "6":
            add_single_abstract(mode="augmented")
            pass
        elif choice == "7":
            normalize_completed_graph()
            pass
        elif choice == "8":
            refine_graph()
            pass
        elif choice == "r":
            break
        elif choice == "q":
            print("Exiting...")
            sys.exit()
            # break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()