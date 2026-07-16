"""
Main entry point and orchestrator for the Ovarian Cancer Knowledge Graph project.
Provides an interactive menu to navigate through the article building, 
abstract augmentation, release management, experiments, interactive visualization,
and GraphRAG modules.
"""

import sys
import logging
import subprocess
import webbrowser

# Import sub-modules
from src.article_builder.main import main as article_builder_main
from src.abstract_augmenter.main import main as abstract_augmenter_main
from src.experiments.main import main as experiments_main
from src.utils.release_manager import run_release_interface
from src.utils.constants import PROJECT_ROOT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """Displays the interactive CLI menu to manage the Knowledge Graph pipeline."""
    while True:
        print("\n" + "="*60)
        print(" OVARIAN CANCER KNOWLEDGE GRAPH ".center(60))
        print("="*60)
        print("  [1] Article Builder (PDFs -> Graph)")
        print("  [2] Abstract Augmenter (PubMed -> Augmentation)")
        print("  [3] Release Manager (Publish to 'data/releases')")
        print("  [4] Experiments & Analysis")
        print("  [5] Interactive Visualization (Web)")
        print("  [6] Reasoning Framework (Streamlit)")
        print("  ---")
        print("  [Q] Exit Application")
        print("="*60)
        
        choice = input("Select module: ").strip().upper()
        
        if choice == '1':
            try:
                # The sub-modules have their own loops and use 'R' to break/return
                article_builder_main()
            except SystemExit:
                # If the submodule calls sys.exit(), we catch it if we want to stay alive,
                # but currently they are designed to exit or return.
                pass 
        elif choice == '2':
            try:
                abstract_augmenter_main()
            except SystemExit:
                pass
        elif choice == '3': # Release Manager
            run_release_interface()
        elif choice == '4': # Experiments
            try:
                experiments_main()
            except SystemExit:
                pass
        elif choice == '5': # Visualization
            viz_path = PROJECT_ROOT / "src"/ "visualization" / "index.html"
            if viz_path.exists():
                print(f"\n[INFO] Opening visualization in your browser...")
                print(f"Path: {viz_path}")
                webbrowser.open(f"file://{viz_path.absolute()}")
            else:
                print(f"\n[ERROR] Visualization file not found at: {viz_path}")
                print("Make sure you have moved the 'client' files to the 'visualization' folder.")
        elif choice == '6': # GraphRAG
            app_path = PROJECT_ROOT / "src" / "graph_rag" / "app.py"
            if app_path.exists():
                print(f"\n[INFO] Launching GraphRAG Dashboard...")
                try:
                    # Executes streamlit run as a blocking process
                    subprocess.run(["streamlit", "run", str(app_path)])
                except KeyboardInterrupt:
                    print("\n[INFO] GraphRAG session ended.")
                except Exception as e:
                    print(f"\n[ERROR] Could not start GraphRAG: {e}")
            else:
                print(f"\n[ERROR] GraphRAG app not found at: {app_path}")
        elif choice == 'Q':
            print("Exiting application. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()