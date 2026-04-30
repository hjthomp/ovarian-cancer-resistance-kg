"""
Main entry point for the Experiments module.
Provides an interactive menu to run various analysis and evaluation scripts
on the Knowledge Graph releases.
"""

import sys
import logging

# Import individual experiment scripts
from .calculate_statistics import run_statistics
from .expert_eval_calculate_metrics import process_metrics
from .expert_eval_splits import main as expert_eval_splits_main
from .link_prediction_eval import run_link_prediction_eval
from .ner_pipeline import main as ner_pipeline_main
from .plot_ner_results import run_analysis_generator
from .plot_wordclouds import run_wordclouds

# Configure logging (if not already configured by the root main)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """Displays the interactive CLI menu for the Experiments module."""
    menu_width = 60
    title = " MODULE: EXPERIMENTS "
    while True:
        print("\n" + title.center(menu_width, "="))
        print("  [1] Calculate Statistics")
        print("  [2] Expert Evaluation: Generate Splits")
        print("  [3] Expert Evaluation: Calculate Metrics")
        print("  [4] Link Prediction Evaluation")
        print("  [5] NER Pipeline (Extraction & Normalization)")
        print("  [6] Plot NER Results")
        print("  [7] Plot WordClouds")
        print("  ---")
        print("  [R] Return to main menu")
        print("  [Q] Exit Application")
        print("="*menu_width)
        
        choice = input("Select experiment: ").strip().upper()
        
        try:
            if choice == '1':
                run_statistics()
            elif choice == '2':
                expert_eval_splits_main()
            elif choice == '3':
                process_metrics()
            elif choice == '4':
                run_link_prediction_eval()
            elif choice == '5':
                ner_pipeline_main()
            elif choice == '6':
                run_analysis_generator()
            elif choice == '7':
                run_wordclouds()
            elif choice == 'R':
                break
            elif choice == 'Q':
                print("Exiting application. Goodbye!")
                sys.exit(0)
            else:
                print("Invalid option.")
        except SystemExit:
            # Catch SystemExit from sub-modules if they decide to exit the whole app
            raise
        except Exception as e:
            logging.error(f"An error occurred during experiment execution: {e}", exc_info=True)
            print(f"\n[ERROR] An error occurred: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
