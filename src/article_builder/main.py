"""
Main orchestrator for the Article Builder pipeline. 
Manages the execution of evidence extraction, triplet extraction, graph building, 
and normalization for individual research articles. Supports batch processing, 
individual step execution, and session resumption.
"""

import logging
import datetime
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from ..article_builder.build_graph import run_graph_building
from ..article_builder.extract_evidence import run_extract_evidence
from ..article_builder.extract_triplets import run_extract_triplets
from ..article_builder.normalize_graph import run_graph_normalizer
from ..article_builder.unify_graphs import run_unify_articles
from ..utils.io import load_yaml
from ..utils.constants import CONFIG_DIR, ARTICLE_BUILDER_DIR, SUFFIX_NORMALIZED_KG

# Configure logging to be less verbose than in individual scripts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Centralized definition of pipeline steps
PIPELINE_STEPS = [
    # Step 1
    {"name": "Extract Evidence", "func": run_extract_evidence,
     "output_dir": "extract_evidence", "dependency_dir": None},
    # Step 2
    {"name": "Extract Triplets", "func": run_extract_triplets,
     "output_dir": "extract_triplets", "dependency_dir": "extract_evidence"},
    # Step 3
    {"name": "Build Graph", "func": run_graph_building,
     "output_dir": "build_graph", "dependency_dir": "extract_triplets"}, 
    # Step 4
    {"name": "Normalize Graph", "func": run_graph_normalizer,
     "output_dir": "normalize_graph", "dependency_dir": "build_graph"},
]

def load_articles_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates the articles configuration from a YAML file."""
    try:
        articles_data = load_yaml(config_path)

        if not isinstance(articles_data, dict):
            raise ValueError("Articles config must be a dictionary")

        for key, val in articles_data.items():
            if not isinstance(val, dict) or "doi" not in val:
                raise KeyError(f"Missing 'doi' for article '{key}'")

            if not isinstance(val["doi"], str):
                raise TypeError(f"DOI for article '{key}' must be a string")
            
            val["doi"] = (
                val["doi"]
                .replace("https://doi.org/", "")
                .replace("doi:", "")
                .strip()
                .lower()
            )

        return articles_data

    except FileNotFoundError:
        logging.error(f"Configuration file '{config_path}' not found.")
        raise

    except Exception as e:
        logging.error(f"Invalid articles configuration: {e}")
        raise

def select_article(articles_config: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Displays a menu for the user to select an article."""
    print("\nArticles available in 'config/articles.yaml':")
    article_ids = list(articles_config.keys())
    for i, article_id in enumerate(article_ids):
        print(f"[{i+1}] {article_id}")
    
    while True:
        try:
            choice = input(f"Select an article (1-{len(article_ids)}) or press Enter to cancel: ")
            if not choice:
                return None, None
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(article_ids):
                selected_id = article_ids[choice_idx]
                return selected_id, articles_config[selected_id]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def clean_step_output(article_id: str, step_info: Dict[str, Any], base_dir: Optional[Path] = None) -> None:
    """Cleans the output directory and removes the success flag for a specific step."""
    root = base_dir if base_dir else ARTICLE_BUILDER_DIR
    base_output_dir = root / article_id
    step_output_dir = base_output_dir / step_info["output_dir"]
    success_flag_path = step_output_dir / "_SUCCESS"

    # Remove the success flag
    if success_flag_path.exists():
        success_flag_path.unlink()
    
    # Clean the directory contents to avoid 'zombie' files from previous runs
    # We keep the directory itself, just remove contents
    if step_output_dir.exists():
        for item in step_output_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        step_output_dir.mkdir(parents=True, exist_ok=True)

def run_all_steps(
    article_id: str, 
    article_data: Dict[str, Any], 
    run_dir: Optional[Path] = None, 
    resume: bool = False
) -> bool:
    """Runs all pipeline steps in sequence for an article."""
    print("\n" + "="*80)
    print(f"--- Starting full pipeline for article: {article_id} ---")
    if run_dir:
        print(f"--- Run Directory: {run_dir.name} ---")
    print("="*80)
    
    # Clean all steps first to ensure no zombie states from future steps remain
    if not resume:
        for step in PIPELINE_STEPS:
            clean_step_output(article_id, step, base_dir=run_dir)

    article_pipeline_start_time = time.time()
    
    for i, step in enumerate(PIPELINE_STEPS):
        step_name = f"STEP {i+1}: {step['name'].upper()}"
        func = step["func"]
        # Determine the correct arguments for each function. Steps 0 and 2 need article_data
        # Steps 0 (Extract evidence) and 2 (Build Graph) need article_data
        if i in [0, 2]:
            args = (article_id, article_data, run_dir)
        else:
            args = (article_id, run_dir)
        
        try:
            root = run_dir if run_dir else ARTICLE_BUILDER_DIR
            base_output_dir = root / article_id
            success_flag_path = base_output_dir / step["output_dir"] / "_SUCCESS"

            # If resuming and step is already done, skip it
            if resume and success_flag_path.exists():
                print(f"  [SKIP] {step_name} (Already completed)")
                continue

            print("\n" + "-"*28 + f" {step_name} " + "-"*28)
            
            # If we are running the step, clean its specific output first to be safe
            if not resume:
                clean_step_output(article_id, step, base_dir=run_dir)
            
            func(*args)
            success_flag_path.touch() # Create the flag only if the function finishes successfully
        except Exception as e:
            logging.error(f"\nERROR during '{step_name}' for article '{article_id}'. Reason: {e}", exc_info=True)
            print(f"Skipping remaining steps for {article_id}.")
            return False # Stop execution for this article and indicate failure
            
    article_pipeline_end_time = time.time()
    execution_time = article_pipeline_end_time - article_pipeline_start_time
    print("\n" + "="*80)
    print(f"--- Full pipeline for '{article_id}' COMPLETED in {execution_time:.2f} seconds ---")
    results_path = (run_dir if run_dir else ARTICLE_BUILDER_DIR) / article_id
    print(f"Results directory: {results_path}")
    print("="*80)
    return True

def get_pipeline_status(article_id: str, base_dir: Optional[Path] = None) -> Dict[int, bool]:
    """Checks which pipeline steps have already been completed."""
    root = base_dir if base_dir else ARTICLE_BUILDER_DIR
    base_output_dir = root / article_id
    status = {}
    for i, step in enumerate(PIPELINE_STEPS):
        success_flag_path = base_output_dir / step["output_dir"] / "_SUCCESS"
        status[i + 1] = success_flag_path.exists()
    return status

def show_task_menu(article_id: str, article_data: Dict[str, Any], run_dir: Optional[Path] = None) -> None:
    """Displays the task menu for an article and manages the selection."""
    while True:
        statuses = get_pipeline_status(article_id, base_dir=run_dir)
        
        root = run_dir if run_dir else ARTICLE_BUILDER_DIR

        print("\n" + "-"*28 + f" Processing Article: {article_id} " + "-"*28)
        print(f"Output Dir: {root / article_id}")
        print("Which step do you want to run?")
        for i, step in enumerate(PIPELINE_STEPS):
            status_marker = "✔ DONE" if statuses.get(i + 1) else "  PENDING"
            print(f"[{i+1}] {step['name']} {' '*(22-len(step['name']))} [{status_marker}]")

        print("-" * 70)
        print("[A] Run ALL steps in sequence")
        print("[B] Back to article selection")
        print("[Q] Exit program")

        choice = input("Your choice: ").strip().upper()

        if choice.isdigit() and 1 <= int(choice) <= len(PIPELINE_STEPS):
            step_index = int(choice) - 1
            step_info = PIPELINE_STEPS[step_index]
            is_completed = statuses.get(step_index + 1)

            # --- Confirmation to re-run ---
            if is_completed:
                confirm = input(f"\n[NOTICE] Step {step_index + 1} has already been completed. Do you want to run it again? (This will overwrite the results) [y/N]: ").strip().lower()
                if confirm != 'y':
                    print("      Operation cancelled. Returning to menu.")
                    continue

            # --- Dependency Check ---
            if step_info["dependency_dir"]:
                # The check should be for the previous step (step_index), not step_index + 1
                if step_index > 0 and not statuses.get(step_index):
                    print(f"\n[ERROR] Cannot run Step {step_index + 1}.")
                    print(f"      Dependency Step {step_index} has not been completed successfully.")
                    print(f"      Please run the previous step first.")
                    continue # Back to menu
            # --- End of Check ---

        try:
            if choice.isdigit() and 1 <= int(choice) <= len(PIPELINE_STEPS):
                step_index = int(choice) - 1
                step_info = PIPELINE_STEPS[step_index]
                func = step_info["func"]
                
                # 1. Clean the current step output
                clean_step_output(article_id, step_info, base_dir=run_dir)

                # 2. Invalidate subsequent steps (Cascade invalidation)
                base_output_dir = root / article_id
                for future_step_idx in range(step_index + 1, len(PIPELINE_STEPS)):
                    future_step = PIPELINE_STEPS[future_step_idx]
                    clean_step_output(article_id, future_step, base_dir=run_dir)
                    print(f"      [INFO] Invalidated downstream step: {future_step['name']}")

                # Run the selected step
                success_flag_path = base_output_dir / step_info["output_dir"] / "_SUCCESS"
                # The 0 y 2 steps need 'article_data', the others only 'article_id'
                if step_index in [0, 2]:
                    args = (article_id, article_data, run_dir)
                else:
                    args = (article_id, run_dir)
                
                func(*args)
                success_flag_path.touch() # Create the flag only if the function finishes successfully
            elif choice == 'A':
                # --- OVERWRITE VERIFICATION ---
                statuses = get_pipeline_status(article_id, base_dir=run_dir)
                if any(statuses.values()):
                    confirm = input(f"\n[NOTICE] Article '{article_id}' already has completed steps. Run again and overwrite? [y/N]: ").strip().lower()
                    if confirm != 'y':
                        print(f"      Operation cancelled. Returning to task menu.")
                        continue
    
                run_all_steps(article_id, article_data, run_dir=run_dir)
            elif choice == 'B':
                return # Exits this menu to return to the main one
            elif choice == 'Q':
                print("Exiting program.")
                sys.exit(0)
            else:
                print("Invalid option, please try again.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during step execution: {e}", exc_info=True)
            print("\nAn unexpected error occurred. Returning to the task menu.")

def select_and_run_unification() -> None:
    """Select and run the unification of processed graphs."""
    
    # Find all consolidated graphs recursively (to support runs structure)
    # Searches in runs/article_builder/**/kg_*.json
    pattern = f"**/kg_{SUFFIX_NORMALIZED_KG}.json"
    available_graphs = sorted(list(ARTICLE_BUILDER_DIR.glob(pattern)))

    if not available_graphs:
        print("\n[ERROR] No processed article graphs (kg_*.json) found in the 'output' directory.")
        print("      Please run the full pipeline for at least one article first (Option 1 or 2).")
        input("Press Enter to return to the main menu...")
        return

    print("\n" + "="*28 + " Unify Article Graphs " + "="*28)
    print("The following processed graphs are available for unification:")
    for i, graph_path in enumerate(available_graphs):
        # Extract article_id from the parent directory name for better display
        article_id = graph_path.parent.name
        print(f"[{i+1}] {article_id} (Found at: {graph_path})")

    while True:
        print("\nSelect the graphs to unify.")
        print("You can select multiple by separating numbers with commas (e.g., 1, 3, 4).")
        print("Or type 'A' to select ALL.")
        choice = input("Your choice (or press Enter to cancel): ").strip().upper()

        if not choice:
            return

        selected_indices = []
        try:
            if choice == 'A':
                selected_indices = list(range(len(available_graphs)))
            else:
                selected_indices = [int(i.strip()) - 1 for i in choice.replace(",", " ").split()]

            # Validate indices
            if all(0 <= idx < len(available_graphs) for idx in selected_indices):
                selected_files = [available_graphs[i] for i in selected_indices]
                print("\nGraphs selected for unification:")
                for f in selected_files:
                    print(f"  - {f.parent.name}/{f.name}")
                
                confirm = input("Proceed with unification? [Y/n]: ").strip().lower()
                if confirm in ('', 'y'):
                    run_unify_articles(selected_files)
                else:
                    print("Unification cancelled.")
                break # Exit the selection loop
            else:
                print("[ERROR] Invalid selection. One or more numbers are out of range.")
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers separated by commas, or 'A'.")

def handle_resume_run(articles_config: Dict[str, Any]) -> None:
    """Handles the interaction to resume an interrupted run."""
    runs = sorted(
        [d for d in ARTICLE_BUILDER_DIR.iterdir() if d.is_dir() and d.name[0].isdigit() and not (d / "unified_graphs").exists()],
        key=lambda x: x.name,
        reverse=True
    )
    
    if not runs:
        print("\n[INFO] No previous runs found to resume.")
        return

    print("\nSelect a previous run to resume:")
    for i, r in enumerate(runs):
        print(f"[{i+1}] {r.name}")
    
    try:
        choice = input("Select run number (or press Enter to cancel): ")
        if not choice:
            return

        if not choice.isdigit() or not (1 <= int(choice) <= len(runs)):
            print("Invalid selection.")
            return
        
        selected_run = runs[int(choice)-1]
        
        # --- Identify status for ALL articles in the config for this run ---
        all_articles_status = []
        incomplete_articles = []
        
        # Iterate over articles defined in config to ensure we catch those not yet started in the run
        for article_id in articles_config.keys():
            status = get_pipeline_status(article_id, base_dir=selected_run)
            
            if all(status.values()):
                status_text = "ALL COMPLETED"
            else:
                # Determine next step
                status_text = "Next: UNKNOWN" # Fallback
                for i, step in enumerate(PIPELINE_STEPS):
                    if not status.get(i + 1):
                        status_text = f"Next: Step {i+1}: {step['name']}"
                        break
                incomplete_articles.append((article_id, status_text))
            
            all_articles_status.append((article_id, status_text))

        # --- Display Full Plan ---
        print(f"\nResume strategy for run: {selected_run.name}")
        print("-" * 80)
        print(f"{'Article ID':<35} | {'Status'}")
        print("-" * 80)
        for aid, text in all_articles_status:
            print(f"{aid:<35} | {text}")
        print("-" * 80)
        
        if not incomplete_articles:
            print(f"\n[INFO] All articles in run '{selected_run.name}' are already completed.")
            return

        # --- Article Selection for Resume (only incomplete ones) ---
        print("\nWhich articles do you want to resume?")
        print("You can select multiple by separating numbers with commas (e.g., 1, 3).")
        print("Or type 'A' to resume ALL (only incomplete ones).")
        
        available_ids = [item[0] for item in incomplete_articles] # Only list incomplete articles for selection
        for i, aid in enumerate(available_ids):
            print(f"[{i+1}] {aid}")

        choice_arts = input("Your choice (or press Enter to cancel): ").strip().upper()
        if not choice_arts:
            return

        selected_articles = []
        if choice_arts == 'A':
            selected_articles = available_ids
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice_arts.replace(",", " ").split()]
                selected_articles = [available_ids[i] for i in indices if 0 <= i < len(available_ids)]
            except ValueError:
                print("Invalid input.")
                return

        if not selected_articles:
            print("No valid articles selected.")
            return

        print(f"\n[INFO] Resuming run: {selected_run.name} for {len(selected_articles)} articles.")
        print(f"Articles: {selected_articles}")

        for article_id in selected_articles:
            run_all_steps(article_id, articles_config[article_id], run_dir=selected_run, resume=True)
            
    except Exception as e:
        print(f"Error resuming run: {e}")

def main() -> None:
    """Main function that runs the interactive menu."""

    try:
        config_path = CONFIG_DIR / "article_builder" / "articles.yaml"
        articles_config = load_articles_config(config_path)
    except Exception:
        sys.exit(1)

    
    menu_width = 60
    title = " MODULE: BUILD_GRAPH "
    while True:
        print("\n" + title.center(menu_width, "="))
        print(f"Articles are loaded from 'config/articles.yaml'".center(menu_width))
        print("\n  [1] Create a graph from all articles")
        print("  [2] Create a graph from an article")
        print("  [3] Unify processed graphs")
        print("  [4] Resume interrupted run")
        print("  ---")
        print("  [R] Return to main menu")
        print("  [Q] Exit")
        print("="*menu_width)
        main_choice = input("Your choice: ").strip().upper()

        try:
            if main_choice == '1':
                
                # Create a unique RUN ID for this batch execution
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_dir = ARTICLE_BUILDER_DIR / timestamp
                print(f"\n[INFO] Starting new run session: {timestamp}")
                
                # Confirm operation before continue
                print("The following items defined in 'config/articles.yaml' will be processed:")
                print(f"  - {list(articles_config.keys())}")
                print(
                    "For each item, the following steps will be executed:\n"
                    "  • Extract Evidence\n"
                    "  • Extract Triplets\n"
                    "  • Build Graph\n"
                    "  • Normalize Graph"
                )
                if input("\n[NOTICE] Confirm to continue? [y/N]: ").strip().lower() != "y":
                    continue

                print("\nProcessing all articles...")
                total_start_time = time.time()
                for article_id, article_data in articles_config.items():
                    # No overwrite check needed because we are in a new timestamped folder
                    success = run_all_steps(article_id, article_data, run_dir=run_dir)
                    if not success:
                        print(f"\n[WARNING] Pipeline failed for article '{article_id}'. Check logs for details.")
                        # We continue with the next article in the batch, or break if desired.
                        # Usually in batch processing, we want to try the rest.
                    
                total_end_time = time.time()
                total_time = total_end_time - total_start_time
                print("\n" + "="*80)
                print(f"PROCESSING OF ALL ARTICLES FINISHED. Total time: {total_time:.2f} seconds")
                print("="*80)

            elif main_choice == '2':
                article_id, article_data = select_article(articles_config)
                if article_id:
                    # Create a unique RUN ID for this single article session too
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    run_dir = ARTICLE_BUILDER_DIR / timestamp
                    print(f"\n[INFO] Starting new run session for single article: {timestamp}")
                    show_task_menu(article_id, article_data, run_dir=run_dir)
            
            elif main_choice == '3':
                select_and_run_unification()

            elif main_choice == '4':
                handle_resume_run(articles_config)

            elif main_choice == 'R':
                break
            elif main_choice == 'Q':
                print("Exiting program.")
                sys.exit(0)
            else:
                print("Invalid option, please try again.")
        except Exception as e:
            logging.error(f"Critical error in main menu: {e}", exc_info=True)
            print(f"\n[ERROR] A critical error occurred: {e}")
            print("Returning to main menu...")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
