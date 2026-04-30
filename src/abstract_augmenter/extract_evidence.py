"""
Extracts evidence and triplets from PubMed abstracts using LLMs.
Manages state for batch processing and supports resuming interrupted extraction 
tasks with detailed error logging and execution metrics.
"""

import hashlib
import json
import logging
from pathlib import Path
from string import Template
from typing import List, Dict, Any, Optional

from ..utils.io import load_json, load_txt, save_json
from ..utils.constants import PROMPT_DIR
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import get_llm_client

def load_state(state_file: Path) -> Dict[str, Any]:
    """Loads the processing state from a file."""
    if state_file.exists():
        data = load_json(state_file)
        return data
    # Initial state
    return {"processed_item_ids": [], "failed_item_ids": [], "extracted_evidence": [], "total_execution_time": 0.0}

def get_item_id(item: Dict[str, Any], fallback_index: Optional[int] = None) -> str:
    """
    Generate a stable unique identifier for an item.
    
    - If the item has a DOI, the DOI (lowercased and stripped) is used.
    - Otherwise, a hash of the title is used, optionally suffixed with `fallback_index`.
    - Raises ValueError if both DOI and title are missing.

    Args:
        item (dict): Dictionary representing the item. Expected keys: 'doi' (optional), 'title' (optional).
        fallback_index (int, optional): Suffix to differentiate items with identical titles. Defaults to None.

    Returns:
        str: A unique identifier string, e.g., "10.1000/xyz123" or "no-doi-5-3f5c2a1b9d".

    Raises:
        ValueError: If item has neither DOI nor title.

    Example:
        item = {"title": "Chemoresistance in ovarian cancer", "doi": "10.1000/xyz123"}
        generate_unique_id(item)
        # Returns: "10.1000/xyz123"

        item = {"title": "Unknown article"}
        generate_unique_id(item, fallback_index=5)
        # Returns: "no-doi-5-3f5c2a1b9d"  # stable hash of title
    """
    if item.get("doi"):
        return item["doi"].strip().lower()

    title = item.get("title", "").strip().lower()
    if not title:
        raise ValueError("Item without DOI and without title cannot be identified")

    suffix = f"-{fallback_index}" if fallback_index is not None else ""
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    return f"no-doi{suffix}-{digest}"

def run_extract_evidence(
    items_to_process: List[Dict[str, Any]], 
    cluster_names: List[str], 
    config: Dict[str, Any], 
    run_dir: Path
) -> Dict[str, Any]:
    """
    Main function to extract triplets from a list of items (abstracts or text).
    Manages state to allow for resumption.
    """

    run_settings = config['run_settings']
    model_settings = config['model_settings']
    
    # Setup paths and state
    state_filename = "processing_state.json"
    state_file = run_dir / state_filename
    error_log_file = run_dir / "error_log.txt"

    state = load_state(state_file)
    
    # Initialize execution_measures in state if not present
    if "execution_measures" not in state:
        state["execution_measures"] = {}
        # Migrate legacy time if exists
        if "total_execution_time" in state:
             state["execution_measures"]["evidence_extraction_augmentation"] = {
                 "duration_seconds": state["total_execution_time"],
                 "status": "interrupted"
             }
    # Ensure 'failed_item_ids' exists for backward compatibility with old state files
    if 'failed_item_ids' not in state:
        state['failed_item_ids'] = []
    logging.info(f"Resuming process. {len(state['processed_item_ids'])} items successfully processed, {len(state['failed_item_ids'])} failed in previous runs.")

    # Initialize LLM client
    llm_client = get_llm_client(model_settings)

    # Prepare items to process for this run
    processed_ids = set(state['processed_item_ids']) | set(state['failed_item_ids'])

    items_to_analyze = [
        item
        for idx, item in enumerate(items_to_process)
        if get_item_id(item, idx) not in processed_ids
    ]  

    # Ask to process all items or skip
    if state['processed_item_ids'] and items_to_analyze:
        print("If the process hung previously, the next batch might contain problematic input.")
        print("\nOptions:")
        print("  [Enter] Retry next item (No skip)")
        print("  [1]     Skip next item")
        choice = input("Choice: ").strip().lower()
        if choice == '1':
            item = items_to_analyze[0]
            unique_id = get_item_id(item, items_to_process.index(item)) #items_to_process is needed to generate the ID
            items_to_analyze.pop(0) #remove the skipped item
            state['failed_item_ids'].append(unique_id)
            logging.info(f"Skipping item {unique_id} at user request.")
            save_json(state, state_file)

    max_per_run = run_settings.get("max_processing_steps")
    if max_per_run and max_per_run > 0:
        items_to_analyze = items_to_analyze[:max_per_run]
        logging.warning(f"'max_processing_steps' is set to {max_per_run}. Processing a batch of {len(items_to_analyze)} items.")

    total_to_analyze = len(items_to_analyze)
    logging.info(f"{total_to_analyze} new items to process in this run.")

    # Load Prompt Template
    prompt_template_path = PROMPT_DIR / "abstract_augmenter/extract_evidence.txt" 
    prompt_template = load_txt(prompt_template_path)

    # Determine initial duration for this specific stage
    initial_duration = 0.0
    if "evidence_extraction_augmentation" in state["execution_measures"]:
        initial_duration = state["execution_measures"]["evidence_extraction_augmentation"].get("duration_seconds", 0.0)

    with ExecutionMeasures(state["execution_measures"], "evidence_extraction_augmentation", module_name="extract_evidence", initial_duration=initial_duration) as em:
        # Process each item
        for i, item in enumerate(items_to_analyze):
            title = item.get('title', 'N/A')
            abstract = item.get('abstract', 'N/A')
            
            # Use DOI if available, otherwise create a unique ID based on the title and index
            unique_id = get_item_id(item, i)

            authors = item.get('authors', [])
            year = item.get('year', 'N/A')
            journal = item.get('journal', 'N/A')
            if 'N/A' in [title, abstract]:
                logging.warning(f"Skipping item {unique_id} due to missing title or abstract.")
                state['failed_item_ids'].append(unique_id) # Mark as failed
                continue

            logging.info(f"Processing item {i+1}/{total_to_analyze} (ID: {unique_id})...")
            
            # Convert list of cluster names to a formatted string for the prompt
            theme_list_str = ""
            for name in cluster_names:
                theme_list_str += f"- '{name}'\n"
            
            template = Template(prompt_template)
            full_prompt = template.substitute(theme_list=theme_list_str, input_text=abstract)
            
            # Estimate tokens (approx 4 chars per token)
            est_tokens = int(len(full_prompt) / 4)
            logging.info(f"  Prompt size: {len(full_prompt)} chars (~{est_tokens} tokens)")

            # Call to LLM with retries
            response = llm_client.generate(full_prompt, retries=run_settings["llm_retries"]["max_attempts"], delay=run_settings["llm_retries"]["delay_seconds"])
            try:

                parsed_json = json.loads(response)

                # Unify handling of list vs. dict response
                if isinstance(parsed_json, list) and parsed_json:
                    result = parsed_json[0]
                elif isinstance(parsed_json, dict):
                    result = parsed_json
                    
            except json.JSONDecodeError as e:
                # This error occurs if the LLM response is not valid JSON.
                logging.error(f"JSON parsing error for item {unique_id}. Raw output was: '{response}'. Details: {e}")
                state['failed_item_ids'].append(unique_id)
                continue # Skip to the next item, as we can't process this response
            except Exception as e:
                # Catch other potential errors during LLM call or processing
                logging.error(f"An unexpected error occurred for item {unique_id}: {e}")
                state['failed_item_ids'].append(unique_id)
                continue # Skip to the next item

            # Check if the expected structure is present in the LLM response
            if result and all(k in result for k in ("summary", "triplets")):
                raw_triplets = result.get("triplets", [])
                valid_triplets = []

                if raw_triplets: # Only add if triplets were actually extracted
                    REQUIRED_KEYS = {'subject', 'relation', 'object'}
                    for t in raw_triplets:
                        # Validate structure
                        if not isinstance(t, dict) or not REQUIRED_KEYS.issubset(t.keys()):
                            logging.warning(f"Skipping malformed triplet in item '{unique_id}': {t}")
                            continue

                    # Normalize relations to snake_case
                        if 'relation' in t and isinstance(t['relation'], str):
                            t['relation'] = "_".join(t['relation'].strip().lower().split())
                        
                        valid_triplets.append(t)

                if valid_triplets:
                    new_evidence = {
                        "doi": unique_id, # Keep the original DOI field, which can be None
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "generated_summary": result.get("summary", "N/A"),
                        "extracted_triplets": valid_triplets
                    }
                    state['extracted_evidence'].append(new_evidence)
                    logging.info(f"Extracted: a summary and {len(valid_triplets)} triplets for '{unique_id}'.")
                    # IMPORTANT: Add ID to processed list only on success
                    state['processed_item_ids'].append(unique_id)
                else:
                    # Log if the structure is valid but there are no triplets
                    logging.warning(f"No valid triplets extracted for item '{unique_id}'.")
                    state['failed_item_ids'].append(unique_id)
            else:
                # This now catches parsing errors (result=None) and structural errors
                error_message = f"Failed to extract valid data structure for item '{unique_id}'. Parsed JSON: {parsed_json}\n"
                logging.error(error_message)
                with open(error_log_file, "a", encoding='utf-8') as log:
                    log.write(error_message)
                state['failed_item_ids'].append(unique_id)

            # Save state periodically
            if (i + 1) % run_settings.get('save_state_interval', 10) == 0:
                state["total_processed_items"] = len(state['processed_item_ids'])
                state["total_failed_items"] = len(state['failed_item_ids'])
                # Update duration for checkpoint
                state["execution_measures"]["evidence_extraction_augmentation"]["duration_seconds"] = round(em.get_current_duration(), 2)
                save_json(state, state_file)
                logging.info(f"--- Progress saved. {len(state['processed_item_ids'])} successful, {len(state['failed_item_ids'])} failed. ---")
        
        # Register LLM stats
        em.add_llm_stats(llm_client.get_stats())

    # 6. Save final state
    save_json(state, state_file)
    logging.info(f"Extraction run complete. ")
    
    return state
