"""
Extracts triplets from textual evidence using LLMs.
Processes evidence in batches, validates triplet structure, and supports
interactive resumption of interrupted processes.
"""

import json
import datetime
import logging
from string import Template
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

from ..utils.io import load_yaml, load_json, load_txt, save_json
from ..utils.constants import CONFIG_DIR, PROMPT_DIR, ARTICLE_BUILDER_DIR
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import BaseLLM, get_llm_client

def load_evidence(file_path: Path) -> List[Dict[str, Any]]:
    """Loads evidence items from a JSON file."""
    data = load_json(file_path)
    evidence_list = data.get("evidence_list", [])
    logging.info(f"Loaded {len(evidence_list)} evidence items from {file_path}")
    return evidence_list

def chunk_evidence(evidence_list: List[Dict[str, Any]], batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
    """Yields successive n-sized chunks from evidence_list."""
    for i in range(0, len(evidence_list), batch_size):
        yield evidence_list[i:i+batch_size]

def build_prompt(batch: List[Dict[str, Any]], prompt_template: str) -> str:
    """Constructs the LLM prompt for a batch of evidence."""
    evs_batch = "\n".join([f"- Evidence {i}: " + ev.get("evidence_text", "") for i, ev in enumerate(batch)])
    template = Template(prompt_template)
    prompt = template.substitute(evidence_list=evs_batch)
    return prompt

def extract_triplets_from_batch(
    llm_client: BaseLLM, 
    batch: List[Dict[str, Any]], 
    prompt_template: str, 
    retries: int, 
    delay: float
) -> List[Dict[str, Any]]:
    """Sends a batch to the LLM and parses the returned triplets."""
    prompt = build_prompt(batch, prompt_template)
    
    # Estimate tokens (approx 4 chars per token)
    est_tokens = int(len(prompt) / 4)
    logging.info(f"  Prompt size: {len(prompt)} chars (~{est_tokens} tokens)")

    # Call to LLM with retries
    response = llm_client.generate(prompt, retries=retries, delay=delay)

    try:
        triplets_json = json.loads(response)
        if not isinstance(triplets_json, list):
            triplets_json = [triplets_json]

        structured_triplets = []
        REQUIRED_TRIPLET_KEYS = {'subject', 'relation', 'object', 'triplet_topic'}
        
        # Add original evidence from the corresponding evidence to each triplet
        for t in triplets_json:
            # 1. Validate that 't' is a dictionary and has the required structure
            if not isinstance(t, dict):
                logging.warning(f"Skipping malformed triplet (not a dictionary): {t}")
                continue

            if not REQUIRED_TRIPLET_KEYS.issubset(t.keys()):
                logging.warning(f"Skipping malformed triplet, missing one or more required keys: {t}")
                continue

            # 2. Validate and link the evidence using the evidence_id
            try:
                ev_id = int(t.pop("evidence_id"))
                if 0 <= ev_id < len(batch):
                    t.update(batch[ev_id])
                    structured_triplets.append(t)
                else:
                    logging.warning(f"LLM returned an out-of-bounds evidence_id {ev_id} for a batch of size {len(batch)}. Skipping triplet.")
            except (ValueError, KeyError):
                logging.warning(f"Skipping triplet with an invalid or missing 'evidence_id': {t}")
                continue
                
        return structured_triplets
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Error processing batch. Details: {e}. Model response: {response}")
        return []

def run_extract_triplets(article_id: str, output_dir: Optional[Path] = None) -> None:
    """Orchestrates the triplet extraction pipeline for an article."""
    logging.info(f"Processing article: {article_id}")

    # Load configuration and prompt
    try:
        config_path = CONFIG_DIR / "article_builder/settings.yaml"
        config = load_yaml(config_path)["extract_triplets"]
        
        prompt_path = PROMPT_DIR  / "article_builder/extract_triplets.txt"
        prompt_template = load_txt(prompt_path)
        logging.info(f"Configuration and prompt loaded.")
    except Exception:
        sys.exit(1)
        
    run_settings = config["run_settings"]
    model_settings = config["model_settings"]

    # 2. Define input and output paths
    base_root = output_dir if output_dir else ARTICLE_BUILDER_DIR
    article_dir = base_root / article_id 
    evidence_file = article_dir / "extract_evidence" / "evidence.json"

    triplets_output_dir = article_dir / "extract_triplets"
    triplets_output_dir.mkdir(parents=True, exist_ok=True)
    triplets_file = triplets_output_dir / "triplets.json"
    temp_file = triplets_output_dir / "temp_progress.json"

    # 3. Initialize LLM client
    llm_client = get_llm_client(model_settings)

    # 4. Execute pipeline
    # Load evidence and metadata
    evidence_data = load_json(evidence_file)
    evidence_list = evidence_data.get("evidence_list", [])
    execution_measures = evidence_data.get("execution_measures", {})

    # Check for previous execution time to resume timer correctly
    previous_duration = 0.0
    if temp_file.exists():
        try:
            t_data = load_json(temp_file)
            if t_data:
                previous_duration = t_data.get("elapsed_time", 0.0)
        except Exception:
            pass

    with ExecutionMeasures(execution_measures, "triplet_extraction", module_name="extract_triplets", initial_duration=previous_duration) as em:
        batch_size = run_settings.get("batch_size", 10)
        
        all_triplets = []
        triplets_readable = []
        processed_count = 0

        # Load progress if exists
        if temp_file.exists():
            try:
                logging.info(f"Found temporary file: {temp_file}. Loading progress...")
                temp_data = load_json(temp_file)
                if temp_data:
                    all_triplets = temp_data.get("triplets", [])
                    triplets_readable = temp_data.get("triplets_readable", [])
                    
                    # Check for new format 'processed_count'
                    if "processed_count" in temp_data:
                        processed_count = temp_data["processed_count"]
                        logging.info(f"Resuming: {processed_count} evidence items already processed.")
                        
                        # --- Interactive Resume Options ---
                        if processed_count < len(evidence_list):
                            print(f"\n[RESUME DETECTED] Article: {article_id}")
                            print(f"Progress: {processed_count}/{len(evidence_list)} evidence items processed.")
                            print(f"Next batch size: {batch_size}.")
                            print("If the process hung previously, the next batch might contain problematic input.")
                            print("Options:")
                            print("  [Enter] Retry next batch (No skip)")
                            print("  [1]     Skip 1 evidence (Shift window)")
                            print(f"  [b]     Skip entire batch ({batch_size} items)")
                            
                            choice = input("Choice: ").strip().lower()
                            skip_n = 0
                            if choice == '1':
                                skip_n = 1
                            elif choice == 'b':
                                skip_n = batch_size
                            
                            if skip_n > 0:
                                skipped_indices = list(range(processed_count, min(processed_count + skip_n, len(evidence_list))))
                                logging.warning(f"Skipping {len(skipped_indices)} evidence items by user request.")
                                
                                # Log skipped items
                                skipped_log = triplets_output_dir / "skipped_evidence.jsonl"
                                with open(skipped_log, "a", encoding="utf-8") as f:
                                    for idx in skipped_indices:
                                        rec = {
                                            "timestamp": datetime.datetime.now().isoformat(),
                                            "article_id": article_id,
                                            "evidence_idx": idx,
                                            "content": evidence_list[idx]
                                        }
                                        f.write(json.dumps(rec) + "\n")
                                print(f"  -> Skipped items logged to {skipped_log.name}")
                                
                                processed_count += len(skipped_indices)
                                # Update temp file immediately
                                save_json({
                                    "processed_count": processed_count,
                                    "triplets": all_triplets,
                                    "triplets_readable": triplets_readable,
                                    "elapsed_time": em.get_current_duration()
                                }, temp_file)

                    elif "last_batch_index" in temp_data:
                        logging.warning("Temporary file has old format (last_batch_index). Cannot resume safely with variable batch sizes. Starting from scratch.")
                        processed_count = 0
                        all_triplets = []
                        triplets_readable = []
            except Exception as e:
                logging.warning(f"Error loading temporary file: {e}. Starting from scratch.")

        # Slice evidence based on processed count
        evidence_to_process = evidence_list[processed_count:]
        
        if not evidence_to_process:
             logging.info("All evidence items have already been processed.")
             all_batches = []
        else:
            all_batches = list(chunk_evidence(evidence_to_process, batch_size))

        n_batches_to_process = run_settings.get("batches_to_process_limit")
        if n_batches_to_process is not None and n_batches_to_process > 0:
            all_batches = all_batches[:n_batches_to_process]
            logging.info(f"Limiting processing to the first {n_batches_to_process} batches of the remaining evidence.")

        logging.info(f"Running with {len(all_batches)} batches of size {batch_size}")

        for i, batch in enumerate(all_batches):
            logging.info(f"  Extracting triplets: batch {i+1}/{len(all_batches)}")
            triplets = extract_triplets_from_batch(
                llm_client, batch, prompt_template,
                run_settings["llm_retries"]["max_attempts"], 
                run_settings["llm_retries"]["delay_seconds"]
            )
            all_triplets.extend(triplets)
            for t in triplets:
                triplets_readable.append(f"{t['subject']} -> {t['relation']} -> {t['object']} [{t['triplet_topic']}]")

            # Update processed count
            processed_count += len(batch)

            # Save progress
            save_json({
                "processed_count": processed_count,
                "triplets": all_triplets,
                "triplets_readable": triplets_readable,
                "elapsed_time": em.get_current_duration()
            }, temp_file)

        # Register LLM stats
        em.add_llm_stats(llm_client.get_stats())

    output_data = {
        "execution_measures": execution_measures,
        "total_triplets": len(all_triplets),
        "triplets": all_triplets,
        "triplets_readable": triplets_readable,
        "settings": {
            "prompt": prompt_template,
            "model": model_settings,
            "run": run_settings
        },
    }
    # Save triplets
    save_json(output_data, triplets_file)
    
    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()

    logging.info(f"✔ Triplet extraction finished for {article_id}")
    logging.info(f"  Triplets file created - {len(all_triplets)} triplets saved to: {triplets_file}")

if __name__ == "__main__":
    logging.warning("This script is not intended to be run directly. Please use 'main.py'.")