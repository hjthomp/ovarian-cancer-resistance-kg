"""
Named Entity Recognition (NER) pipeline for knowledge graph nodes.
Combines LLM-based entity extraction with rule-based normalization.
Features semantic batching for context optimization and session resumption.
"""

import csv
import datetime
import json
import re
import logging
import time
import sys
from pathlib import Path
from string import Template
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from itertools import chain

# External
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

# Project Utils
from src.utils.io import load_txt, load_yaml, load_json, save_json
from src.utils.constants import CONFIG_DIR, PROMPT_DIR, DATA_DIR
from src.utils.execution_measures import ExecutionMeasures
from src.utils.llm_calls import get_llm_client

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
RELEASES_DIR = DATA_DIR / "releases"
NER_OUTPUT_DIR = DATA_DIR / "experiments" / "ner"
CONFIG_PATH = CONFIG_DIR / "experiments" / "ner.yaml"
PROMPT_PATH = PROMPT_DIR / "experiments" / "ner.txt"

def get_user_choice(options: List[Any], prompt_text: str) -> Any:
    """Helper to select an option from a list."""
    print(f"\n{prompt_text}")
    for i, option in enumerate(options, 1):
        if isinstance(option, Path):
            try:
                display = option.relative_to(DATA_DIR)
            except ValueError:
                display = option
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

def load_config() -> Dict[str, Any]:
    """Loads the centralized NER configuration."""
    if not CONFIG_PATH.exists():
        logging.error(f"Config file not found: {CONFIG_PATH}")
        sys.exit(1)
    return load_yaml(CONFIG_PATH)

# --- EXTRACTION STAGE ---

def load_graph_triplets_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """Loads triplets from a knowledge graph JSON file."""
    kg_data = load_json(file_path)
    graph_data = kg_data.get("graph", kg_data)  # Handle nested graph key

    triplets = []

    # The graph can be a dict of themes or a flat list
    if isinstance(graph_data, dict):
        # It's a dictionary of themes, e.g., {"Theme 1": [triplets], "Theme 2": [triplets]}
        triplet_lists = graph_data.values()
    elif isinstance(graph_data, list):
        # It's a flat list of triplets
        triplet_lists = [graph_data]
    else:
        logging.error("Unsupported graph format in JSON file.")
        return []

    for triplet_list in triplet_lists:
        for t in triplet_list:
            if 'subject' in t and 'relation' in t and 'object' in t:
                # The structure is already what we need.
                # The original script put the whole row in 'original_row'.
                # Here, the triplet 't' is the original row.
                triplets.append({
                    "subject": t['subject'],
                    "relation": t['relation'],
                    "object": t['object'],
                    "original_row": t
                })
    return triplets

def save_progress(
    run_dir: Path, 
    processed_nodes: Set[str], 
    results: Dict[str, Any], 
    metrics: Dict[str, Any]
) -> None:
    """Saves the current progress to a JSON file."""
    progress_file = run_dir / "progress.json"
    data = {
        "processed_nodes": list(processed_nodes),
        "results": results,
        "metrics": metrics
    }
    save_json(data, progress_file)

def semantic_sort_nodes(
    model_name: str, 
    nodes: List[str], 
    llm_batch_size: int, 
    embedding_batch_size: int = 32
) -> List[str]:
    """Semantically sorts nodes to optimize LLM context."""
    if not nodes:
        return []
        
    logging.info("Generating embeddings for semantic batching...")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(nodes, batch_size=embedding_batch_size, show_progress_bar=True)

    n_clusters = max(1, len(nodes) // llm_batch_size)
    logging.info(f"Clustering {len(nodes)} nodes into {n_clusters} semantic groups...")
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
    cluster_labels = kmeans.fit_predict(embeddings)

    clustered_nodes = sorted(zip(nodes, cluster_labels), key=lambda x: x[1])
    sorted_nodes = [node for node, label in clustered_nodes]
    
    return sorted_nodes

def process_batch(
    llm_client: Any, 
    batch: List[str], 
    prompt_template: str, 
    retries: int = 3, 
    delay: float = 2.0
) -> Optional[Dict[str, Any]]:
    """Processes a batch of nodes with the LLM."""
    input_texts = "\n".join(batch)
    template = Template(prompt_template)
    prompt = template.substitute(input_texts=input_texts)
    
    # Simple token estimation
    est_tokens = len(prompt) // 4
    logging.info(f"  Prompt size: ~{est_tokens} tokens")

    start_time = time.time()
    response = llm_client.generate(prompt, retries=retries, delay=delay)
    elapsed = time.time() - start_time
    logging.info(f"  LLM Generation time: {elapsed:.2f}s")
    
    try:
        # Extract JSON 
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_str = match.group(0)
            # Common cleaning for malformed JSON from LLMs
            json_str = re.sub(r',(\s*[\}\]])', r'\1', json_str)
            return json.loads(json_str)
        else:
            cleaned = re.sub(r"```json|```", "", response).strip()
            return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse JSON: {e}")
        return None

def run_extraction_stage() -> Optional[Path]:
    """Runs the NER extraction stage."""
    NER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Step 1: NER Extraction ---")
    action = get_user_choice(["New Run", "Resume Run"], "Select extraction mode:")
    
    run_dir = None
    triplets = []
    global_entity_counts = defaultdict(int)
    processed_nodes = set()
    node_extractions = {}
    execution_measures = {}

    # 1. Setup Run Directory & Input
    if action == "New Run":
        json_files = list(RELEASES_DIR.glob("*.json"))
        if not json_files:
            logging.error(f"No JSON graph files found in {RELEASES_DIR}")
            return None
        
        input_file = get_user_choice(json_files, "Select input graph JSON file:")
        triplets = load_graph_triplets_from_json(input_file)
        logging.info(f"Loaded {len(triplets)} triplets.")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = NER_OUTPUT_DIR / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        execution_measures = {
            "start_time": datetime.datetime.now().isoformat(),
            "total_triplets": len(triplets),
            "input_file": str(input_file)
        }
    else:
        runs = sorted(list(NER_OUTPUT_DIR.glob("*")), reverse=True)
        if not runs:
            logging.error("No previous runs found.")
            return None
            
        run_dir = get_user_choice(runs, "Select run to resume:")
        progress_file = run_dir / "progress.json"
        
        if not progress_file.exists():
            logging.error("Progress file not found.")
            return None
            
        progress = load_json(progress_file)
        
        # Load processed nodes to allow safe resumption even if batch size changes
        if "processed_nodes" in progress:
            processed_nodes = set(progress["processed_nodes"])
        else:
            # Fallback for legacy runs: assume nodes with extracted entities are done.
            # Note: This might re-process nodes that yielded NO entities, but it's safer than skipping.
            saved_extractions = progress.get("results", {}).get("node_extractions", {})
            processed_nodes = set(saved_extractions.keys())
            
        execution_measures = progress.get("metrics", {})
        
        saved_counts = progress.get("results", {}).get("global_entity_counts", [])
        for item in saved_counts:
            global_entity_counts[(item["category"], item["entity"])] = item["frequency"]
        node_extractions = progress.get("results", {}).get("node_extractions", {})
        
        input_file_path = execution_measures.get("input_file")
        if input_file_path:
            triplets = load_graph_triplets_from_json(Path(input_file_path))
        else:
            logging.error("Missing input file in metadata.")
            return None

    # 2. Load Config & Prompt
    full_config = load_config()
    ner_config = full_config.get("ner", {})
    prompt_template = load_txt(PROMPT_PATH)
    
    # 3. Initialize LLM
    llm_client = get_llm_client(ner_config["model_settings"])
    run_settings = ner_config["run_settings"]
    
    # 4. Prepare Nodes (Semantic Batching)
    node_to_triplets = defaultdict(list)
    for idx, triplet in enumerate(triplets):
        node_to_triplets[triplet["subject"]].append(idx)
        node_to_triplets[triplet["object"]].append(idx)
    
    unique_nodes = list(node_to_triplets.keys())
    
    # Filter out already processed nodes
    nodes_to_process = [n for n in unique_nodes if n not in processed_nodes]
    
    if len(nodes_to_process) < len(unique_nodes):
        logging.info(f"Resuming: {len(processed_nodes)} nodes already processed. {len(nodes_to_process)} remaining.")

    # Sort/Cluster only the remaining nodes
    nodes_to_process = semantic_sort_nodes(
        run_settings.get("embedding_model", "all-MiniLM-L6-v2"),
        nodes_to_process,
        run_settings.get("llm_batch_size", 10),
        run_settings.get("embedding_batch_size", 32)
    )
    
    # 5. Processing Loop
    batch_size = run_settings.get("llm_batch_size", 10)
    total_batches = (len(nodes_to_process) + batch_size - 1) // batch_size
    
    with ExecutionMeasures(execution_measures, "ner") as em:
        for i in range(total_batches):
            batch = nodes_to_process[i * batch_size : (i + 1) * batch_size]
            logging.info(f"Processing batch {i+1}/{total_batches}")
            
            llm_output = process_batch(
                llm_client, batch, prompt_template,
                run_settings.get("llm_retries", {}).get("max_attempts", 3),
                run_settings.get("llm_retries", {}).get("delay_seconds", 1.0)
            )
            
            if llm_output and isinstance(llm_output, dict):
                # Reverse mapping: Category -> [Entities]  to  Node -> [(Cat, [Ents])]
                temp_node_map = defaultdict(lambda: defaultdict(set))
                
                for category, entities in llm_output.items():
                    if not isinstance(entities, list): continue
                    for entity in entities:
                        if not isinstance(entity, str) or not entity.strip(): continue
                        
                        # Regex match to assign entity to node
                        escaped = re.escape(entity.strip())
                        pattern = re.compile(r'(?<!\w)' + escaped + r'(?!\w)', re.IGNORECASE)
                        
                        for node in batch:
                            if pattern.search(node):
                                temp_node_map[node][category].add(entity.strip())
                
                # Update stats
                for node, cats in temp_node_map.items():
                    # Save mapping for graph reconstruction
                    node_extractions[node] = {c: list(e) for c, e in cats.items()}
                    
                    # Calculate frequency based on node usage in triplets
                    freq_weight = len(node_to_triplets.get(node, []))
                    for cat, ents in cats.items():
                        for ent in ents:
                            global_entity_counts[(cat, ent)] += freq_weight

            # Update processed list
            processed_nodes.update(batch)

            # Save Progress
            serializable_counts = [
                {"category": k[0], "entity": k[1], "frequency": v} 
                for k, v in global_entity_counts.items()
            ]
            em.add_llm_stats(llm_client.get_stats())
            save_progress(run_dir, processed_nodes, 
                          {"global_entity_counts": serializable_counts, "node_extractions": node_extractions}, 
                          execution_measures)

    # Calculate final total execution time across all sessions
    ExecutionMeasures.calculate_final_times(execution_measures)

    # 6. Final Save
    output_csv = run_dir / "entities.csv"
    
    sorted_entities = sorted(global_entity_counts.items(), key=lambda x: (x[0][0], -x[1]))
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Category", "Entity", "Frequency"])
        for (cat, entity), freq in sorted_entities:
            writer.writerow([cat, entity, freq])

    save_json(node_extractions, run_dir / "node_extractions.json")
    
    summary_data = {
        "execution_measures": execution_measures,
        "settings": ner_config,
    }
    save_json(summary_data, run_dir / "summary.json")
    
    logging.info(f"Extraction complete. Results saved to {run_dir}")
    return run_dir

# --- NORMALIZATION STAGE ---

def load_raw_entities(csv_path: Path) -> List[Dict[str, Any]]:
    """Loads entities.csv: Category, Entity, Frequency."""
    entities = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            row['Frequency'] = int(row['Frequency'])
            entities.append(row)
    return entities

def run_normalization_stage(run_dir: Optional[Path] = None) -> None:
    """Runs the normalization stage."""
    print("\n--- Step 2: NER Normalization ---")
    
    # 1. Select Run Directory if not provided
    if not run_dir:
        csv_files = sorted(list(NER_OUTPUT_DIR.rglob("*.csv")))
        if not csv_files:
            logging.error(f"No entities.csv found in {NER_OUTPUT_DIR}")
            return
        input_file = get_user_choice(csv_files, "Select input entities file:")
        run_dir = input_file.parent
    else:
        input_file = run_dir / "entities.csv"
        if not input_file.exists():
            logging.error(f"entities.csv not found in {run_dir}")
            return

    # 2. Load Config & Data
    full_config = load_config()
    ner_config = full_config.get("ner", {})
    norm_config = ner_config.get("normalization", {})
    
    # Load rules from new keys
    normalize_rules = norm_config.get("normalize_entities", {})
    remove_entities_list = norm_config.get("remove_entities", [])
    change_entities_rules = norm_config.get("change_entities", {})
    allowed_categories = set(ner_config.get("categories", []))
    
    entities_rows = load_raw_entities(input_file)
    logging.info(f"Loaded {len(entities_rows)} raw entities.")
    unique_before = len(set(row['Entity'] for row in entities_rows))

    # 3. Prepare Mapping and Sets
    variant_to_canonical = {}
    for canonical, variants in normalize_rules.items():
        variant_to_canonical[canonical.lower()] = canonical
        for v in variants:
            variant_to_canonical[v.lower()] = canonical
            
    remove_entities_set = {e.lower() for e in remove_entities_list}

    # Category-specific 1-to-N mapping
    category_map = defaultdict(list)
    for cat, rules in change_entities_rules.items():
        for canonical_name, variants in rules.items():
            for v in variants:
                category_map[(cat, v.lower())].append(canonical_name)

    # 4. Process, Filter and Aggregate
    aggregated_data = defaultdict(lambda: defaultdict(int))
    normalized_count = 0
    removed_count = 0
    absorbed_count = 0
    cat_filter_count = 0
    
    for row in entities_rows:
        original = row['Entity']
        category = row['Category']
        freq = row['Frequency']

        # Step A: Normalize
        canonical = variant_to_canonical.get(original.lower(), original).strip()
        if canonical != original:
            normalized_count += 1

        # Step B: Filter by Category (Whitelist)
        if allowed_categories and category not in allowed_categories:
            cat_filter_count += 1
            continue

        # Step C: Global Removal
        if canonical.lower() in remove_entities_set:
            removed_count += 1
            continue

        # Step D: Category-specific absorption (1-to-N)
        targets = category_map.get((category, canonical.lower()))
        if targets:
            absorbed_count += 1
            for target in targets:
                aggregated_data[target][category] += freq
        else:
            # No specific rule, keep the canonical name
            aggregated_data[canonical][category] += freq

    unique_after = len(aggregated_data)
    logging.info(f"Normalization stats: {normalized_count} renamed, {removed_count} removed, {absorbed_count} absorbed (1-to-N), {cat_filter_count} filtered by category.")
    logging.info(f"Vocabulary reduced from {unique_before} to {unique_after}.")

    # 5. Save Normalized Output
    output_path = run_dir / "entities_normalized.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Category', 'Entity', 'Frequency'])
        
        # Sort by Category then Frequency
        flat_rows = sorted(
            [(cat, entity, freq) for entity, cat_data in aggregated_data.items() for cat, freq in cat_data.items()],
            key=lambda x: (x[0], -x[2])
        )
        writer.writerows(flat_rows)
            
    logging.info(f"Saved normalized entities to {output_path}")

# --- MAIN ENTRY POINT ---

def main() -> None:
    print("      NER PIPELINE")
    
    options = ["Extract NER (LLM)", "Normalize NER (Manual Rules)", "Full Pipeline (Extract + Normalize)"]
    choice = get_user_choice(options, "Select action:")
    
    if choice == options[0]:
        run_extraction_stage()
        
    elif choice == options[1]:
        run_normalization_stage()
        
    elif choice == options[2]:
        run_dir = run_extraction_stage()
        if run_dir:
            run_normalization_stage(run_dir)

if __name__ == "__main__":
    main()
