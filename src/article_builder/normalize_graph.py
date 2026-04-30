"""
Normalizes Knowledge Graph clusters using LLMs.
Refines entity names, standardizes relations to snake_case, and consolidates 
evidence for each normalized triplet. Features batch processing per cluster 
and session resumption via temporal files.
"""

import datetime
from itertools import chain
import json
import logging
from string import Template
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple, Union

from ..utils.io import load_txt, load_yaml, load_json, save_json
from ..utils.constants import CONFIG_DIR, PROMPT_DIR, ARTICLE_BUILDER_DIR, SUFFIX_NORMALIZED_KG, SUFFIX_BASE_KG
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import BaseLLM, get_llm_client

def normalize_cluster_with_llm(
    llm_client: BaseLLM, 
    cluster_theme: str, 
    triplets: List[Dict[str, Any]], 
    prompt_template: str, 
    batch_size: int, 
    start_batch: int = 0, 
    retries: int = 3, 
    delay: float = 3.0
) -> Generator[Tuple[int, List[Dict[str, Any]]], None, None]:
    """Processes triplets within a cluster in batches and yields normalized results."""
    # Map each triplet with an internal id
    for i in range(start_batch, len(triplets), batch_size):
        batch = triplets[i:i+batch_size]
        input_texts = "\n".join([
            f"- Triplet {j}: [{t.get('subject')}] -> [{t.get('relation')}] -> [{t.get('object')}]"
            for j, t in enumerate(batch)
        ])
        template = Template(prompt_template)
        prompt = template.substitute(cluster_theme=cluster_theme, input_texts=input_texts)
        
        # Estimate tokens (approx 4 chars per token)
        est_tokens = int(len(prompt) / 4)
        logging.info(f"  Prompt size: {len(prompt)} chars (~{est_tokens} tokens)")

        # Call to LLM with retries
        response = llm_client.generate(prompt, retries=retries, delay=delay)

        try:
            llm_output = json.loads(response)
            # Reconstruct triplets with the originals
            for t in llm_output:
                # Convert relation to snake_case
                if 'relation' in t and isinstance(t['relation'], str):
                    t['relation'] = "_".join(t['relation'].strip().lower().split())

                ids = t.pop("ids", [])
                normalized_ids = []
                for t_id in ids:
                    try:
                        idx = int(t_id)
                        if 0 <= idx < len(batch):  # <-- Chequea rango
                            normalized_ids.append(idx)
                    except (ValueError, TypeError):
                        # Si es algo raro como "id_3" o None -> ignorar
                        logging.warning(f"Invalid evidence ID received from LLM: {t_id}")
                        continue

                # Save original triplet 
                t["evidence_list"] = []
                for idx in normalized_ids:
                    # Check evidence_text repeted in evidence_list
                    if any(e["evidence_text"] == batch[idx].get("evidence_text") for e in t["evidence_list"]):
                        continue
                    t["evidence_list"].append({
                        "evidence_doi": batch[idx].get("evidence_doi"),
                        "evidence_page": batch[idx].get("evidence_page"),
                        "evidence_text": batch[idx].get("evidence_text"),
                        "extracted_triplet": {
                            "subject": batch[idx].get("subject"),
                            "relation": batch[idx].get("relation"),
                            "object": batch[idx].get("object"),
                            "triplet_topic": batch[idx].get("triplet_topic"),
                        },
                    })
                # t["evidence_list"] = [batch[idx] for idx in normalized_ids]

            yield i // batch_size, llm_output # (batch_idx, batch_result)

        except json.JSONDecodeError:
            logging.error(f"JSON parsing error in cluster {cluster_theme}, batch {i//batch_size}. Skipping. Response: {response}")
            continue

def load_temporal(path: Path) -> Dict[str, Any]:
    """Loads progress from a temporal JSON file if it exists."""
    if path.exists():
        return load_json(path)

    return {
        "batch_processed": 0,
        "triplets_processed": []
    }   

def save_temporal(path: Path, progress: Dict[str, Any]) -> None:
    """Saves normalization progress to a temporal file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

def process_all_clusters(
    llm_client: BaseLLM, 
    clusters: Dict[str, List[Dict[str, Any]]], 
    prompt_template: str, 
    run_settings: Dict[str, Any], 
    output_clusters_dir: Path
) -> None:
    """Iterates through all graph clusters to apply LLM-based normalization."""
    total_clusters = len(clusters)
    output_dir = output_clusters_dir
    batch_size = run_settings["batch_size"]
    retries = run_settings["llm_retries"]["max_attempts"]
    delay = run_settings["llm_retries"]["delay_seconds"]

    for i, (cluster_label, triplets) in enumerate(clusters.items(), start=1):

        cluster_dir = output_dir / f"cluster_{i}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        temporal_file = cluster_dir / "temporal.json"
        temporal_progress = load_temporal(temporal_file)

        cluster_prog = temporal_progress.get("batch_processed", 0)
        
        num_batches = (len(triplets) + batch_size - 1) // batch_size
        
        logging.info(f"\n[{i}/{total_clusters}] Processing cluster '{cluster_label}' ({len(triplets)} triplets, {num_batches} batches)")
        
        for batch_idx, normalized in normalize_cluster_with_llm(
            llm_client, cluster_label, triplets, prompt_template,
            batch_size, start_batch=cluster_prog * batch_size, retries=retries, delay=delay
        ):
            temporal_progress["batch_processed"] = batch_idx + 1
            temporal_progress["triplets_processed"].append(normalized)
            
            # Save progress for this batch
            save_temporal(temporal_file, temporal_progress)
            
            logging.info(f"  Batch {batch_idx+1}/{num_batches} saved, progress updated.")
        
        
        # Save the processed cluster (previously saved in temporal_file) 
        data = load_json(temporal_file)
        cluster_result = { 
            cluster_label: data["triplets_processed"] 
        }

        cluster_file = cluster_dir / "norm.json"
        save_json(cluster_result, cluster_file)

        logging.info(f"✔ Cluster '{cluster_label}' completed and saved to {cluster_file}.\n")

def consolidate_clusters(find_files: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Gathers normalized triplets from all cluster subdirectories into a single graph."""
    final_graph = {}
    found_files = 0
    for cluster_dir in sorted(find_files.glob("cluster_*")):
        cluster_file = cluster_dir / "norm.json" # Assumes cluster_norm.json is directly inside cluster_dir
        if cluster_file.exists():
            found_files += 1
            data = load_json(cluster_file)
            for cluster_label, triplets in data.items():
                # Ensure triplets is iterable (list of dicts)
                if isinstance(triplets, list):
                    # final_graph.setdefault(cluster_label, []).extend(triplets[0])
                    final_graph.setdefault(cluster_label, []).extend(chain.from_iterable(triplets)) # Esto concatena todos los batches en una sola secuencia
                else:
                    logging.warning(f"{cluster_file} cluster {cluster_label} has unexpected format")   

    if found_files == 0:
        logging.warning(f"No 'norm.json' files found in directory {find_files}. The consolidated file will be empty.")

    return final_graph

def run_graph_normalizer(article_id: str, output_dir: Optional[Path] = None) -> None:
    """
    Executes the cluster normalization pipeline for a single article.
    """
    logging.info(f"Processing article: {article_id}")
    
    # Load configuration and prompt
    try:
        config_path = CONFIG_DIR / "article_builder/settings.yaml"
        config = load_yaml(config_path)["normalize_graph"]
        
        prompt_path = PROMPT_DIR / "article_builder/normalize_graph.txt"
        prompt_template = load_txt(prompt_path)
        logging.info(f"Configuration and prompt loaded.")
    except Exception:
        sys.exit(1)
    
    run_settings = config["run_settings"]
    model_settings = config["model_settings"]

    # 2. Define input and output paths
    base_root = output_dir if output_dir else ARTICLE_BUILDER_DIR
    base_output_dir = base_root / article_id
    input_graph_file = base_output_dir / f"kg_{SUFFIX_BASE_KG}.json" # "kg_base.json"
    output_graph_file = base_output_dir / f"kg_{SUFFIX_NORMALIZED_KG}.json" # "kg_normalized.json"

    # Output directory for the normalized clusters (will contain cluster_*/ subfolders)
    output_dir = base_output_dir / "normalize_graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_summary_file = output_dir / "summary.json"

    # 3. Initialize LLM client
    llm_client = get_llm_client(model_settings)

    # 4. Execute pipeline
    # Load graph
    data = load_json(input_graph_file)

    # Prepare final graph structure with inherited metadata
    final_kg = {
        "graph": {},
        "metadata": data.get("metadata", {}),
    }

    with ExecutionMeasures(final_kg["metadata"]["execution_measures"], "normalization", module_name="normalize_graph") as em:

        clusters = data.get("graph", {})
        logging.info(f"Loaded {len(clusters)} clusters from {input_graph_file}.")

        # Sort clusters by size (ascending) to process smaller ones first
        clusters_sorted = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=False))

        # Normalize each cluster
        process_all_clusters(llm_client, clusters_sorted, prompt_template, run_settings, output_dir)

        # Register LLM stats
        em.add_llm_stats(llm_client.get_stats())

            
    # Consolidate normalization
    final_kg["graph"] = consolidate_clusters(output_dir)
    final_kg["metadata"]["kg_name"] = article_id
    final_kg["metadata"]["kg_type"] = "normalized"
    final_kg["metadata"]["kg_creation_date"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Make triplets_readable
    triplets_readable = [
        f"{t['subject']} -> {t['relation']} -> {t['object']} [{cluster_label}]"
        for cluster_label, triplets in final_kg["graph"].items() 
        for t in triplets
    ]   
    final_kg["metadata"]["triplets_readable"] = triplets_readable
    
    # Calculate final times
    ExecutionMeasures.calculate_final_times(final_kg["metadata"]["execution_measures"])
    

    # Save graph
    save_json(final_kg, output_graph_file)
    logging.info(f"Consolidated normalization. Final graph saved to {output_graph_file}.")

    output_data = {
        "total_clusters_processed": len(clusters_sorted),
        "execution_measures": final_kg["metadata"]["execution_measures"],
        "settings": {
            "prompt": prompt_template,
            "model": model_settings,
            "run": run_settings,
        }
    }
    # Save summary
    save_json(output_data, output_summary_file)

if __name__ == '__main__':
    logging.warning("This script is not intended to be run directly. Please use 'main.py'.")