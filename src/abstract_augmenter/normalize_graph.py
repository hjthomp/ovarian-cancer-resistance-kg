"""
Normalizes augmented Knowledge Graphs by standardizing node names globally.
Uses semantic clustering (embeddings) and LLM-based mapping to ensure 
consistency across clusters and newly added abstract data.
"""

import copy
import datetime
import json
import logging
import time
from pathlib import Path
from string import Template
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

from ..abstract_augmenter.augment_graph import select_graph, show_metadata_and_confirm, calculate_statistics
from ..utils.io import load_json, load_txt, save_json, load_yaml
from ..utils.constants import CONFIG_DIR, PROMPT_DIR, ABSTRACT_AUGMENTER_DIR, SUFFIX_NORMALIZED_KG
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import get_llm_client

def create_dynamic_batches(node_list: List[str], prompt_template: str, max_batch_size: int, num_ctx: int) -> List[List[str]]:
    """
    Generates batches ensuring that the token limit and batch size are not exceeded.
    Uses a heuristic of 1 token ~= 2.5 characters (conservative).
    Reserves 50% of context for the LLM response.
    """
    safe_input_token_limit = num_ctx * 0.5 
    max_input_chars = int(safe_input_token_limit * 2.5)
    
    template_overhead = len(prompt_template)
    
    batches = []
    current_batch = []
    current_chars = template_overhead
    
    logging.info(f"Batching strategy: Max Batch Size={max_batch_size}, Context={num_ctx}, Safe Input Limit={max_input_chars} chars")

    for node in node_list:
        # Estimate entry length: "- '{node}'\n"
        node_entry_len = len(node) + 4 
        
        if current_batch and ((current_chars + node_entry_len > max_input_chars) or (len(current_batch) >= max_batch_size)):
            batches.append(current_batch)
            current_batch = [node]
            current_chars = template_overhead + node_entry_len
        else:
            current_batch.append(node)
            current_chars += node_entry_len
            
    if current_batch:
        batches.append(current_batch)
        
    return batches

def get_all_unique_nodes(graph_data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """
    Extracts all unique node names (subjects and objects) from the entire graph.
    """
    unique_nodes = set()
    for triplets in graph_data.values():
        for t in triplets:
            unique_nodes.add(t["subject"])
            unique_nodes.add(t["object"])
    
    node_list = list(unique_nodes)
    logging.info(f"Found {len(node_list)} unique nodes in the entire graph.")
    return node_list

def sort_nodes_semantically(nodes: List[str], model_name: str) -> List[str]:
    """
    Sorts nodes based on semantic similarity using embeddings and clustering.
    Falls back to token-based sorting if libraries are missing.
    """
    try:
        
        logging.info(f"Using Embeddings ({model_name}) for semantic sorting...")
        # Load a lightweight, fast model
        model = SentenceTransformer(model_name)
        embeddings = model.encode(nodes, show_progress_bar=True)
        
        # Normalize embeddings to use Euclidean distance as Cosine distance equivalent
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Cluster nodes that are semantically close
        # distance_threshold=1.5 is roughly equivalent to a cosine similarity check
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.5, metric='euclidean', linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Sort nodes primarily by their cluster label, ensuring similar nodes stay together
        scored_nodes = sorted(zip(labels, nodes), key=lambda x: (x[0], x[1]))
        return [n for _, n in scored_nodes]

    except ImportError:
        logging.warning("Libraries 'sentence-transformers' or 'scikit-learn' not found.")
        logging.info("Falling back to Token-Based Sorting (better than pure alphabetical).")
        # Token sort: "Treatment Cisplatin" and "Cisplatin Treatment" become adjacent
        return sorted(nodes, key=lambda x: " ".join(sorted(x.lower().split())))
    
    except Exception as e:
        logging.error(f"Error in semantic sorting: {e}. Falling back to alphabetical.")

    return sorted(nodes)

def merge_llm_stats(stats1: Dict[str, Any], stats2: Dict[str, Any]) -> Dict[str, Any]:
    """Merges two LLM stats dictionaries by summing their values."""
    merged = stats1.copy()
    for k, v in stats2.items():
        merged[k] = merged.get(k, 0) + v
    return merged

def get_mapping_from_llm(
    node_list: List[str], 
    llm_client: Any, 
    prompt_template: str, 
    batch_size: int, 
    num_ctx: int, 
    retries: int = 3, 
    delay: float = 10.0, 
    state_path: Optional[Path] = None, 
    previous_elapsed_time: float = 0.0, 
    previous_llm_stats: Optional[Dict[str, Any]] = None, 
    session_start_time: Optional[float] = None, 
    embedding_model_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Uses an LLM to get a normalization map for a list of node names.
    Supports resuming from a state file if state_path is provided.
    Updates execution statistics (time and tokens) in the state file incrementally.

    Args:
        node_list: A list of unique node names from the graph.
        llm_client: An instance of the LLM client (OllamaClient).
        prompt_template: The string template for the prompt.
        batch_size: The number of nodes to process in each batch.
        num_ctx: The context window size of the model.
        state_path: Path to the state file for saving partial progress.
        previous_elapsed_time: Time accumulated in previous runs.
        previous_llm_stats: LLM stats accumulated in previous runs.
        session_start_time: Timestamp when the current session started.
        embedding_model_name: Name of the embedding model used (to save in state).

    Returns:
        A dictionary mapping each original node name to its new canonical name.
        Example: {"Cisplatino": "Cisplatin", "cisplatin": "Cisplatin", "EGFR": "EGFR"}
    """
    final_reverse_map = {}
    if previous_llm_stats is None: previous_llm_stats = {}

    # 1. Try to load existing mapping from state to resume progress
    if state_path and state_path.exists():
        try:
            data = load_json(state_path)
            if "partial_node_mapping" in data:
                final_reverse_map = data["partial_node_mapping"]
                logging.info(f"Resuming: Loaded {len(final_reverse_map)} cached mappings from state.")
        except Exception as e:
            logging.warning(f"Could not load partial state from {state_path}: {e}")

    # 2. Filter out nodes that are already mapped
    nodes_to_process = [n for n in node_list if n not in final_reverse_map]
    
    if not nodes_to_process:
        logging.info("All nodes are already mapped. Skipping LLM calls.")
        return final_reverse_map

    node_batches = create_dynamic_batches(nodes_to_process, prompt_template, batch_size, num_ctx)
    logging.info(f"Requesting normalization for {len(nodes_to_process)} nodes (out of {len(node_list)}) from LLM in {len(node_batches)} batches...")

    for i, batch in enumerate(node_batches):
        logging.info(f"Processing batch {i+1}/{len(node_batches)}...")
        # 1. Construct the prompt for the current batch
        node_list_str = ""
        for node in batch:
            node_list_str += f"- '{node}'\n"
        template = Template(prompt_template)
        prompt = template.substitute(node_list=node_list_str)
        
        # Estimate tokens (approx 4 chars per token)
        est_tokens = int(len(prompt) / 4)
        logging.info(f"Sending batch {i+1} with {len(batch)} nodes. Prompt length: {len(prompt)} chars (~{est_tokens} tokens)")
        
        # 2. Call the LLM
        response = llm_client.generate(prompt, retries=retries, delay=delay)

        # 3. Parse the LLM's JSON response for the current batch
        try:
            llm_output = response
            parsed_json = json.loads(llm_output)
            
            # Create and merge the reverse map: {variant: canonical_name}
            # The prompt asks for a dict {canonical: [variants]}
            for canonical, variants in parsed_json.items():
                for variant in variants:
                    # Ensure the variant from the original list is mapped
                    if variant in batch:
                        final_reverse_map[variant] = canonical

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON for batch {i+1}: {e}")
            # Fallback for this batch: map each node to itself.
            for node in batch:
                final_reverse_map[node] = node
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM normalization for batch {i+1}: {e}")
            for node in batch:
                final_reverse_map[node] = node
        
        # Save partial state after each batch
        if state_path:
            try:
                current_state = {}
                if state_path.exists():
                    current_state = load_json(state_path)
                current_state["partial_node_mapping"] = final_reverse_map
                
                # Update statistics
                current_duration = previous_elapsed_time
                if session_start_time:
                    current_duration += (time.time() - session_start_time)
                
                current_state["elapsed_time_seconds"] = current_duration
                current_state["llm_stats"] = merge_llm_stats(previous_llm_stats, llm_client.get_stats())
                if embedding_model_name:
                    current_state["embedding_model"] = embedding_model_name
                
                save_json(current_state, state_path)
            except Exception as e:
                logging.warning(f"Failed to save partial state: {e}")

    return final_reverse_map

def recalculate_graph_statistics(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Recalcula las estadísticas de un grafo normalizado."""
    logging.info("Calculating graph statistics...")
    
    stats = {
        "total_triplets": 0,
        "total_articles_triplets": 0,
        "total_abstracts_triplets": 0,
    }
    all_nodes = set()
    article_nodes = set()
    abstract_nodes = set()
    
    for triplets in graph.values():
        stats["total_triplets"] += len(triplets)
        for triplet in triplets:
            subject, obj = triplet["subject"], triplet["object"]
            all_nodes.add(subject)
            all_nodes.add(obj)
            
            if triplet.get("source_type") == 'article':
                stats["total_articles_triplets"] += 1
                article_nodes.add(subject)
                article_nodes.add(obj)
            elif triplet.get("source_type") == 'abstract':
                stats["total_abstracts_triplets"] += 1
                abstract_nodes.add(subject)
                abstract_nodes.add(obj)

    stats["total_nodes"] = len(all_nodes)
    stats["total_article_nodes"] = len(article_nodes)
    stats["total_abstract_nodes"] = len(abstract_nodes)

    logging.info("Graph statistics recalculated successfully.")
    return stats

def normalize_graph_globally(
    run_id: str, 
    input_kg: Dict[str, Any], 
    config: Dict[str, Any], 
    state_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Normalizes the nodes in the graph GLOBALLY using Semantic Sorting.
    It collects all nodes, groups them by meaning (Embeddings), creates a master mapping via LLM,
    and then applies it to all clusters.

    Args:
        graph_data: The graph data, expected to have a "graph" key where each
                    value is a list of triplets.
        normalization_config: Configuration dictionary for the normalization process.
        state_path: Path to the state file for resuming and saving progress.

    Returns:
        A new graph data dictionary with nodes normalized.
    """
    logging.info("Starting GLOBAL graph normalization process...")

    previous_elapsed_time = 0
    previous_llm_stats = {}

    # --- Load settings from config ---
    run_settings = config.get('run_settings', {})
    model_settings = config.get('model_settings', {})
    batch_size = run_settings.get('batch_size', 400)

    # Load Prompt Template
    prompt_path = PROMPT_DIR / "abstract_augmenter/normalize_graph.txt" 
    prompt_template = load_txt(prompt_path)

    # Load model client
    llm_client = get_llm_client(model_settings)
    num_ctx = model_settings.get('num_ctx', 4096) # Used to normalize

    # Use deepcopy to ensure that the original graph is not modified
    normalized_kg = copy.deepcopy(input_kg)
    normalized_graph = normalized_kg["graph"]

    total_original_nodes = set()
    total_normalized_nodes = set()

    if state_path and state_path.exists():
        print(f"\nFound a previous state for this run at {state_path}.")
        resume = input("Do you want to resume the previous normalization session? (y/n): ").strip().lower()
        if resume == 'y':
            logging.info("Loading previous state...")
            checkpoint_data = load_json(state_path)
            
            # Check embedding model consistency
            saved_model = checkpoint_data.get("embedding_model")
            if saved_model and saved_model != config.get("embedding_model"):
                logging.warning(f"State file uses embedding model '{saved_model}', but config specifies '{config.get('embedding_model')}'. Switching to '{saved_model}' for consistency.")
                config["embedding_model"] = saved_model

            # Check if we have a full graph checkpoint or just partial mappings
            if "graph" in checkpoint_data:
                normalized_graph = checkpoint_data["graph"]
                normalized_kg["graph"] = normalized_graph
                total_original_nodes = set(checkpoint_data.get("total_original_nodes", []))
                total_normalized_nodes = set(checkpoint_data.get("total_normalized_nodes", []))
                previous_elapsed_time = checkpoint_data.get("elapsed_time_seconds", 0)
                previous_llm_stats = checkpoint_data.get("llm_stats", {})
            else:
                logging.info("State file contains partial data (likely mappings) but no graph checkpoint. Proceeding with original graph and cached mappings.")
        else:
            logging.info("Starting fresh. Previous checkpoint will be overwritten.")
            if state_path.exists():
                state_path.unlink()

    # We pass normalized_kg["metadata"] so it updates in place if execution_measures exists, or creates it
    if "execution_measures" not in normalized_kg.get("metadata", {}):
        if "metadata" not in normalized_kg: normalized_kg["metadata"] = {}
        normalized_kg["metadata"]["execution_measures"] = {}

    stage_name = "normalization_augmentation"
    if stage_name in normalized_kg["metadata"]["execution_measures"]:
        stage_name = f"{stage_name}_{run_id}"

    with ExecutionMeasures(normalized_kg["metadata"]["execution_measures"], stage_name, module_name="normalize_graph", initial_duration=previous_elapsed_time) as em:

        session_start_time = time.time()

        # 1. Collect ALL nodes from the graph
        logging.info("Collecting all unique nodes from the graph...")
        all_nodes = get_all_unique_nodes(normalized_graph)
        total_original_nodes.update(all_nodes)
        
        if not all_nodes:
            logging.warning("No nodes found in the graph to normalize.")
            return input_kg

        # 2. Sort nodes Semantically (Embeddings) or Lexically (Token Sort)
        sorted_nodes = sort_nodes_semantically(all_nodes, config["embedding_model"])

        # 3. Create the Master Mapping (Global Context)
        node_rename_map = get_mapping_from_llm(
            sorted_nodes, 
            llm_client, 
            prompt_template, 
            batch_size, 
            num_ctx, 
            run_settings["llm_retries"]["max_attempts"], 
            run_settings["llm_retries"]["delay_seconds"],
            state_path=state_path,
            previous_elapsed_time=previous_elapsed_time,
            previous_llm_stats=previous_llm_stats,
            session_start_time=session_start_time,
            embedding_model_name=config["embedding_model"]
        )
            
        # 4. Apply the mapping to ALL clusters
        logging.info("Applying normalization mapping to all clusters...")
        for cluster_name, triplets in normalized_graph.items():
            
            # --- Rebuild the triplets ---
            normalized_triplets = []
            for triplet in triplets:
                new_triplet = triplet.copy()            
                # Create new normalized triplet
                new_subject = node_rename_map.get(triplet["subject"], triplet["subject"])
                new_object = node_rename_map.get(triplet["object"], triplet["object"])
                new_triplet["subject"] = new_subject
                new_triplet["object"] = new_object
                
                # Check if this triplet already exists in the normalized list
                t_exists = False
                for t in normalized_triplets:
                    if (t["subject"] == new_triplet["subject"] and
                        t["relation"] == new_triplet["relation"] and
                        t["object"] == new_triplet["object"]):
                        t_exists = True
                        break
                if t_exists:
                    continue  # Skip adding this triple

                normalized_triplets.append(new_triplet)
                total_normalized_nodes.add(new_subject)
                total_normalized_nodes.add(new_object)
            
            # Update the graph with normalized triplets
            normalized_graph[cluster_name] = normalized_triplets
        
        final_llm_stats = merge_llm_stats(previous_llm_stats, llm_client.get_stats())
        em.add_llm_stats(final_llm_stats)

    # --- Create/Update Metadata ---
    new_metadata = input_kg.get("metadata", {}).copy()
    # Ensure execution_measures is copied over from the timer's operation
    if "execution_measures" in normalized_kg.get("metadata", {}):
        new_metadata["execution_measures"] = normalized_kg["metadata"]["execution_measures"]
    
    ExecutionMeasures.calculate_final_times(new_metadata["execution_measures"])

    kg_name = new_metadata["kg_name"]
    if not kg_name.endswith(f"_{SUFFIX_NORMALIZED_KG}"):
        kg_name += f"_{SUFFIX_NORMALIZED_KG}"
    new_metadata["kg_name"] = kg_name
    new_metadata["kg_creation_date"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    
    # --- Calculate statistics ---
    new_metadata["statistics"] = calculate_statistics(normalized_graph, new_metadata)

    # Make triplets_readable      
    triplets_readable = [
        f"{t['subject']} -> {t['relation']} -> {t['object']} [{topic}]"
        for topic, triplets in normalized_graph.items()
        for t in triplets
    ]
    new_metadata["triplets_readable"] = triplets_readable
    
    # --- Append current run ---    
    new_metadata["run_history"].append(
        {
            "run_id": run_id,
            "date": new_metadata["kg_creation_date"],
            "description": "Normalized graph",
        }
    )
    
    # Finalize normalization
    normalized_kg["graph"] = normalized_graph
    normalized_kg["metadata"] = new_metadata

    # Final save of state file
    if state_path:
        ckpt = {
            "graph": normalized_graph,
            "total_original_nodes": list(total_original_nodes),
            "total_normalized_nodes": list(total_normalized_nodes),
            "elapsed_time_seconds": new_metadata["execution_measures"]["normalization_augmentation"]["duration_seconds"],
            "llm_stats": final_llm_stats,
            "embedding_model": config["embedding_model"],
            "completed": True
        }
        save_json(ckpt, state_path)

    logging.info(f"--- Normalization Finished ---")
    logging.info("Total Triplets - original: {}, normalized: {}".format(
        input_kg["metadata"]["statistics"].get("graph_totals", {}).get("unique_triplets", input_kg["metadata"]["statistics"].get("total_triplets", "N/A")),
        normalized_kg["metadata"]["statistics"]["graph_totals"]["unique_triplets"]
    ))
    logging.info("Total Nodes - original: {}, normalized: {}".format(
        input_kg["metadata"]["statistics"].get("graph_totals", {}).get("unique_nodes", input_kg["metadata"]["statistics"].get("total_nodes", "N/A")),
        normalized_kg["metadata"]["statistics"]["graph_totals"]["unique_nodes"]
    ))
                
    # logging.info(f"Total original unique nodes: {len(total_original_nodes)}, Total normalized unique nodes: {len(total_normalized_nodes)}")    
    return normalized_kg

def normalize_completed_graph() -> None:
    """Main entry point to normalize a graph from a completed run."""

    print("\n--- Normalizing an Augmented Graph (Completed Run Session) ---")

    runs_dir = ABSTRACT_AUGMENTER_DIR
    graph_path = select_graph(runs_dir, require_completed=True)
    if not graph_path:
        print("No graph selected. Returning to main menu.")
        return
    
    # --- Load graph ---
    input_kg = load_json(graph_path)
    if not show_metadata_and_confirm(input_kg.get("metadata", {})):
        print("Operation cancelled. Returning to main menu.")
        return
    
    if not input_kg.get("graph"):
        logging.warning("Graph data is empty or missing 'graph' key. Returning as is.")
        return 

    # Load configuration to normalize graph    
    config_path = CONFIG_DIR / "abstract_augmenter/settings.yaml"
    config = load_yaml(config_path)["normalize_graph"]
    if not config:
        logging.error(f"Could not load file from '{config_path}'.")
        return
    

    run_id = graph_path.parent.name
    state_path = graph_path.parent / "processing_norm_state.json"
    # Normalize graph
    normalized_graph = normalize_graph_globally(run_id, input_kg, config, state_path)

    # Save graph
    print("\n--- Saving normalized graph ---")
    output_file = graph_path.parent / f"kg_{SUFFIX_NORMALIZED_KG}.json"
    save_json(normalized_graph, output_file)
    # Save the run configuration (just in case)
    save_json(config, output_file.parent / "run_settings_norm.json")
    print(f"Normalization complete. \nThe new graph is saved at: {output_file}")
    
    return
