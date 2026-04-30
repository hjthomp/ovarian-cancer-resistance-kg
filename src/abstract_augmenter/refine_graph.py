"""
Refines Knowledge Graphs by applying declarative cleaning and normalization rules.
Supports node/relation canonicalization, removal of specific items, 
and global evidence deduplication to ensure graph consistency.
"""

import logging
import datetime
import json
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set, Union

from ..abstract_augmenter.augment_graph import select_graph, show_metadata_and_confirm, calculate_statistics, print_statistics
from ..utils.io import load_json, save_json, load_yaml
from ..utils.constants import CONFIG_DIR, ABSTRACT_AUGMENTER_DIR, SUFFIX_REFINED_KG

def clean_graph(data: Dict[str, Any], filters: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Cleans and refines a knowledge graph based on a set of declarative rules.

    It applies changes (canonicalization) and removals for nodes, relations,
    and triplets as defined in the 'refine_graph' section of the settings file.
    - Removals are case-sensitive for precise control.
    - Changes are case-insensitive for robust matching.
    
    See config/abstract_augmenter/settings.yaml for detailed configuration instructions.
    """
    filters = filters or {}

    # Safely get filter lists/dicts, defaulting to empty if a key is missing or its value is None (e.g., 'null' in YAML)
    remove_nodes = filters.get("remove_nodes") or []
    remove_relations = filters.get("remove_relations") or []
    remove_triplets = filters.get("remove_triplets") or []
    change_nodes = filters.get("change_nodes") or {}
    change_relations = filters.get("change_relations") or {}
    change_triplets = filters.get("change_triplets") or {}
    
    # Counters for logging
    stats = {
        "nodes_removed_by_filter": 0,
        "relations_removed_by_filter": 0,
        "triplets_removed_by_filter": 0,
        "nodes_changed": 0,
        "relations_changed": 0,
        "triplets_changed": 0,
        "triplets_merged": 0
    }
    # Revovals: Use case-sensitive matching for exact filtering
    remove_nodes_set = set(str(n).strip() for n in remove_nodes)
    remove_relations_set = set(str(r).strip() for r in remove_relations)

    # Parse remove_triplets strings (e.g., "s|r|o") into tuples for case-sensitive comparison
    remove_triplets_set = set()
    for t in remove_triplets:
        t_str = str(t).strip()
        # Support common separators like '|' or ';'
        parts = []
        if '|' in t_str:
            parts = t_str.split('|')
        elif ';' in t_str:
            parts = t_str.split(';')
        elif ',' in t_str:
            parts = t_str.split(',')
        
        if len(parts) == 3:
            remove_triplets_set.add(tuple(p.strip() for p in parts))

    # Changes (canonicalization): Use case-insensitive lookup for robustness
    # This allows mapping "Cisplatin" and "cisplatin" to the same canonical form, e.g., "cisplatin".
    # The final name will have the exact casing from the YAML key (canonical term).
    # Config format expected: { CanonicalName: [Variant1, Variant2, ...] }
    # Target format: { variant_lower: CanonicalName }
    node_lookup = {}
    for canonical, variants in change_nodes.items():
        if isinstance(variants, list):
            for v in variants:
                node_lookup[str(v).strip().lower()] = canonical
        elif isinstance(variants, str):
             node_lookup[str(variants).strip().lower()] = canonical

    relation_lookup = {}
    for canonical, variants in change_relations.items():
        if isinstance(variants, list):
            for v in variants:
                relation_lookup[str(v).strip().lower()] = canonical
        elif isinstance(variants, str):
            relation_lookup[str(variants).strip().lower()] = canonical

    # Build lookup for full triplet changes
    change_triplet_lookup = {}
    if change_triplets: # Check if it's not None or empty
        for canonical_triplet_str, variant_triplet_strs in change_triplets.items():
            canonical_parts = [p.strip() for p in canonical_triplet_str.split('|')]
            if len(canonical_parts) != 3: continue
            canonical_tuple = tuple(canonical_parts)

            variants_to_process = variant_triplet_strs if isinstance(variant_triplet_strs, list) else [variant_triplet_strs]

            for v_str in variants_to_process:
                variant_parts = [p.strip().lower() for p in str(v_str).split('|')]
                if len(variant_parts) != 3: continue
                variant_tuple_lower = tuple(variant_parts)
                change_triplet_lookup[variant_tuple_lower] = canonical_tuple

    # Use a global lookup to handle merging of duplicates across all clusters
    # key: (s_lower, r_lower, o_lower), value: triplet_dict_in_refined_graph
    global_deduplication_lookup = {}
    refined_graph = defaultdict(list)
    
    for cluster, triples in data["graph"].items():
        for t in triples:
            # Initialize with original values, which will be progressively updated.
            subject = t["subject"]
            relation = t["relation"]
            object_ = t["object"]

            # 1. Check for a full triplet change first (most specific rule)
            original_triplet_key = (subject.strip().lower(), relation.strip().lower(), object_.strip().lower())

            if original_triplet_key in change_triplet_lookup:
                new_s, new_r, new_o = change_triplet_lookup[original_triplet_key]
                if (new_s, new_r, new_o) != (subject, relation, object_):
                    stats["triplets_changed"] += 1
                subject = new_s
                relation = new_r
                object_ = new_o

            # 2. Apply node/relation CHANGES (Canonicalization) in cascade.
            # This runs *after* a potential triplet change, ensuring nodes are also normalized.
            intermediate_subject, intermediate_object, intermediate_relation = subject, object_, relation
            subject = node_lookup.get(intermediate_subject.strip().lower(), intermediate_subject)
            object_ = node_lookup.get(intermediate_object.strip().lower(), intermediate_object)
            relation = relation_lookup.get(intermediate_relation.strip().lower(), intermediate_relation)

            if subject != intermediate_subject or object_ != intermediate_object: stats["nodes_changed"] += 1
            if relation != intermediate_relation: stats["relations_changed"] += 1

            # 3. Apply REMOVALS (Filtering) on the potentially changed triplet.
            if subject.strip() in remove_nodes_set or object_.strip() in remove_nodes_set:
                stats["nodes_removed_by_filter"] += 1
                continue
            if relation.strip() in remove_relations_set:
                stats["relations_removed_by_filter"] += 1
                continue
            if (subject.strip(), relation.strip(), object_.strip()) in remove_triplets_set:
                stats["triplets_removed_by_filter"] += 1
                continue

            # 4. Global Deduplication and Merging (case-insensitive)
            key = (subject.strip().lower(), relation.strip().lower(), object_.strip().lower())
            if key in global_deduplication_lookup:
                stats["triplets_merged"] += 1
                # Global duplicate found, merge evidence into the existing one
                existing_triplet = global_deduplication_lookup[key]
                for evidence_item in t.get("evidence_list", []):
                    if evidence_item not in existing_triplet["evidence_list"]:
                        existing_triplet["evidence_list"].append(evidence_item)
                # Skip adding this triplet, as it's a duplicate of one in another cluster
                continue

            # 5. This is a new, unique triplet. Add it to the graph.
            if not t.get("evidence_list"):
                continue

            new_triplet = t.copy()
            new_triplet["subject"] = subject
            new_triplet["object"] = object_
            new_triplet["relation"] = relation
            
            # Add to the new graph under the CURRENT cluster
            refined_graph[cluster].append(new_triplet)

            # Add to the global lookup for future duplicates
            global_deduplication_lookup[key] = new_triplet

    # Convert defaultdict back to a regular dict for JSON serialization
    final_graph = {k: v for k, v in refined_graph.items() if v}

    data["graph"] = final_graph
    
    logging.info(f"Refinement stats: {json.dumps(stats, indent=2)}")

    return data, stats

def refine_graph() -> None:
    """Interactive tool to select a graph and apply refinement filters based on configuration."""
    print("\n--- Refine a Completed Graph ---")

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

    # Load configuration to refine graph    
    config_path = CONFIG_DIR / "abstract_augmenter/settings.yaml"
    config = load_yaml(config_path)["refine_graph"]
    if not config:
        logging.error(f"Could not load run_settings from '{config_path}'.")
        return
    
    # Refine graph
    kg_source = load_json(graph_path)
    filters = config.get("run_settings")
    kg_clean, refinement_stats = clean_graph(
        kg_source, 
        filters
    )

    # --- Update Metadata ---
    metadata = kg_clean.get("metadata", {})
    
    # Cleanup from previous stages: 'triplets_readable' and 'embedding_model'
    # Remove embedding_model if present 
    if "embedding_model" in metadata:
        del metadata["embedding_model"]
    # Remove triplets_readable if present 
    if "triplets_readable" in metadata:
        del metadata["triplets_readable"]

    # Update name
    if not metadata.get("kg_name", "").endswith(f"_{SUFFIX_REFINED_KG}"):
        metadata["kg_name"] = f"{metadata.get('kg_name', 'kg')}_{SUFFIX_REFINED_KG}"
    
    metadata["kg_creation_date"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Recalculate statistics
    metadata["statistics"] = calculate_statistics(kg_clean["graph"], metadata)
    print_statistics(metadata["statistics"])

    # Update run history
    if "run_history" not in metadata:
        metadata["run_history"] = []
    
    metadata["run_history"].append({
        "date": metadata["kg_creation_date"],
        "description": "Refined graph"
    })

    kg_clean["metadata"] = metadata

    # Save graph
    print("\n--- Saving cleaned graph ---")
    output_file = graph_path.parent / f"kg_{SUFFIX_REFINED_KG}.json"

    save_json(kg_clean, output_file)

    # Save the run configuration and stats separately
    refinement_run_data = {
        "refinement_config": config,
        "refinement_stats": refinement_stats,
        "date": metadata["kg_creation_date"]
    }
    save_json(refinement_run_data, output_file.parent / "run_settings_refinement.json")
    print(f"Refining is complete. \nThe refined graph is saved at: {output_file}")
    
    return