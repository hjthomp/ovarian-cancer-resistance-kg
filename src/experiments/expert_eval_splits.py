"""
Generates stratified evaluation splits (CSVs) for expert quality assessment.
Uses a hybrid approach (Core/Top-N vs. Tail/Random) to create unique and overlapping
sets for Inter-Annotator Agreement (IAA) and source verification.
"""

import csv
import collections
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Settings
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parents[1]
RELEASES_DIR = PROJECT_ROOT / "data" / "releases"
OUTPUT_DIR = PROJECT_ROOT / "data" / "experiments" / "expert_evaluation" / "splits"

# Default values
TOP_N_NODES = 2
DEFAULT_HOPS = 2
SPLIT = 3
DEFAULT_SAMPLES = 200
DEFAULT_VERIFICATION_PERCENT = 10
DEFAULT_OVERLAP_PERCENT = 20
DEFAULT_CORE_PERCENT = 80

def load_graph(path: Path) -> Dict[str, Any]:
    """Loads a JSON knowledge graph file."""
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    print(f"Loading graph from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Preprocessing (simple but could scale)
def normalize(text: str) -> str:
    return text.strip().lower()

# Graph indexing 
def build_adjacency_index(graph: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build an adjacency index:
    normalized_node -> list of incident relations (in/out)
    """
    index: Dict[str, List[Dict[str, Any]]] = {}

    for theme, triplets in graph.items():
        for t in triplets:
            subj = t.get("subject", "")
            obj = t.get("object", "")
            rel = t.get("relation")

            subj_norm = normalize(subj)
            obj_norm = normalize(obj) 
            
            item = {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "theme": theme,
                "evidence": t.get("evidence_list", []),
                "source_type": t.get("source_type", ""),
            }
            index.setdefault(subj_norm, []).append(
                {**item, "direction": "out"}
            )
            index.setdefault(obj_norm, []).append(
                {**item, "direction": "in"}
            )

    return index

# Neighborhood extraction
def extract_n_hop_neighborhood(seed_node: str, index: Dict[str, List[Dict[str, Any]]], max_hops: int) -> Dict[str, Any]:
    """Extract N-hop neighborhood around a node using BFS."""
    
    frontier = {normalize(seed_node)}
    visited = {normalize(seed_node)}
    seen_edges = set()
    relations = []

    for hop in range(1, max_hops + 1):
        next_frontier = set()

        for current_node_norm in frontier:
            for edge in index.get(current_node_norm, []):
                edge_sig = (
                    edge["subject"],
                    edge["relation"],
                    edge["object"],
                    edge["theme"]
                )

                if edge_sig in seen_edges:
                    continue
                seen_edges.add(edge_sig)
                
                relations.append({
                    "subject": edge["subject"],
                    "relation": edge["relation"],
                    "object": edge["object"],
                    "theme": edge["theme"],
                    "evidence": edge["evidence"],
                    "source_type": edge["source_type"],
                    "direction": edge["direction"],
                    "hop": hop,
                })
                
                other = normalize(
                    edge["object"]
                    if normalize(edge["subject"]) == current_node_norm
                    else edge["subject"]
                )

                if other not in visited:
                    visited.add(other)
                    next_frontier.add(other)

        frontier = next_frontier

    return {
        "seed_node": seed_node,
        "max_hops": max_hops,
        "num_relations": len(relations),
        "relations": relations
    }

# Saving results
def save_results(
    relations_pool: List[Dict[str, Any]], 
    registry: Dict[int, Dict[str, Any]], 
    output_dir: Path, 
    file_prefix: str, 
    num_splits: int = SPLIT, 
    num_unique_per_expert: int = 0, 
    num_overlap: int = 0, 
    verification_percent: float = 10.0
) -> Dict[str, int]:
    """Distributes triplets among experts and saves them to CSV files."""

    if not relations_pool:
        print("No relations to save.")
        return {}
    
    split_stats = {}

    globally_assigned_checks = set()
    
    # Headers as requested
    csv_headers = [
        "ID",
        "DOI",
        "Location",
        "Evidence",
        "Triplet",
        "Coherent? (YES/NO)",
        "Plausible? (YES/NO/UNKNOWN)",
        "Source Verification (OK/ERROR)",
        "Notes"
    ]

    # Helper to write chunks
    def _write_csv(rows: List[Dict[str, Any]], filename: str) -> None:
        if not rows: return
        csv_path = output_dir / filename
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        split_stats[csv_path.name] = len(rows)
        
    # 2. Create Overlap and Unique Sets
    random.shuffle(relations_pool) # Shuffle FIRST so shared items are random
    shared_relations = relations_pool[:num_overlap]
    unique_relations = relations_pool[num_overlap:]

    # Update registry with shared status AFTER selection
    for r in shared_relations:
        registry[r['id']]["is_shared"] = True

    print(f"\nSplitting logic:")
    print(f"  - Shared set for all experts: {len(shared_relations)} triplets")
    print(f"  - Unique triplets available for distribution: {len(unique_relations)} triplets")

    # 3. Distribute to experts
    for i in range(num_splits):
        start_idx = i * num_unique_per_expert
        end_idx = start_idx + num_unique_per_expert
        
        if start_idx >= len(unique_relations):
            print(f"Warning: Not enough unique relations to create split #{i+1}. Stopping.")
            break

        expert_unique_slice = unique_relations[start_idx:end_idx]
        
        # Combine unique slice with shared set
        expert_final_relations = expert_unique_slice + shared_relations
        random.shuffle(expert_final_relations) # IMPORTANT: Shuffle to mix shared and unique items

        # --- Stratified selection for Source Verification ---
        num_checks_needed = int(len(expert_final_relations) * (verification_percent / 100))
        expert_verification_ids = set()

        # Only consider relations that haven't been marked as CHECK for another expert
        eligible_for_check = [r for r in expert_final_relations if r['id'] not in globally_assigned_checks]

        if num_checks_needed > 0 and eligible_for_check:
            # Separate relations by source type, ensuring no overlap
            from_articles = []
            from_abstracts = []
            
            for r in eligible_for_check:
                source_type = r.get("source_type", "").lower()
                if "article" in source_type:
                    from_articles.append(r)
                elif "abstract" in source_type: # Use elif to make them exclusive
                    from_abstracts.append(r)

            quota_articles = num_checks_needed // 2
            quota_abstracts = num_checks_needed - quota_articles
            
            # Sample from articles
            sampled_articles = random.sample(from_articles, min(len(from_articles), quota_articles))
            expert_verification_ids.update(r['id'] for r in sampled_articles)
            
            # Adjust quota for abstracts if articles were insufficient
            remaining_checks = num_checks_needed - len(expert_verification_ids)
            
            # Sample from abstracts
            sampled_abstracts = random.sample(from_abstracts, min(len(from_abstracts), remaining_checks))
            expert_verification_ids.update(r['id'] for r in sampled_abstracts)

            # If we still haven't met the quota, fill from the rest of the pool
            if len(expert_verification_ids) < num_checks_needed:
                remaining_pool = [r for r in eligible_for_check if r['id'] not in expert_verification_ids]
                fill_samples = random.sample(remaining_pool, min(len(remaining_pool), num_checks_needed - len(expert_verification_ids)))
                expert_verification_ids.update(r['id'] for r in fill_samples)

        # Update global tracker
        globally_assigned_checks.update(expert_verification_ids)

        # 4. Format rows for this specific expert's file
        expert_rows = []
        for j, r in enumerate(expert_final_relations):
            # Pick only ONE evidence
            evidence_list = r.get("evidence", [])
            selected_ev = evidence_list[0] if evidence_list else {}
            
            # Extract fields
            ev_text = selected_ev.get("evidence_text", "")
            doi = selected_ev.get("evidence_doi", "")
            
            # Format Location
            page = selected_ev.get("evidence_page", "")
            location = f"Page {page}" if page else ("Abstract" if "abstract" in str(r.get("source_type", "")).lower() else "Full Text")

            # Format Triplet
            triplet_str = f"{r['subject']} → {r['relation']} → {r['object']}"

            row = {
                "ID": r['id'],
                "DOI": doi,
                "Location": location,
                "Evidence": ev_text,
                "Triplet": triplet_str,
                "Coherent? (YES/NO)": "",
                "Plausible? (YES/NO/UNKNOWN)": "",
                # Apply audit check based on the row number *within this file*
                "Source Verification (OK/ERROR)": "CHECK" if r['id'] in expert_verification_ids else "",
                "Notes": ""
            }
            expert_rows.append(row)
            
            # Update registry assignments
            registry[r['id']]["assigned_to"].append(f"{i+1}")

        # 5. Write this expert's file
        filename = f"{file_prefix}{i+1}.csv"
        _write_csv(expert_rows, filename)
        print(f"  - Created file '{filename}' with {len(expert_rows)} rows.")
            
    return split_stats

def save_metadata(
    exp_name: str,
    description: str,
    output_dir: Path,
    graph_path: Path,
    seed_nodes: List[str],
    top_n_value: int,
    random_seed_nodes: str,
    max_hops: int,
    total_samples: int,
    samples_selected_core: int,
    samples_selected_tail: int,
    pool_size_core: int,
    pool_size_tail: int,
    samples_per_expert: int,
    overlap_percent: float,
    verification_percent: float,
    split_stats: Dict[str, int] = None,
) -> None:
    """Saves experiment configuration and execution metadata for reproducibility."""
    metadata = {
        "experiment_name": exp_name,
        "description": description,
        "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "graph_path": str(graph_path.relative_to(PROJECT_ROOT)),
        "parameters": {
            "n_per_theme": top_n_value,
            "top_n_nodes": seed_nodes,
            "random_nodes": random_seed_nodes,
            "max_hops": max_hops,
            "total_samples_generated": total_samples,
            "sample_distribution": {
                "selected_core": samples_selected_core,
                "selected_tail": samples_selected_tail,
                "available_pool_core": pool_size_core,
                "available_pool_tail": pool_size_tail
            },
            "samples_per_expert": samples_per_expert,
            "overlap_percentage": overlap_percent,
            "verification_percentage": verification_percent,
        },
        "generated_files": split_stats,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def select_graph_from_releases() -> Optional[Path]:
    """Lists available graphs in the releases folder and prompts the user to select one."""
    print("--- Select a graph to analyze ---")
    graph_files = sorted([f for f in RELEASES_DIR.glob("*.json") if f.is_file()])
    if not graph_files:
        print(f"No graphs found in {RELEASES_DIR}")
        return None

    for i, path in enumerate(graph_files):
        print(f"[{i+1}] {path.name}")

    while True:
        try:
            choice = int(input(f"Enter number (1-{len(graph_files)}): ")) - 1
            if 0 <= choice < len(graph_files):
                return graph_files[choice]
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")

def get_top_nodes_per_theme(graph: Dict[str, List[Dict[str, Any]]], top_n: int) -> Dict[str, List[str]]:
    """
    Finds the Top N most connected nodes (by degree) within each theme/cluster.
    Returns a dictionary mapping each unique seed node to a list of themes it was a top node in.
    """
    print(f"Calculating top {top_n} nodes per theme...")
    seed_to_themes = collections.defaultdict(list)

    for theme, triplets in graph.items():
        degrees = collections.Counter()
        for t in triplets:
            if subj := t.get("subject"):
                degrees[subj] += 1
            if obj := t.get("object"):
                degrees[obj] += 1
        
        top_nodes = [node for node, count in degrees.most_common(top_n)]
        for node in top_nodes:
            seed_to_themes[node].append(theme)
        print(f"  Theme '{theme}': {top_nodes}")
    
    print(f"Total unique seed nodes identified across all themes: {len(seed_to_themes)}")
    return seed_to_themes

def extract_all_relations(graph: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten the graph into a list of relation dictionaries."""
    all_rels = []
    for theme, triplets in graph.items():
        for t in triplets:
            all_rels.append({
                "subject": t.get("subject"),
                "relation": t.get("relation"),
                "object": t.get("object"),
                "theme": theme,
                "evidence": t.get("evidence_list", []),
                "source_type": t.get("source_type", "")
            })
    return all_rels

def main() -> None:
    """Main execution flow."""
    # Ensure reproducibility
    random.seed(42)

    # 1. Select and load graph
    graph_path = select_graph_from_releases()
    if not graph_path:
        return
    
    kg = load_graph(graph_path)
    graph = kg["graph"]

    # 2. Configuration & Parameters
    try:
        print("\n--- Parameters ---")
        n_input = input(f"Enter Top N nodes per theme (default {TOP_N_NODES}): ").strip()
        top_n = int(n_input) if n_input else TOP_N_NODES

        hops_input = input(f"Enter number of hops to extract (default {DEFAULT_HOPS}): ").strip()
        hops = int(hops_input) if hops_input else DEFAULT_HOPS
        
        experts_input = input(f"Enter number of experts (files to create) (default {SPLIT}): ").strip()
        num_experts = int(experts_input) if experts_input else SPLIT

        samples_per_expert_input = input(f"Enter total samples PER expert file (default {DEFAULT_SAMPLES}): ").strip()
        samples_per_expert = int(samples_per_expert_input) if samples_per_expert_input else DEFAULT_SAMPLES

        overlap_input = input(f"Enter overlap percentage for IAA (e.g., 20 for 20%, default {DEFAULT_OVERLAP_PERCENT}): ").strip()
        overlap_percent = float(overlap_input) if overlap_input else DEFAULT_OVERLAP_PERCENT

        verification_p_input = input(f"Enter % of samples for Source Verification (default {DEFAULT_VERIFICATION_PERCENT}%): ").strip()
        verification_percent = float(verification_p_input) if verification_p_input else DEFAULT_VERIFICATION_PERCENT

        core_percent_input = input(f"Enter % of samples from CORE (Top-N) (default {DEFAULT_CORE_PERCENT}%): ").strip()
        core_percent = float(core_percent_input) if core_percent_input else DEFAULT_CORE_PERCENT

    except ValueError:
        print("\nInvalid number entered. Using defaults for all parameters.")
        top_n = TOP_N_NODES
        hops = DEFAULT_HOPS
        num_experts = SPLIT
        samples_per_expert = DEFAULT_SAMPLES
        overlap_percent = DEFAULT_OVERLAP_PERCENT
        verification_percent = DEFAULT_VERIFICATION_PERCENT
        core_percent = DEFAULT_CORE_PERCENT

    # Calculate sample sizes based on user input
    num_overlap = int(samples_per_expert * (overlap_percent / 100))
    num_unique_per_expert = samples_per_expert - num_overlap
    total_unique_needed = num_unique_per_expert * num_experts
    total_samples_to_draw = total_unique_needed + num_overlap

    print("\n--- Experiment Setup ---")
    print(f"Each of the {num_experts} experts will receive a file with {samples_per_expert} rows.")
    print(f"  - {num_overlap} rows will be shared among all experts (for IAA).")
    print(f"  - {num_unique_per_expert} rows will be unique to each expert.")
    print(f"Total triplets to draw from the graph pool: {total_samples_to_draw}")

    seed_theme_map = get_top_nodes_per_theme(graph, top_n=top_n)
    seed_nodes = list(seed_theme_map.keys())

    # 3. Build index for efficient traversal
    print("Building adjacency index...")
    index = build_adjacency_index(graph)

    # 4. Hybrid Extraction (Core + Tail)
    print(f"\n>>> Running Hybrid Extraction...")
    
    # A. Extract CORE (Top-N Neighborhood)
    print(f"  1. Extracting {hops}-hop neighborhood for Top-{top_n} nodes...")
    core_pool = []
    core_signatures = set()
    for seed in seed_nodes:
        neighborhood = extract_n_hop_neighborhood(
            seed_node=seed,
            index=index,
            max_hops=hops
        )
        for rel in neighborhood['relations']:
            triplet_sig = (rel['subject'], rel['relation'], rel['object'])
            if triplet_sig not in core_signatures:
                core_pool.append(rel)
                core_signatures.add(triplet_sig)
    print(f"     -> Found {len(core_pool)} unique Core triplets.")

    # B. Extract TAIL (Rest of graph)
    print(f"  2. Extracting Tail (Random) pool from remaining graph...")
    all_graph_rels = extract_all_relations(graph)
    tail_pool = []
    for rel in all_graph_rels:
        triplet_sig = (rel['subject'], rel['relation'], rel['object'])
        if triplet_sig not in core_signatures:
            tail_pool.append(rel)
    print(f"     -> Found {len(tail_pool)} unique Tail triplets.")

    # 5. Stratified Sampling (User Defined %)
    print(f"\n>>> Performing Stratified Sampling ({core_percent}% Core / {100-core_percent}% Tail)...")
    
    quota_core = int(total_samples_to_draw * (core_percent / 100))
    quota_tail = total_samples_to_draw - quota_core
    
    # Handle cases where pools are too small
    if len(core_pool) < quota_core:
        print(f"Warning: Not enough Core relations ({len(core_pool)} < {quota_core}). Taking all Core.")
        quota_tail += (quota_core - len(core_pool)) # Add deficit to tail
        sampled_core = core_pool
    else:
        sampled_core = random.sample(core_pool, quota_core)

    if len(tail_pool) < quota_tail:
        print(f"Warning: Not enough Tail relations ({len(tail_pool)} < {quota_tail}). Taking all Tail.")
        sampled_tail = tail_pool
    else:
        sampled_tail = random.sample(tail_pool, quota_tail)

    # Tag Origin and ID
    final_pool = []
    registry = {}
    
    for i, r in enumerate(sampled_core):
        r['origin'] = "top-n"
        final_pool.append(r)
        
    for i, r in enumerate(sampled_tail):
        r['origin'] = "random"
        final_pool.append(r)

    # Assign stable IDs and build registry
    for idx, r in enumerate(final_pool, 1001): # Start IDs at 1001
        evidence_list = r.get("evidence", [])
        selected_ev = evidence_list[0] if evidence_list else {}
        
        r['id'] = idx
        registry[idx] = {
            "subject": r['subject'],
            "relation": r['relation'],
            "object": r['object'],
            "evidence": {
                "text": selected_ev.get("evidence_text", ""),
                "doi": selected_ev.get("evidence_doi", ""),
                "page": selected_ev.get("evidence_page", "")
            },
            "theme": r['theme'],
            "origin": r['origin'],
            "source_type": r.get("source_type", "Unknown"),
            "is_shared": False, # Will be updated in save_results
            "assigned_to": []   # Will be updated in save_results
        }

    # Create output directory for this run
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}"
    
    outpath_dir = OUTPUT_DIR / run_name
    outpath_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save results and metadata
    csv_prefix = f""
    split_stats = save_results(
        relations_pool=final_pool, 
        registry=registry,
        output_dir=outpath_dir, 
        file_prefix=csv_prefix, 
        num_splits=num_experts,
        num_unique_per_expert=num_unique_per_expert,
        num_overlap=num_overlap,
        verification_percent=verification_percent
    )
    
    # Save Registry
    registry_path = outpath_dir / "samples_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    save_metadata(
        exp_name="Quality Evaluation",
        description=f"{hops}-hop neighborhood extraction for quality (coherence and plausibility) analysis.",
        output_dir=outpath_dir,
        graph_path=graph_path,
        seed_nodes=seed_nodes,
        top_n_value=top_n,
        random_seed_nodes="Remainder of graph (Random Selection)",
        max_hops=hops,
        total_samples=len(final_pool),
        samples_selected_core=len(sampled_core),
        samples_selected_tail=len(sampled_tail),
        pool_size_core=len(core_pool),
        pool_size_tail=len(tail_pool),
        samples_per_expert=samples_per_expert,
        overlap_percent=overlap_percent,
        verification_percent=verification_percent,
        split_stats=split_stats,
    )
    
    print(f"   Saved results to {outpath_dir}")
    print(f"   Master Log saved to {registry_path.name}")
        
if __name__ == "__main__":
    main()
