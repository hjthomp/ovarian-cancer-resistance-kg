"""
Generates comprehensive statistics for a Knowledge Graph (KG) release.
Calculates global coverage, source-specific metrics, and graph topology features.
Outputs results to a timestamped folder in /data/experiments/statistics/.
"""

import logging
import datetime
import statistics
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Union

import networkx as nx

from src.utils.io import load_json, save_json
from src.utils.constants import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = DATA_DIR / "experiments" / "statistics"
RELEASES_DIR = DATA_DIR / "releases"

TOP_N = 50
FLOAT_PRECISION = 6

def get_user_choice(options: List[Any], prompt_text: str) -> Any:
    """Helper to ask user for a choice from a list."""
    print(f"\n{prompt_text}")
    for i, option in enumerate(options, 1):
        display = option.name if isinstance(option, Path) else str(option)
        print(f"[{i}] {display}")
    
    while True:
        try:
            choice = int(input(f"Select an option (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def calculate_statistics(kg_data: Dict[str, Any]) -> Dict[str, Any]:
    """Computes various graph and content metrics from the KG data."""
    logging.info("Calculating statistics...")
    graph_data = kg_data.get("graph", kg_data)
    
    stats = {
        "global": {},
        "source_coverage": {},
        "clusters": {},
        "top_lists": {},
        "additional": {}
    }
    
    # Extract metadata if available
    if isinstance(kg_data, dict) and "metadata" in kg_data:
        stats["metadata"] = kg_data["metadata"]
    
    # Sets for unique counts
    all_triplets = set() # (s, r, o, theme)
    all_nodes = set()
    all_relations = set()
    
    # Source tracking
    nodes_from_articles = set()
    nodes_from_abstracts = set()
    triplets_from_articles = set()
    triplets_from_abstracts = set()
    
    # Cluster stats
    cluster_stats = defaultdict(lambda: {"nodes": set(), "triplets": set(), "relations": set()})
    
    # Evidence tracking
    unique_evidences = set()
    evidence_to_source = {} # To track source for each unique evidence

    # NetworkX Graph (MultiDiGraph to capture all edges/degrees correctly)
    G = nx.MultiDiGraph()
    
    total_evidences = 0
    
    # Normalize graph_data to dict if it's a list (though it's {theme: [triplets]})
    if isinstance(graph_data, list):
        graph_data = {"default": graph_data}

    for theme, triplets in graph_data.items():
        for t in triplets:
            s = t.get("subject")
            o = t.get("object")
            r = t.get("relation")
            
            if not s or not o or not r:
                continue
                
            # Unique Triplet ID (including theme)
            triplet_id = (s, r, o, theme)
            all_triplets.add(triplet_id)
            all_nodes.add(s)
            all_nodes.add(o)
            all_relations.add(r)
            
            # Add to graph
            G.add_edge(s, o, relation=r, theme=theme)
            
            # Cluster stats
            cluster_stats[theme]["nodes"].update([s, o])
            cluster_stats[theme]["triplets"].add(triplet_id)
            cluster_stats[theme]["relations"].add(r)
            
            # Evidence count
            total_evidences += len(t.get("evidence_list", []))
            
            # Determine the triplet source (article or abstract)
            source = t.get("source_type", "").lower()
            if "article" in source:
                nodes_from_articles.update([s, o])
                triplets_from_articles.add(triplet_id)
            
            if "abstract" in source:
                nodes_from_abstracts.update([s, o])
                triplets_from_abstracts.add(triplet_id)

            # Process evidences for unique counting
            for ev in t.get("evidence_list", []):
                ev_text = ev.get("evidence_text")
                if not ev_text:
                    continue
                
                unique_evidences.add(ev_text)

                # Store the source only the first time we see an evidence
                if ev_text not in evidence_to_source:
                    if "article" in source:
                        evidence_to_source[ev_text] = "article"
                    elif "abstract" in source:
                        evidence_to_source[ev_text] = "abstract"

    # --- 1. Global Coverage ---
    stats["global"]["total_triplets"] = len(all_triplets)
    stats["global"]["total_nodes"] = len(all_nodes)
    stats["global"]["total_relations"] = len(all_relations)
    stats["global"]["total_unique_evidences"] = len(unique_evidences)
    stats["global"]["total_evidences"] = total_evidences
    stats["global"]["total_themes"] = len(graph_data.keys())

    # Directed graph metrics (from the original MultiDiGraph)
    scc = list(nx.strongly_connected_components(G))
    stats["global"]["strongly_connected_components"] = len(scc)
    stats["global"]["directed_density"] = nx.density(G)
    
    # Graph Topology
    # Convert to undirected simple graph for density and components (calculate without directions)
    G_undirected = nx.Graph(G)
    
    # Weakly Connected Components
    wcc = list(nx.connected_components(G_undirected))
    stats["global"]["connected_components"] = len(wcc)
    
    # Density (undirected)
    stats["global"]["undirected_density"] = nx.density(G_undirected)
    
    # Degree (using MultiDiGraph to count all connections)
    degrees = [d for n, d in G.degree()]
    stats["global"]["avg_degree"] = statistics.mean(degrees) if degrees else 0
    stats["global"]["max_degree"] = max(degrees) if degrees else 0
    
    # --- 2. Coverage by Source ---
    stats["source_coverage"]["nodes_articles"] = len(nodes_from_articles)
    stats["source_coverage"]["nodes_abstracts"] = len(nodes_from_abstracts)
    stats["source_coverage"]["nodes_intersection"] = len(nodes_from_articles & nodes_from_abstracts)
    stats["source_coverage"]["nodes_exclusive_articles"] = len(nodes_from_articles - nodes_from_abstracts)
    stats["source_coverage"]["nodes_exclusive_abstracts"] = len(nodes_from_abstracts - nodes_from_articles)
    
    stats["source_coverage"]["triplets_articles"] = len(triplets_from_articles)
    stats["source_coverage"]["triplets_abstracts"] = len(triplets_from_abstracts)
    stats["source_coverage"]["triplets_intersection"] = len(triplets_from_articles & triplets_from_abstracts)
    
    # Count unique evidences per source
    evidence_source_counts = Counter(evidence_to_source.values())
    stats["source_coverage"]["evidences_articles"] = evidence_source_counts.get("article", 0)
    stats["source_coverage"]["evidences_abstracts"] = evidence_source_counts.get("abstract", 0)

    if stats["global"]["total_unique_evidences"] > 0:
        stats["global"]["avg_triplets_per_evidence"] = stats["global"]["total_triplets"] / stats["global"]["total_unique_evidences"]

    # --- 3. Additional Stats ---
    # Per Cluster
    for theme, data in cluster_stats.items():
        stats["clusters"][theme] = {
            "nodes": len(data["nodes"]),
            "triplets": len(data["triplets"]),
            "relations": len(data["relations"])
        }
        
    # Largest Component
    if wcc:
        largest_wcc = max(wcc, key=len)
        stats["additional"]["largest_wcc_size"] = len(largest_wcc)
        stats["additional"]["largest_wcc_ratio"] = len(largest_wcc) / len(all_nodes) if all_nodes else 0
    
    # Largest Strongly Connected Component
    if scc:
        largest_scc = max(scc, key=len)
        stats["additional"]["largest_scc_size"] = len(largest_scc)
        stats["additional"]["largest_scc_ratio"] = len(largest_scc) / len(all_nodes) if all_nodes else 0
    
    # Top Nodes (Degree)
    node_degrees = Counter(dict(G.degree()))
    stats["top_lists"]["top_nodes_by_degree"] = node_degrees.most_common(TOP_N)
    
    # Top Relations (Frequency)
    relation_counts = Counter([r for s, r, o, t in all_triplets])
    stats["top_lists"]["top_relations"] = relation_counts.most_common(TOP_N)
    
    return stats

def save_results(stats: Dict[str, Any], output_dir: Path) -> None:
    """Saves statistics as JSON and a human-readable text summary."""
    # Save JSON
    save_json(stats, output_dir / "statistics.json")
    
    # Save Readable Summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== KG STATISTICS SUMMARY ===\n\n")
        
        g = stats["global"]
        f.write("1. GLOBAL COVERAGE\n")
        f.write(f"- Total Triplets (unique w/ theme): {g['total_triplets']}\n")
        f.write(f"- Total Nodes: {g['total_nodes']}\n")
        f.write(f"- Total Relations: {g['total_relations']}\n")
        f.write(f"- Total Unique Evidences: {g['total_unique_evidences']}\n")
        f.write(f"- Avg. Triplets per Evidence: {g.get('avg_triplets_per_evidence', 0):.2f}\n")
        f.write(f"- Themes: {g['total_themes']}\n")
        f.write(f"- Weakly Connected Components: {g.get('connected_components', 0)}\n")
        f.write(f"- Strongly Connected Components: {g.get('strongly_connected_components', 0)}\n")
        f.write(f"- Graph Density (undirected): {g.get('undirected_density', 0):.{FLOAT_PRECISION}f}\n")
        f.write(f"- Graph Density (directed): {g.get('directed_density', 0):.{FLOAT_PRECISION}f}\n")
        f.write(f"- Avg Degree: {g.get('avg_degree', 0):.2f}\n")
        f.write(f"- Max Degree: {g.get('max_degree', 0)}\n\n")
        
        s = stats["source_coverage"]
        f.write("2. SOURCE COVERAGE\n")
        f.write(f"- Nodes from Articles: {s['nodes_articles']} (Exclusive: {s['nodes_exclusive_articles']})\n")
        f.write(f"- Nodes from Abstracts: {s['nodes_abstracts']} (Exclusive: {s['nodes_exclusive_abstracts']})\n")
        f.write(f"- Nodes Intersection: {s['nodes_intersection']}\n")
        f.write(f"- Triplets from Articles: {s['triplets_articles']}\n")
        f.write(f"- Triplets from Abstracts: {s['triplets_abstracts']}\n")
        f.write(f"- Unique Evidences from Articles: {s.get('evidences_articles', 0)}\n")
        f.write(f"- Unique Evidences from Abstracts: {s.get('evidences_abstracts', 0)}\n")
        f.write(f"- Triplets Intersection: {s['triplets_intersection']}\n\n")
        
        add = stats["additional"]
        f.write("3. ADDITIONAL\n")
        f.write(f"- Largest Weakly Connected Component Size: {add.get('largest_wcc_size', 0)}\n")
        f.write(f"- Largest Weakly Connected Component Ratio: {add.get('largest_wcc_ratio', 0):.2%}\n")
        f.write(f"- Largest Strongly Connected Component Size: {add.get('largest_scc_size', 0)}\n")
        f.write(f"- Largest Strongly Connected Component Ratio: {add.get('largest_scc_ratio', 0):.2%}\n\n")
        
        f.write("4. TOP NODES (Degree)\n")
        for node, deg in stats["top_lists"]["top_nodes_by_degree"]:
            f.write(f"- {node}: {deg}\n")
        f.write("\n")
        
        f.write("5. TOP RELATIONS\n")
        for rel, count in stats["top_lists"]["top_relations"]:
            f.write(f"- {rel}: {count}\n")
            
        if "metadata" in stats:
            f.write("\n6. METADATA\n")
            f.write(json.dumps(stats["metadata"], indent=2, ensure_ascii=False))
            f.write("\n")
            
    logging.info(f"Results saved to {output_dir}")

def run_statistics() -> None:
    """Main execution flow for the statistics tool."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("--- KG Statistics Tool ---")
    
    # 1. Select KG
    json_files = sorted(list(RELEASES_DIR.glob("*.json")))
    if not json_files:
        logging.error(f"No KG files found in {RELEASES_DIR}")
        return

    kg_file = get_user_choice(json_files, "Select Knowledge Graph file:")
    
    logging.info(f"Loading {kg_file.name}...")
    kg_data = load_json(kg_file)
    
    # 2. Calculate
    stats = calculate_statistics(kg_data)
    
    # 3. Save
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    save_results(stats, run_dir)
    print(f"\nDone. Check results in {run_dir}")

if __name__ == "__main__":
    run_statistics()