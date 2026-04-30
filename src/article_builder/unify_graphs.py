"""
Unifies multiple normalized article graphs into a single Knowledge Graph.
Calculates the intersection of themes and merges triplets, 
deduplicating evidence and aggregating execution metrics across all sources.
"""

import copy
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

from ..utils.io import load_json, save_json
from ..utils.constants import PROJECT_ROOT, ARTICLE_BUILDER_DIR, SUFFIX_ARTICLE_KG
from ..utils.execution_measures import ExecutionMeasures

def get_triplet_signature(theme: str, triplet: Dict[str, Any]) -> str:
    """Generates a unique signature for a triplet."""
    s = triplet.get("subject", "").strip().lower()
    r = triplet.get("relation", "").strip().lower()
    o = triplet.get("object", "").strip().lower()
    return f"{theme}|{s}|{r}|{o}"

def unify_graphs_by_themes(kgs_to_unify: List[Dict[str, Any]], common_themes: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Unifies graphs by filtering triplets that belong to the common themes and merging duplicates."""
    sorted_themes = sorted(list(common_themes))
    
    # Accumulator: signature -> triplet_data
    unified_index = {}

    for kg in kgs_to_unify:
        try:
            # Only iterate over themes that are in our common set
            current_themes = set(kg["graph"].keys())
            themes_to_process = current_themes.intersection(common_themes)
            kg_type = kg["metadata"].get("kg_type")

            for theme in themes_to_process:
                triplets = kg["graph"][theme]
                for t in triplets:
                    # Normalize structure if coming from base KG (legacy support or mixed types)
                    if kg_type == 'base' and "evidence_list" not in t:
                        t["evidence_list"] = [{
                            "evidence_doi": t.pop("evidence_doi", None), 
                            "evidence_page": t.pop("evidence_page", None), 
                            "evidence_text": t.pop("evidence_text", None), 
                            "triplet_topic": t.pop("triplet_topic", None)
                        }]
                    
                    # Ensure source_type exists
                    if "source_type" not in t:
                        t["source_type"] = "article"

                    sig = get_triplet_signature(theme, t)
                    
                    if sig not in unified_index:
                        # New entry
                        unified_index[sig] = {
                            "subject": t["subject"],
                            "relation": t["relation"],
                            "object": t["object"],
                            "theme": theme,
                            "source_type": t["source_type"],
                            "evidence_list": t["evidence_list"][:] # Copy list
                        }
                    else:
                        # Merge existing
                        existing = unified_index[sig]
                        
                        # Deduplication logic for evidence
                        existing_ev_signatures = set()
                        for ev in existing["evidence_list"]:
                            # Create a signature for evidence to avoid exact duplicates
                            # Using DOI and text as unique identifier
                            ev_sig = (ev.get("evidence_doi"), ev.get("evidence_text"))
                            existing_ev_signatures.add(ev_sig)
                        
                        for new_ev in t["evidence_list"]:
                            new_ev_sig = (new_ev.get("evidence_doi"), new_ev.get("evidence_text"))
                            if new_ev_sig not in existing_ev_signatures:
                                existing["evidence_list"].append(new_ev)
                                existing_ev_signatures.add(new_ev_sig)
                        
                        # Update source_type priority (article > abstract)
                        if t["source_type"] == "article" and existing["source_type"] != "article":
                            existing["source_type"] = "article"
        
        except Exception as e:
            logging.warning(
                f"Error while unifying a KG. Skipping this graph. Error: {e}"
            )

    # Reconstruct graph structure
    unified_graph = {theme: [] for theme in sorted_themes}
    
    for sig, data in unified_index.items():
        theme = data["theme"]
        triplet_obj = {
            "subject": data["subject"],
            "relation": data["relation"],
            "object": data["object"],
            "source_type": data["source_type"],
            "evidence_list": data["evidence_list"]
        }
        if theme in unified_graph:
            unified_graph[theme].append(triplet_obj)

    return unified_graph

def merge_execution_measures(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Merges source execution measures into target, summing durations and stats."""
    for key, value in source.items():
        if key == "total_pipeline_duration":
            continue
        
        if key not in target:
            target[key] = copy.deepcopy(value)
        else:
            # Sum duration
            if "duration_seconds" in value:
                target[key]["duration_seconds"] = round(target[key].get("duration_seconds", 0) + value["duration_seconds"], 2)
            
            # Sum LLM stats
            if "llm_stats" in value:
                if "llm_stats" not in target[key]:
                    target[key]["llm_stats"] = value["llm_stats"].copy()
                else:
                    for k, v in value["llm_stats"].items():
                        target[key]["llm_stats"][k] = target[key]["llm_stats"].get(k, 0) + v

def run_unify_articles(article_graph_files: List[Path]) -> None:
    """Orchestrates the unification process for a list of article graph files."""
    
    # load graphs to unify
    kgs_to_unify = []
    execution_measures = { "total_pipeline_duration": 0.0 }
    for kg_file in article_graph_files:
        kg_data = load_json(kg_file)
        kgs_to_unify.append(kg_data)
        execution_measures["total_pipeline_duration"] += kg_data["metadata"]["execution_measures"]["total_pipeline_duration"]

    with ExecutionMeasures(execution_measures, "unification", module_name="unify_graphs") as em:
        # --- Step 1: Identify Common Themes (Intersection) ---
        logging.info("\n--- STEP 1: Identifying Common Themes (Intersection) ---")
        
        if not kgs_to_unify:
            raise ValueError("No graphs provided to unify.")

        # Calculate intersection: Start with keys from the first graph and intersect with the rest
        common_themes = set(kgs_to_unify[0]["graph"].keys())
        for kg in kgs_to_unify[1:]:
            common_themes.intersection_update(kg["graph"].keys())
        
        logging.info(f"Found {len(common_themes)} themes common to all {len(kgs_to_unify)} graphs.")
        
        if not common_themes:
            logging.warning("No common themes found. The resulting unified graph will be empty.")

        # --- Step 2: Unify graphs using common themes ---
        logging.info("\n--- STEP 2: Building Base Graph Using Common Themes ---")
        unified_graph = unify_graphs_by_themes(kgs_to_unify, common_themes)

    # --- Step 3: Finalize and Save the Unified Graph ---
    logging.info("\n--- STEP 3: Finalizing and Saving the Graph ---")

    # Extract DOIs from each source graph for metadata traceability
    name_list = []
    processed_articles = {}
    final_execution_measures = {}
    for kg in kgs_to_unify:
        name_list.append(kg["metadata"].get("kg_name", "")),
        doi = kg["metadata"].get("doi", None)
        title = kg["metadata"].get("title", None)
        authors = kg["metadata"].get("authors", None)
        journal = kg["metadata"].get("journal", None)
        published = kg["metadata"].get("published", None)  
        processed_articles.update({
            doi : {
                "title": title,
                "authors": authors,
                "journal": journal,
                "published": published,
            }       
        })
        merge_execution_measures(final_execution_measures, kg["metadata"]["execution_measures"])
    
    merge_execution_measures(final_execution_measures, execution_measures)
    final_execution_measures["total_pipeline_duration"] = execution_measures["total_pipeline_duration"]

    themes = list(unified_graph.keys())
    triplets_per_theme = {theme: len(triplets) for theme, triplets in unified_graph.items()}
    # Make triplets_readable      
    triplets_readable = [
        f"{t['subject']} -> {t['relation']} -> {t['object']} [{theme}]"
        for theme, triplets in unified_graph.items()
        for t in triplets
    ]

    metadata = {
        "kg_name": "_".join(name_list),
        "kg_creation_date": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "kg_type": "unified",
        "processed_articles": processed_articles,
        "statistics": {
            "total_themes": len(themes),
            "total_triplets": sum(triplets_per_theme.values()),
            "triplets_per_themes": triplets_per_theme
        },
        "triplets_readable": triplets_readable,
        "source_files": [str(f.relative_to(PROJECT_ROOT)) for f in article_graph_files],
        "execution_measures": final_execution_measures
    }
    final_kg = {
        "graph": unified_graph,
        "metadata": metadata
    }

    # Calculate final times (only for this stage, as it is a new graph)
    ExecutionMeasures.calculate_final_times(final_kg["metadata"]["execution_measures"])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_kg_path = ARTICLE_BUILDER_DIR / f"{timestamp}" / "unified_graphs" / f"kg_{SUFFIX_ARTICLE_KG}.json"
    output_kg_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(final_kg, output_kg_path, indent=2)

    logging.info(f"\n✔ Unified graph creation complete! Saved to: {output_kg_path}")
  
if __name__ == '__main__':
    logging.warning("This script is not intended to be run directly. Please use 'main.py'.")