"""
Orchestrates the augmentation of Knowledge Graphs using PubMed abstracts.
Handles scraping via PubMedScraper, evidence extraction, graph merging, 
and session management (new runs, resumption, and partial generation).
"""

from copy import deepcopy
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Union

from .extract_evidence import run_extract_evidence
from ..abstract_augmenter.pubmed_scraper import PubMedScraper
from ..utils.io import load_json, save_json, load_yaml
from ..utils.constants import PROJECT_ROOT, DATA_DIR, CONFIG_DIR, ARTICLE_BUILDER_DIR, ABSTRACT_AUGMENTER_DIR, COMPLETION_FLAG, SUFFIX_AUGMENTED_KG, SUFFIX_PARTIAL_AUGMENTED_KG, SUFFIX_ARTICLE_KG
from ..utils.execution_measures import ExecutionMeasures

# --- util functions ---

def select_graph(search_dirs: Path, search_pattern: str = "kg*.json", require_completed: bool = False) -> Optional[Path]:
    """Lists and prompts the user to select a Knowledge Graph from a directory."""
    print("--- Select a Knowledge Graph ---")
    selectable_options = []

    if not search_dirs.exists():
        logging.warning(f"Directory not found, skipping: {search_dirs}")
        return None

    # Use glob with '**/' for recursive search. This finds all matching files.
    all_found_files = sorted(
        list(search_dirs.glob(f"**/{search_pattern}")),
        reverse=True
    )

    
    print(f"Graphs found in: {search_dirs.relative_to(PROJECT_ROOT)}")
    # Group files by their parent directory for cleaner display
    files_by_dir = {}
    for file_path in all_found_files:
        directory = file_path.parent
        # 🔹 filter by completion flag if required
        if require_completed and not (directory / COMPLETION_FLAG).exists():
            continue
        if directory not in files_by_dir:
            files_by_dir[directory] = []
        files_by_dir[directory].append(file_path)

    for directory, files in files_by_dir.items():
        # Display the directory relative to the search start
        relative_dir_path = directory.relative_to(search_dirs)
        print(f"  📁 {relative_dir_path}")
        for file_path in sorted(files):
            selectable_options.append(file_path)
            print(f"    [{len(selectable_options)}] {file_path.name}")

    if not selectable_options:
        print(f"\nNo files matching '{search_pattern}' were found in the specified directories.")
        return None

    print("-" * 32)
    while True:
        try:
            choice_str = input(
                f"Enter your choice (1-{len(selectable_options)}) or press Enter to go back: "
            )
            if not choice_str:
                print("No selection made. Returning.")
                return None

            choice = int(choice_str)
            if 1 <= choice <= len(selectable_options):
                return selectable_options[choice - 1]

            print("Invalid selection. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")

def show_metadata_and_confirm(metadata: Dict[str, Any]) -> bool:
    """Displays graph metadata and asks for user confirmation."""
    print("--- Graph Metadata ---")
    print("kg_name:", metadata["kg_name"])
    print("kg_creation_date:", metadata["kg_creation_date"])
    print("-" * 30)
    correct_graph = input("Is this the correct graph? (Y/n): ").strip().lower()
    return correct_graph in ('y', '')

def show_preview_and_confirm(items_to_process: List[Dict[str, Any]], show_item: int = 3, show_chars: int = 100) -> bool:
    """Shows a preview of the loaded abstracts and a confirmation message."""    
    if not items_to_process:
        print("\nNo text was loaded.")
        return False # Return early if there's nothing to show

    print("\n--- Preview of loaded abstracts ---")
    for i, article in enumerate(items_to_process[:show_item]):  # Preview up to show_item items from the list
        title = str(article.get('title', 'No Title'))[:show_chars] + '...'
        abstract_preview = str(article.get('abstract', 'No Abstract'))[:show_chars] + '...'
        print(f"[{i+1}] Title: {title}\n    Abstract: {abstract_preview}")
    if len(items_to_process) > show_item:
        print(f"... and {len(items_to_process) - show_item} more article(s).")

    print("\n" + "-"*30)
    print(f"{len(items_to_process)} item(s) are ready to be processed.")
    
    while True:
        confirm = input("Do you want to proceed with extraction of evidence? (Y/n): ").strip().lower()
        if confirm in ('y', ''):
            return True
        elif confirm == 'n':
            return False

def calculate_statistics(graph: Dict[str, List[Dict[str, Any]]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comparative statistics between article-based and abstract-based data."""
    article_node_list = set()
    abstract_node_list = set()
    
    # Sets for unique triplets (subject, relation, object)
    unique_article_triplets = set()
    unique_abstract_triplets = set()
    unique_relation_types = set()

    for theme, triplets in graph.items():
        for t in triplets:
            subj = t.get('subject')
            obj = t.get('object')
            rel = t.get('relation')

            if subj and obj and rel:
                # Tuple for uniqueness: Include theme to distinguish context
                triplet_tuple = (subj, rel, obj, theme)
                unique_relation_types.add(rel)

                if t.get('source_type') == 'article':
                    article_node_list.add(subj)
                    article_node_list.add(obj)
                    unique_article_triplets.add(triplet_tuple)
                elif t.get('source_type') == 'abstract':
                    abstract_node_list.add(subj)   
                    abstract_node_list.add(obj) 
                    unique_abstract_triplets.add(triplet_tuple)
    
    # Calculate Unions and Intersections
    all_nodes = article_node_list.union(abstract_node_list)
    intersection_nodes = article_node_list.intersection(abstract_node_list)
    exclusive_abstract_nodes = abstract_node_list - article_node_list
    intersection_triplets = unique_article_triplets.intersection(unique_abstract_triplets)
    exclusive_abstract_triplets = unique_abstract_triplets - unique_article_triplets

    statistics = {
        "source_documents": {
            "articles": len(metadata.get("processed_articles", {})),
            "abstracts": len(metadata.get("processed_abstracts", {})),
        },
        "graph_totals": {
            "themes": len(graph),
            "unique_nodes": len(all_nodes),
            "unique_triplets": len(unique_article_triplets.union(unique_abstract_triplets)),
            "unique_relation_types": len(unique_relation_types),
        },
        "nodes_breakdown": {
            "from_articles": len(article_node_list),
            "from_abstracts": len(abstract_node_list),
            "intersection": len(intersection_nodes),
            "exclusive_to_abstracts": len(exclusive_abstract_nodes)
        },
        "triplets_breakdown": {
            "from_articles": len(unique_article_triplets),
            "from_abstracts": len(unique_abstract_triplets),
            "intersection": len(intersection_triplets),
            "intersection_list": [f"{s} -> {r} -> {o} [{th}]" for s, r, o, th in intersection_triplets],
            "exclusive_to_abstracts": len(exclusive_abstract_triplets)
        }
    }
    return statistics

def print_statistics(stats: Dict[str, Any]) -> None:
    """Prints a formatted summary of the graph statistics."""
    print("\n" + "="*40)
    print("   GRAPH STATISTICS SUMMARY")
    print("="*40)
    
    docs = stats.get("source_documents", {})
    print(f"Sources Processed:")
    print(f"  • Articles (PDFs):   {docs.get('articles', 0)}")
    print(f"  • Abstracts:         {docs.get('abstracts', 0)}")
    
    totals = stats.get("graph_totals", {})
    print(f"\nGraph Content:")
    print(f"  • Total Themes:        {totals.get('themes', 0)}")
    print(f"  • Total Unique Nodes:    {totals.get('unique_nodes', 0)}")
    print(f"  • Total Unique Triplets: {totals.get('unique_triplets', 0)}")
    print(f"  • Unique Relation Types: {totals.get('unique_relation_types', 0)}")
    
    nodes = stats.get("nodes_breakdown", {})
    triplets = stats.get("triplets_breakdown", {})
    
    print(f"\nNodes Breakdown:")
    print(f"  • From Articles:       {nodes.get('from_articles', 0)}")
    print(f"  • From Abstracts:      {nodes.get('from_abstracts', 0)}")
    print(f"  • Intersection:        {nodes.get('intersection', 0)} (Common to both)")
    print(f"  • Exclusive to Abs:    {nodes.get('exclusive_to_abstracts', 0)} (New knowledge)")

    print(f"\nTriplets Breakdown:")
    print(f"  • From Articles:     {triplets.get('from_articles', 0)}")
    print(f"  • From Abstracts:    {triplets.get('from_abstracts', 0)}")
    print(f"  • Intersection:      {triplets.get('intersection', 0)} (Common to both)")
    if triplets.get('intersection', 0) > 0 and 'intersection_list' in triplets:
        for triplet_str in triplets['intersection_list']:
            print(f"    - {triplet_str}")
    print(f"  • Exclusive to Abs:    {triplets.get('exclusive_to_abstracts', 0)} (New knowledge)")
    
    print("="*40 + "\n")

# --- create run functions ---

def create_run_from_graph(mode: str = "article") -> None:
    """Starts a new augmentation run by scraping PubMed based on an existing graph."""
    print(f"Creating run from '{mode}' graph")

    # select graph to augment
    search_pattern = "kg*.json"
    if mode == 'article':
        base_graphs_dir = ARTICLE_BUILDER_DIR
        search_pattern = f"kg_{SUFFIX_ARTICLE_KG}.json"
    elif mode == 'augmented':
        base_graphs_dir = ABSTRACT_AUGMENTER_DIR

    graph_path = select_graph(base_graphs_dir, search_pattern=search_pattern)
    if not graph_path:
        print("No valid graph selected. Returning to main menu.")
        return

    # --- Load graph ---
    input_kg = load_json(graph_path)
    if not show_metadata_and_confirm(input_kg.get("metadata", {})):
        print("Operation cancelled. Returning to main menu.")
        return

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = ABSTRACT_AUGMENTER_DIR / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Scrape Articles from PubMed ---
    print("\n--- Scraping Articles from PubMed ---")
    try:
        # Load config to PubMed Scraper        
        config_path = CONFIG_DIR / "abstract_augmenter/pubmed_scraper.yaml"
        scraper_config = load_yaml(config_path)
        # Instantiate the PubMedScraper and run it
        logging.info("Starting scraping process...")
        
        # Use RAW_DIR for scraper cache to avoid re-downloading same queries
        raw_abstracts_dir = DATA_DIR / "raw/abstracts"
        pubmed_scraper = PubMedScraper(scraper_config, raw_abstracts_dir)
        
        scraper_path, pubmed_metadata = pubmed_scraper.scrape_articles()
        logging.info(f"Scraping complete. Output saved to: {scraper_path}")
        # load scraper data
        pubmed_data = load_json(scraper_path)
        # Extract PubMed metadata (all keys except 'results')
        pubmed_metadata = {k: v for k, v in pubmed_data.items() if k != 'results'}
    except (KeyError, FileNotFoundError) as e: 
        logging.error(f"Could not process PubMed Scraper. Details: {e}")
        return

    # Items to process
    all_scraped_articles = list(pubmed_data.get('results', {}).values())
    
    # Filter already processed DOIs to avoid duplicates
    logging.info("Checking duplicates DOIs from the selected graph.")
    existing_dois = set(input_kg.get("metadata", {}).get("processed_articles", {}).keys()) 
    if mode == 'augmented':
        existing_dois |= set(input_kg.get("metadata", {}).get("processed_abstracts", {}).keys())

    items_to_process = [
        article for article in all_scraped_articles
        if article.get("doi") and article["doi"] not in existing_dois
    ]
    logging.info(
        f"They will be processed {len(items_to_process)} new articles "
        f"(duplicated {len(all_scraped_articles) - len(items_to_process)})"
    )
    if not items_to_process:
        print("No new articles to process. Returning to main menu.")
        return

    # Confirm before continuing
    if not show_preview_and_confirm(items_to_process):
        print("Operation cancelled. Returning to main menu.")
        return
    
    # --- Extracting evidence from abstracts ---
    print("\n--- Extracting evidence from abstracts ---")
    # Get settings for extract_evidence
    config_path = CONFIG_DIR / "abstract_augmenter/settings.yaml"
    run_config = load_yaml(config_path)["extract_evidence"]

    # Save settings for the run
    run_settings = {
        'run_id': run_id,
        'input_graph_path': str(graph_path.relative_to(PROJECT_ROOT)),
        'total_items_to_process': len(items_to_process),
        'items_to_process': items_to_process,
        'extract_evidence_config': run_config,
        'pubmed_scraper_config': scraper_config,
        'pubmed_data_path': str(scraper_path.relative_to(PROJECT_ROOT)),
    }
    save_json(run_settings, output_path / "run_settings.json")

    # Get cluster names
    cluster_names = list(input_kg["graph"].keys())

    # Extract evidence
    final_state = run_extract_evidence(items_to_process, cluster_names, run_config, output_path)
    extracted_evidence = final_state.get("extracted_evidence", [])
    execution_measures = final_state.get("execution_measures", {})

    # Augment graph
    augmented_graph_data = augment_graph_with_evidence(input_kg, extracted_evidence, run_id, pubmed_metadata, mode, execution_measures_update=execution_measures)

    # Save graph
    print("\n--- Saving augmented graph ---")
    output_file = output_path / f"kg_{SUFFIX_AUGMENTED_KG}.json" # "kg_augmented.json"
    save_json(augmented_graph_data, output_file)
    print(f"\nAugmentation complete. The new graph is saved at: {output_file}")
    
    completion_flag_path = output_path / COMPLETION_FLAG
    completion_flag_path.touch()
    logging.info(f"Run '{output_path.name}' completed.")

    return

def augment_graph_with_evidence(
    base_graph_data: Dict[str, Any], 
    new_evidence: List[Dict[str, Any]], 
    run_id: str, 
    pubmed_metadata: Dict[str, Any], 
    mode: str, 
    execution_measures_update: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Integrates new extracted triplets and evidence into the base knowledge graph."""
    logging.info("Starting graph augmentation")

    # Make a deep copy to avoid modifying the original graph data in memory
    augmented_graph = deepcopy(base_graph_data).get('graph', {})
    augmented_metadata = deepcopy(base_graph_data.get('metadata', {}))

    # --- Augment the graph with new evidence ---
    evidence_added_count = 0
    processed_abstracts = {}

    for evidence_block in new_evidence:
        new_triplets = evidence_block.get("extracted_triplets", [])
        if not new_triplets:
            continue

        for t in new_triplets:
            theme = t.get("theme")
            if not theme:
                logging.warning(f"Skipping triplet due to missing 'theme': {t}")
                continue

            new_triplet = {
                "subject": t["subject"],
                "relation": t.get("relation"),
                "object": t["object"],
                "evidence_list": [{
                    "evidence_doi": evidence_block.get("doi"),
                    "evidence_text": evidence_block.get("generated_summary"),
                }],
                "source_type": f"abstract"
            }

            if theme not in augmented_graph:
                logging.warning(f"Cluster '{theme}' from evidence not found in base graph. Creating it.")
                augmented_graph[theme] = []

            # Basic check to avoid adding the exact same triplet to the same theme
            if new_triplet not in augmented_graph[theme]:
                augmented_graph[theme].append(new_triplet)
                evidence_added_count += 1
            else:
                logging.info(f"Skipping duplicate evidence for cluster '{theme}'.")

        # Common metadata for all triplets in this block
        ev_metadata = {
            "doi": evidence_block.get("doi"),
            "title": evidence_block.get("title"),
            "abstract": evidence_block.get("abstract"),
            "authors": evidence_block.get("authors"),
            "year": evidence_block.get("year"),
            "journal": evidence_block.get("journal"),
        }
        processed_abstracts.update({ ev_metadata.pop("doi") : ev_metadata })

    logging.info(f"Processed {len(new_evidence)} evidence blocks and added a total of {evidence_added_count} new triplets.")

    # --- Create/Update Metadata ---
    if mode == "augmented":
        # Ger previous abstracts, or empty dict if it doesn't exists
        all_processed_abstracts = augmented_metadata.get("processed_abstracts", {}).copy()
    else:
        all_processed_abstracts = {}

    # Update all_processed_abstracts
    all_processed_abstracts.update(processed_abstracts)
    # Update new kg_name
    kg_name = augmented_metadata["kg_name"]
    if not kg_name.endswith(f"_{SUFFIX_AUGMENTED_KG}"):
        kg_name += f"_{SUFFIX_AUGMENTED_KG}"

    # Make triplets_readable      
    triplets_readable = [
        f"{t['subject']} -> {t['relation']} -> {t['object']} [{topic}]"
        for topic, triplets in augmented_graph.items()
        for t in triplets
    ]
    
    new_metadata = {
        "kg_name" : kg_name,
        "kg_creation_date" : datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "processed_articles" : augmented_metadata["processed_articles"],
        "processed_abstracts" : all_processed_abstracts,
        "statistics" : {},
        "triplets_readable": triplets_readable,
        "run_history" : [],
    }

    # --- Initialize run history ---
    if "run_history" in augmented_metadata and augmented_metadata["run_history"]:
        # Copy previous history
        new_metadata["run_history"] = list(augmented_metadata["run_history"])
    else:
        # First element in history is for the 'article graph'
        new_metadata["run_history"] = [{
            "date": augmented_metadata["kg_creation_date"],
            "description": "Initial article graph.",
            "count_processed_dois": len(augmented_metadata["processed_articles"]),
            "processed_dois": list(augmented_metadata["processed_articles"].keys()),
        }]

    # --- Append current run ---
    new_metadata["run_history"].append(
        {
            "run_id": run_id,
            "date": new_metadata["kg_creation_date"],
            "description": "Augmented graph with abstracts.",
            "count_processed_dois": len(processed_abstracts),
            "processed_dois": list(processed_abstracts.keys()),
            "pubmed_search_params": pubmed_metadata,
        }
    )

    # statitics
    new_metadata["statistics"] = calculate_statistics(augmented_graph, new_metadata)
    print_statistics(new_metadata["statistics"])

    # Handle execution_measures
    if "execution_measures" in augmented_metadata:
        new_metadata["execution_measures"] = augmented_metadata["execution_measures"]
    else:
        new_metadata["execution_measures"] = {}
        
    if execution_measures_update:
        for stage_name, data in execution_measures_update.items():
            if stage_name in new_metadata["execution_measures"]:
                new_key = f"{stage_name}_{run_id}"
                logging.info(f"Execution measure '{stage_name}' already exists. Saving new measure as '{new_key}'.")
                new_metadata["execution_measures"][new_key] = data
            else:
                new_metadata["execution_measures"][stage_name] = data
        
    ExecutionMeasures.calculate_final_times(new_metadata["execution_measures"])

    final_graph = {
        'graph': augmented_graph,
        'metadata': new_metadata
    }
    
    logging.info("Graph augmentation complete.")
    return final_graph

# --- resume run functions ---

def select_run_to_resume(runs_dir: Path) -> Optional[Path]:
    """Lists and prompts to select an incomplete run directory."""
    print(f"\nSearching for incomplete runs in: {runs_dir}")
    if not runs_dir.exists():
        logging.warning("No 'runs' directory found. Nothing to resume.")
        return None

    # Get incomplete run directories (those without the completion flag), newest first
    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and not (d / COMPLETION_FLAG).exists()], reverse=True)

    if not run_dirs:
        logging.warning(f"No incomplete runs found in '{runs_dir}'. All runs are either complete or the directory is empty.")
        return None

    print("--- Found Incomplete Runs ---")
    for i, dir_path in enumerate(run_dirs):
        print(f"  📁 [{i + 1}] {dir_path.name}")
    print("-----------------------------")

    while True:
        choice_input = input(f"Choose a run (1-{len(run_dirs)}) or press Enter to go back: ").strip()
        if not choice_input:
            return None
        try:
            choice = int(choice_input)
            if 1 <= choice <= len(run_dirs):
                return run_dirs[choice - 1]
            else:
                print("Invalid selection. Please choose a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def resume_run() -> None:
    """Resumes an incomplete augmentation run from its last saved state."""
    print("--- Resume Run ---")
    # Select an incomplete run
    output_path = select_run_to_resume(ABSTRACT_AUGMENTER_DIR)
    if not output_path:
        return
    run_id = output_path.name
    
    # Load run_settings
    run_settings_path = output_path / "run_settings.json"
    run_settings = load_json(run_settings_path)
    if not run_settings:
        logging.error(f"Could not load file from '{run_settings_path}'.")
        return

    # Load graph 
    input_graph_path = PROJECT_ROOT / run_settings["input_graph_path"]
    input_kg = load_json(input_graph_path)
    if not input_kg:
        logging.error(f"Could not load graph from '{input_graph_path}'.")
        return

    # Load pubmed_scrapper_results.json using the path from run_settings
    pubmed_data_path = PROJECT_ROOT / run_settings["pubmed_data_path"]
    pubmed_data = load_json(pubmed_data_path)
    if not pubmed_data or not pubmed_data_path.exists():
        logging.error(f"Could not load pubmed_scraper_results.json from '{output_path}'.")
        return
    # Extract PubMed metadata (all keys except 'results')
    pubmed_metadata = {k: v for k, v in pubmed_data.items() if k != 'results'}

    print("\n --- Resuming run ---")
     # Items to process
    items_to_process = run_settings["items_to_process"]
    
    if not items_to_process:
        print("No new articles to process. Returning to main menu.")
        return
    
    logging.info(f"Resuming run that had {len(items_to_process)} articles to be processed.")
    
    # --- Extracting evidence from abstracts ---
    print("\n--- Extracting evidence from abstracts ---")
    
    run_config = run_settings["extract_evidence_config"]
    cluster_names = list(input_kg["graph"].keys())

    # Extract evidence
    final_state = run_extract_evidence(items_to_process, cluster_names, run_config, output_path)
    extracted_evidence = final_state.get("extracted_evidence", [])
    execution_measures = final_state.get("execution_measures", {})

    # Augment graph
    augmented_graph_data = augment_graph_with_evidence(input_kg, extracted_evidence, run_id, pubmed_metadata, "augmented", execution_measures_update=execution_measures)

    # Save graph
    print("\n--- Saving augmented graph ---")
    output_file = output_path / f"kg_{SUFFIX_AUGMENTED_KG}.json" # "kg_augmented.json""
    save_json(augmented_graph_data, output_file)
    print(f"\nAugmentation complete. The new graph is saved at: {output_file}")
    
    completion_flag_path = output_path / COMPLETION_FLAG
    completion_flag_path.touch()
    logging.info(f"Run '{output_path.name}' completed.")

    return


def generate_partial_graph_from_incomplete_run() -> None:
    """Generates a graph file using only the successfully processed items from an incomplete run."""
    print("--- Generate Partial Graph ---")
    # Select an incomplete run
    output_path = select_run_to_resume(ABSTRACT_AUGMENTER_DIR)
    if not output_path:
        return
    run_id = output_path.name
    
    # Load run_settings
    run_settings_path = output_path / "run_settings.json"
    run_settings = load_json(run_settings_path)
    if not run_settings:
        logging.error(f"Could not load file from '{run_settings_path}'.")
        return

    # Load graph 
    input_graph_path = PROJECT_ROOT / run_settings["input_graph_path"]
    input_kg = load_json(input_graph_path)
    if not input_kg:
        logging.error(f"Could not load graph from '{input_graph_path}'.")
        return

    # Load pubmed_scrapper_results.json
    pubmed_data = load_json(output_path / "pubmed_scraper_results.json")
    if not pubmed_data:
        logging.error(f"Could not load pubmed_scraper_results.json from '{output_path}'.")
        return
    # Extract PubMed metadata (all keys except 'results')
    pubmed_metadata = {k: v for k, v in pubmed_data.items() if k != 'results'}

    # Load processing_state 
    processing_state = load_json(output_path / "processing_state.json")
    if not processing_state:
        logging.error(f"Could not load processing_state.json from '{output_path}'.")
        logging.error(f"It doesn't possible generate a partial graph without processed items.")
        return

    print(f"\n Generating a partial graph from run '{output_path.name}'")

    # Handle both keys for backward compatibility
    extracted_evidence = processing_state.get("extracted_evidence", processing_state.get("extracted_evidence", []))
    execution_measures = processing_state.get("execution_measures", {})

    # Augment graph
    augmented_graph_data = augment_graph_with_evidence(input_kg, extracted_evidence, run_id, pubmed_metadata, "augmented", execution_measures_update=execution_measures)

    # Save graph
    print("\n--- Saving augmented graph ---")
    output_file = output_path / f"kg_{SUFFIX_PARTIAL_AUGMENTED_KG}.json" # "kg_partial_augmented.json
    save_json(augmented_graph_data, output_file)
    print(f"\nAugmentation complete. The new graph is saved at: {output_file}")
    
    logging.info(f"Run '{output_path.name}' will continue as incomplete.")
    return

def add_single_abstract(mode: str = "article") -> None:
    """Interactive tool to add a specific abstract (by DOI or manual entry) to a graph."""
    print(f"Add a single abstract from '{mode}' graph")

    # select graph to augment
    search_pattern = "kg*.json"
    if mode == 'article':
        search_pattern = f"kg_{SUFFIX_ARTICLE_KG}.json"
        graph_path = select_graph(ARTICLE_BUILDER_DIR, search_pattern=search_pattern)
    elif mode == 'augmented':
        graph_path = select_graph(ABSTRACT_AUGMENTER_DIR, require_completed=True)

    if not graph_path:
        print("No valid graph selected. Returning to main menu.")
        return
    
    # --- Load graph ---
    input_kg = load_json(graph_path)
    if not show_metadata_and_confirm(input_kg.get("metadata", {})):
        print("Operation cancelled. Returning to main menu.")
        return
    
    # Load run_settings
    if mode == 'article':
        # Force creation of basic run_settings
        run_settings = {
            "run_id": None,
            "input_graph_path": None,
            "total_items_to_process": None,
            "items_to_process": None,
            "extract_evidence_config": load_yaml(CONFIG_DIR / "abstract_augmenter/settings.yaml")["extract_evidence"],
            "pubmed_scraper_config": load_yaml(CONFIG_DIR / "abstract_augmenter/pubmed_scraper.yaml"),
        }

    elif mode == 'augmented':
        run_settings_path = graph_path.parent / "run_settings.json"
        run_settings = load_json(run_settings_path)

    if not run_settings:
        logging.error(f"Could not load file from '{run_settings_path}'.")
        return  

    print(f"Creating a new run")
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = ABSTRACT_AUGMENTER_DIR / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Get item to process
    article_to_process = None
    while True:
        print("\nSelect how to add the article data:")
        print(" [1] Provide a DOI (fetch from PubMed)")
        print(" [2] Enter details manually")
        add_choice = input("Choose an option or press Enter to go back: ").strip()
        if not add_choice:
            return
        if add_choice == "1":
            doi = input("Enter the article DOI: ")
    
            # DOI format: use only number and it must be in lowercase
            doi = doi.replace("https://doi.org/", "").replace("doi:", "").strip().lower()

            if not doi: 
                continue
            
            pubmed_scraper = PubMedScraper(run_settings["pubmed_scraper_config"], output_path)
            article_to_process = pubmed_scraper.scrape_article_by_doi(doi)
            if article_to_process:
                break
            else:
                print(f"Could not fetch article for DOI '{doi}'")
        elif add_choice == "2":
            title = input("Enter Title (required): ").strip()
            abstract = input("Enter Abstract (required): ").strip()
            if not title or not abstract:
                print("Title and Abstract are required.")
                continue
            article_to_process = {"title": title, "abstract": abstract}
            break
        else:
            print("Invalid choice. Try again.")

    
    # Confirm before continuing
    if not show_preview_and_confirm([article_to_process]):
        print("Operation cancelled. Returning to main menu.")
        return    
    # Update run_settings to be consistent with the single abstract to process
    run_settings["run_id"] = run_id
    run_settings["input_graph_path"] = str(graph_path.relative_to(PROJECT_ROOT)) 
    run_settings["total_items_to_process"] = 1
    run_settings["items_to_process"] = article_to_process
    run_settings["pubmed_scraper_config"] = {}
    save_json(run_settings, output_path / "run_settings.json")

    run_config = run_settings["extract_evidence_config"]
    cluster_names = list(input_kg["graph"].keys())

    # Extract evidence
    final_state = run_extract_evidence([article_to_process], cluster_names, run_config, output_path)
    extracted_evidence = final_state.get("extracted_evidence", [])
    execution_measures = final_state.get("execution_measures", {})

    # Augment graph
    augmented_graph_data = augment_graph_with_evidence(input_kg, extracted_evidence, run_id, {}, "augmented", execution_measures_update=execution_measures)

    # Save graph
    print("\n--- Saving augmented graph ---")
    output_file = output_path / f"kg_{SUFFIX_AUGMENTED_KG}.json" # "kg_augmented.json"
    save_json(augmented_graph_data, output_file)
    print(f"\nAugmentation complete. The new graph is saved at: {output_file}")
    
    completion_flag_path = output_path / COMPLETION_FLAG
    completion_flag_path.touch()
    logging.info(f"Run '{output_path.name}' completed.")

    return
