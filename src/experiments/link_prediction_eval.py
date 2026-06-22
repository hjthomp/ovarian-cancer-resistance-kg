"""
Evaluates Link Prediction (Triple Completion) performance using Multiple Choice Questions (MCQ).
Compares Base LLM vs. GraphRAG by generating categorized distractors (Hard/Medium/Easy)
and calculating ranking metrics like Hits@k and Mean Reciprocal Rank (MRR).
"""

import random
import logging
import datetime
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
from typing import List, Dict, Any, Tuple, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.io import load_json, save_json, load_yaml, load_txt
from src.utils.constants import DATA_DIR, CONFIG_DIR, PROMPT_DIR
from src.graph_rag.rag_pipeline import GraphRAGPipeline, ClassicRAGPipeline
from src.utils.execution_measures import ExecutionMeasures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Metrics Configuration
HITS_LIST = [1, 3]
FLOAT_PRECISION = 4

# Paths
RELEASES_DIR = DATA_DIR / "releases"
OUTPUT_BASE_DIR = DATA_DIR / "experiments" / "link_prediction_eval"
CONFIG_PATH = CONFIG_DIR / "experiments" / "link_prediction_eval.yaml"
PROMPT_PATH = PROMPT_DIR / "experiments" / "link_prediction.txt"

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

def prepare_evaluation_data(
    kg_data: Dict[str, Any], 
    embedding_model_name: str
) -> Tuple[List[Dict[str, str]], List[str], np.ndarray, Dict[str, int], Dict[str, Set[str]]]:
    """Extracts triplets and prepares node embeddings for distractor generation."""
    logging.info("Preparing nodes and embeddings for distractor generation...")
    graph = kg_data.get("graph", {})
    all_nodes = set()
    nodes_by_theme = {}
    triplets = []

    for theme, theme_triplets in graph.items():
        nodes_by_theme[theme] = set()
        for t in theme_triplets:
            s, r, o = t['subject'], t['relation'], t['object']
            all_nodes.update([s, o])
            nodes_by_theme[theme].update([s, o])
            triplets.append({"s": s, "r": r, "o": o, "theme": theme})

    node_list = sorted(list(all_nodes))
    embed_model = SentenceTransformer(embedding_model_name)
    node_embeddings = embed_model.encode(node_list, show_progress_bar=True)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    return triplets, node_list, node_embeddings, node_to_idx, nodes_by_theme

def get_distractors(
    target_node: str, 
    theme: str, 
    node_to_idx: Dict[str, int], 
    node_list: List[str], 
    node_embeddings: np.ndarray, 
    nodes_by_theme: Dict[str, Set[str]], 
    bench_cfg: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Generates categorized distractors: Hard (Similarity), Medium (Same Theme), Easy (Random)."""
    dist_dict = {"hard": [], "medium": [], "easy": []}
    
    # 1. HARD (Similarity)
    target_idx = node_to_idx[target_node]
    target_emb = node_embeddings[target_idx].reshape(1, -1)
    sims = cosine_similarity(target_emb, node_embeddings)[0]
    related_indices = np.argsort(sims)[::-1]
    
    for idx in related_indices:
        node = node_list[idx]
        if node != target_node:
            dist_dict["hard"].append(node)
            if len(dist_dict["hard"]) >= bench_cfg['distractors']['n_hard']: break

    # 2. MEDIUM (Same Theme)
    theme_nodes = sorted([n for n in nodes_by_theme[theme] if n != target_node and n not in dist_dict["hard"]])
    if theme_nodes:
        n_needed = bench_cfg['distractors']['n_medium']
        dist_dict["medium"] = random.sample(theme_nodes, min(len(theme_nodes), n_needed))

    # 3. EASY (Random)
    n_easy_needed = bench_cfg['distractors']['n_easy']
    all_used = set(dist_dict["hard"] + dist_dict["medium"] + [target_node])
    easy_pool = [n for n in node_list if n not in all_used]
    if easy_pool:
        dist_dict["easy"] = random.sample(easy_pool, min(len(easy_pool), n_easy_needed))
            
    return dist_dict

def get_error_type(predicted_node: str, distractors_dict: Dict[str, List[str]]) -> str:
    """Identifies which category a distractor belongs to."""
    for d_type, d_list in distractors_dict.items():
        if predicted_node in d_list:
            return d_type
    return "unknown"

def create_benchmark(
    triplets: List[Dict[str, str]], 
    node_to_idx: Dict[str, int], 
    node_list: List[str], 
    node_embeddings: np.ndarray, 
    nodes_by_theme: Dict[str, Set[str]], 
    bench_cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generates the full set of MCQ samples for evaluation."""
    logging.info(f"Generating benchmark of {bench_cfg['sample_size']} samples...")
    sorted_triplets = sorted(triplets, key=lambda x: (x['s'], x['r'], x['o']))
    samples = random.sample(sorted_triplets, min(len(sorted_triplets), bench_cfg['sample_size']))
    benchmark = []
    
    for i, s in enumerate(samples):
        dist_dict = get_distractors(s['o'], s['theme'], node_to_idx, node_list, node_embeddings, nodes_by_theme, bench_cfg)
        
        # Shuffled options for the prompt
        all_distractors = dist_dict["hard"] + dist_dict["medium"] + dist_dict["easy"]
        options = all_distractors + [s['o']]
        random.shuffle(options)
        
        benchmark.append({
            "sample_id": i + 1,
            "subject": s['s'],
            "relation": s['r'],
            "correct_answer": s['o'],
            "theme": s['theme'],
            "distractors": dist_dict,
            "shuffled_options": options
        })
    return benchmark

def calculate_metrics(
    results: List[Dict[str, Any]], 
    hits_list: List[int], 
    skipped_samples: int = 0, 
    total_processed: int = 0
) -> Dict[str, Any]:
    """Calculates Hits@k and MRR based on LLM rankings."""
    hits_counts = {k: 0 for k in hits_list}
    # Initialize error categories with counts and node lists
    confusion = {
        "hard": {"count": 0, "samples": []},
        "medium": {"count": 0, "samples": []},
        "easy": {"count": 0, "samples": []},
        "unknown": {"count": 0, "samples": []}
    }
    mrr = 0.0
    total = len(results)

    if total == 0:
        metrics = OrderedDict()
        for k in hits_list:
            metrics[f"hits@{k}"] = 0.0
        metrics["mrr"] = 0.0
        metrics["processed_samples"] = total_processed
        metrics["fair_comparison_samples"] = 0
        metrics["skipped_samples"] = skipped_samples
        return metrics

    for res in results:
        ranking = res.get("ranking", [])
        correct = res.get("correct_answer")
        error_type = res.get("error_type")
        predicted_top_1 = res.get("predicted_top_1")
        
        rank = (ranking.index(correct) + 1) if correct in ranking else None
        
        if rank:
            for k in hits_list:
                if rank <= k:
                    hits_counts[k] += 1
            mrr += (1.0 / rank)

        # Error analysis in the Top-1
        if rank != 1:
            et = error_type if error_type in ["hard", "medium", "easy"] else "unknown"
            confusion[et]["count"] += 1
            # Save only the ID to avoid a long JSON file
            confusion[et]["samples"].append(res.get("sample_id"))
    
    metrics = OrderedDict()
    for k in hits_list:
        metrics[f"hits@{k}"] = round(hits_counts[k] / total, FLOAT_PRECISION)
        metrics[f"hits@{k}_count"] = hits_counts[k]
    metrics["mrr"] = round(mrr / total, FLOAT_PRECISION)
    metrics["processed_samples"] = total_processed
    metrics["fair_comparison_samples"] = total
    metrics["skipped_samples"] = skipped_samples
    metrics["hits@1_distractor_confusion"] = confusion
    return metrics

def calculate_metrics_for_mode(
    samples_results: List[Dict[str, Any]], 
    benchmark: List[Dict[str, Any]], 
    mode: str, 
    hits_list: List[int], 
    common_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Helper to prepare data for calculate_metrics from the structured results."""
    mode_data = []
    skipped_count = 0
    successful_count = 0
    for i, s_res in enumerate(samples_results):
        status = s_res.get(mode)
        
        if status == "SKIPPED":
            skipped_count += 1
        
        if isinstance(status, dict) and "ranking" in status:
            successful_count += 1
            
        # If there is a list of common indexes (Fair Comparison), we ignore anything outside of it
        if common_indices is not None and i not in common_indices:
            continue
            
        # Only include successful dictionary responses in the metrics calculation
        if isinstance(status, dict) and "ranking" in status:
            mode_data.append({
                "sample_id": s_res.get("sample_id"),
                "ranking": status["ranking"],
                "correct_answer": benchmark[i]["correct_answer"],
                "error_type": status.get("error_type"),
                "predicted_top_1": status.get("predicted_top_1")
            })

    return calculate_metrics(mode_data, hits_list, skipped_samples=skipped_count, total_processed=successful_count)

def _print_error_dist(metrics: Dict[str, Any]) -> None:
    """Helper to print error distribution."""
    dist = metrics.get("hits@1_distractor_confusion", {})
    if dist:
        print("  Hits@1 Error Counts by Distractor Level:")
        for t in ["hard", "medium", "easy", "unknown"]:
            count = dist.get(t, {}).get("count", 0)
            print(f"    - {t.capitalize()}: {count}")

def recalculate_metrics_tool() -> None:
    """Tool to recalculate metrics from an existing results folder."""
    runs = sorted([d for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir()], reverse=True)
    if not runs:
        logging.error("No previous runs found.")
        return
    
    run_dir = get_user_choice(runs, "Select run folder to recalculate metrics:")
    
    if not (run_dir / "results.json").exists() or not (run_dir / "benchmark.json").exists():
        logging.error("Missing results.json or benchmark.json in the selected folder.")
        return

    results = load_json(run_dir / "results.json")
    benchmark = load_json(run_dir / "benchmark.json")
    
    logging.info(f"Recalculating metrics for {run_dir.name}...")
    
    # Fair Comparison: Identify samples successfully completed by all models
    has_classic = any(s.get("classic_rag") is not None for s in results.get("samples", []))
    if has_classic:
        common = [idx for idx, s in enumerate(results["samples"]) 
                  if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                     (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"]) and
                     (isinstance(s.get("classic_rag"), dict) and "ranking" in s["classic_rag"])]
        modes = ["base", "graph_rag", "classic_rag"]
    else:
        common = [idx for idx, s in enumerate(results["samples"]) 
                  if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                     (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"])]
        modes = ["base", "graph_rag"]
    logging.info(f"Fair Comparison: Using {len(common)} common samples for metrics.")

    # 1. Per-mode metrics
    for m_key in modes:
        if results["samples"]:
            results["metrics"][m_key] = calculate_metrics_for_mode(
                results["samples"], benchmark, m_key, HITS_LIST, common_indices=common
            )
    
    save_json(results, run_dir / "results.json")
    logging.info("Metrics recalculated and results.json updated.")
    
    # Print summary
    print(f"\n=== RECALCULATED RESULTS ({run_dir.name}) ===")
    for mode in results["metrics"].keys():
        m = results["metrics"].get(mode, {})
        if not m:
            continue
        fair_total = m.get('fair_comparison_samples', 0)
        print(f"\nMode: {mode} (Processed: {m.get('processed_samples')}, Denominator: {fair_total})")
        for k in HITS_LIST:
            val = m.get(f"hits@{k}", 0.0)
            count = m.get(f"hits@{k}_count", 0)
            print(f"  Hits@{k}: {val:.{FLOAT_PRECISION}f} ({count}/{fair_total})")
        print(f"  MRR:    {m.get('mrr', 0.0):.{FLOAT_PRECISION}f}")
        _print_error_dist(m)

def run_link_prediction_eval() -> None:
    """Main execution flow for the evaluation."""
    print("--- Link Prediction Evaluation Tool ---")
    
    # 1. Action Selection
    action = get_user_choice(["New Run", "Resume Run", "Recalculate Metrics (from folder)"], "Select evaluation mode:")
    
    if action == "Recalculate Metrics (from folder)":
        recalculate_metrics_tool()
        return
    
    run_dir = None
    current_config = load_yaml(CONFIG_PATH)
    prompt_template = load_txt(PROMPT_PATH)
    original_config = current_config # Default for new run

    if action == "New Run":
        # Select KG
        json_files = sorted(list(RELEASES_DIR.glob("*.json")))
        if not json_files:
            logging.error(f"No KG files found in {RELEASES_DIR}")
            return
        kg_path = get_user_choice(json_files, "Select Knowledge Graph file for evaluation:")
        
        # Set seeds BEFORE generating data and benchmark for reproducibility
        bench_cfg = current_config['benchmark']
        seed = bench_cfg.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)

        # Setup Run Directory
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = OUTPUT_BASE_DIR / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare Data and Benchmark
        kg_data = load_json(kg_path)
        triplets, node_list, node_embeddings, node_to_idx, nodes_by_theme = prepare_evaluation_data(
            kg_data, current_config['benchmark']['embedding_model']
        )
        benchmark = create_benchmark(triplets, node_to_idx, node_list, node_embeddings, nodes_by_theme, current_config['benchmark'])
        
        # Initialize Metadata
        metadata = {
            "experiment_name": "Link Prediction Evaluation",
            "action": action,
            "config": current_config,
            "kg_file": str(kg_path.name),
            "prompt_template": prompt_template,
            "execution_measures": {
                "start_time": datetime.datetime.now().isoformat(),
                "sample_size": len(benchmark)
            },
            "hits_tested": HITS_LIST
        }
        save_json(metadata, run_dir / "metadata.json")
        save_json(benchmark, run_dir / "benchmark.json")
        
        results = {
            "metrics": {"base": {}, "graph_rag": {}, "classic_rag": {}},
            "samples": []
        }
        for item in benchmark:
            results["samples"].append({"sample_id": item["sample_id"], "base": None, "graph_rag": None, "classic_rag": None})
        
    else:
        # Resume Run
        runs = sorted([d for d in OUTPUT_BASE_DIR.iterdir() if d.is_dir()], reverse=True)
        if not runs:
            logging.error("No previous runs found.")
            return
        run_dir = get_user_choice(runs, "Select run to resume:")
        
        metadata = load_json(run_dir / "metadata.json")
        benchmark = load_json(run_dir / "benchmark.json")
        original_config = metadata['config']
        
        # Enforce that classic_rag_config is present in metadata config
        if "classic_rag_config" not in original_config:
            original_config["classic_rag_config"] = current_config.get("classic_rag_config", {})
            metadata["config"] = original_config
            save_json(metadata, run_dir / "metadata.json")
            
        kg_path = RELEASES_DIR / metadata['kg_file']
        
        if (run_dir / "results.json").exists():
            results = load_json(run_dir / "results.json")
            
            # Backfill classic_rag fields if resuming a legacy run
            if "classic_rag" not in results["metrics"]:
                results["metrics"]["classic_rag"] = {}
            for s in results["samples"]:
                if "classic_rag" not in s:
                    s["classic_rag"] = None

            # Refresh metrics for all modes upon resume to ensure new logic is applied
            logging.info("Refreshing metrics with Fair Comparison logic...")
            
            has_classic = any(s.get("classic_rag") is not None for s in results.get("samples", []))
            if has_classic:
                common = [idx for idx, s in enumerate(results["samples"]) 
                          if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                             (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"]) and
                             (isinstance(s.get("classic_rag"), dict) and "ranking" in s["classic_rag"])]
                modes = ["base", "graph_rag", "classic_rag"]
            else:
                common = [idx for idx, s in enumerate(results["samples"]) 
                          if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                             (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"])]
                modes = ["base", "graph_rag"]

            for m_key in modes:
                if results["samples"]:
                    results["metrics"][m_key] = calculate_metrics_for_mode(
                        results["samples"], benchmark, m_key, HITS_LIST, common_indices=common)
            save_json(results, run_dir / "results.json")
        else:
            results = {"metrics": {"base": {}, "graph_rag": {}, "classic_rag": {}}, "samples": [{"sample_id": i+1, "base": None, "graph_rag": None, "classic_rag": None} for i in range(len(benchmark))]}

    # Setup logic
    bench_cfg = original_config['benchmark']
    # Re-apply seed to ensure any subsequent random operations (like LLM sampling if temp > 0) are consistent
    seed = bench_cfg.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. Initialize Pipelines
    base_cfg = {
        "name": "Base_LLM",
        "retrieval": {"enabled": False},
        "prompt_template": prompt_template,
        "llm": original_config.get("model_settings")
    }
    
    retrieval_cfg = original_config['graph_rag_config']['retrieval'].copy()
    db_path_session = (run_dir / "vector_db").relative_to(DATA_DIR.parent)
    retrieval_cfg.update({
        "graph_path": str(kg_path.relative_to(DATA_DIR.parent)),
        "db_path": str(db_path_session),
        "embedding_model": retrieval_cfg.get("embedding_model", bench_cfg.get("embedding_model"))
    })

    rag_cfg = base_cfg.copy()
    rag_cfg.update({
        "name": "GraphRAG_LLM",
        "retrieval": retrieval_cfg,
        "prompt": {"use_evidence": original_config['graph_rag_config'].get("use_evidence", False)},
    })
    
    classic_rag_config = original_config.get("classic_rag_config")
    if not classic_rag_config:
        classic_rag_config = current_config.get("classic_rag_config", {})
    classic_cfg = base_cfg.copy()
    classic_cfg.update({
        "name": "Classic_RAG_LLM",
        "retrieval": {
            "enabled": classic_rag_config.get("retrieval", {}).get("enabled", True),
            "db_path": str(db_path_session),
            "embedding_model": classic_rag_config.get("retrieval", {}).get("embedding_model", bench_cfg.get("embedding_model")),
            "top_k": classic_rag_config.get("retrieval", {}).get("top_k", 15)
        },
        "data_sources": classic_rag_config.get("data_sources", {})
    })

    pipelines = {
        "base": GraphRAGPipeline(base_cfg),
        "graph_rag": GraphRAGPipeline(rag_cfg),
        "classic_rag": ClassicRAGPipeline(classic_cfg)
    }

    execution_measures = metadata["execution_measures"]
    if "llm_stats" not in execution_measures:
        execution_measures["llm_stats"] = {"total_requests": 0, "total_tokens": 0, "total_wall_time": 0}

    logging.info(f"Starting evaluation loop. Results will be saved to {run_dir}")

    # 3. Execution Loop
    with ExecutionMeasures(execution_measures, "link_prediction") as em:
        for i, item in enumerate(benchmark):
            logging.info(f" Processing Sample {i+1}/{len(benchmark)}")
            
            for mode in ["base", "graph_rag", "classic_rag"]:
                pipeline = pipelines[mode]
                current_status = results["samples"][i].get(mode)
                
                # 1. Skip if successfully completed or already skipped
                if isinstance(current_status, dict) or current_status == "SKIPPED":
                    continue

                # 2. Handle interrupted samples (Tombstone recovery)
                if current_status == "PROCESSING":
                    print(f"\n[!] ALERT: Sample {i+1} was being processed by '{mode}' but didn't finish (possible crash).")
                    recovery_action = get_user_choice(["Retry", "Skip this sample for this model"], "How to proceed?")
                    if recovery_action == "Skip this sample for this model":
                        results["samples"][i][mode] = "SKIPPED"
                        
                        # Recalculate metrics with Fair Comparison
                        has_classic = any(s.get("classic_rag") is not None for s in results.get("samples", []))
                        if has_classic:
                            common = [idx for idx, s in enumerate(results["samples"]) 
                                      if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                                         (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"]) and
                                         (isinstance(s.get("classic_rag"), dict) and "ranking" in s["classic_rag"])]
                            modes = ["base", "graph_rag", "classic_rag"]
                        else:
                            common = [idx for idx, s in enumerate(results["samples"]) 
                                      if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                                         (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"])]
                            modes = ["base", "graph_rag"]

                        for m_key in modes:
                            results["metrics"][m_key] = calculate_metrics_for_mode(
                                results["samples"], benchmark, m_key, HITS_LIST, common_indices=common)
                        
                        save_json(results, run_dir / "results.json")
                        continue
                    # If Retry, we just fall through to the processing logic
                
                logging.info(f"   [{mode}] Running evaluation...")
                
                # 3. Mark as PROCESSING before the LLM call
                results["samples"][i][mode] = "PROCESSING"
                save_json(results, run_dir / "results.json")

                # We enumerate the options from 1 to 5 
                options_text = "\n".join([f"{idx+1}. {opt}" for idx, opt in enumerate(item['shuffled_options'])])
                full_question = f"Relationship: [{item['subject']}] -> [{item['relation']}] -> ?\n\nOptions:\n{options_text}"
            
                stats, response_data = pipeline.run(full_question)
                em.add_llm_stats(stats)
                
                ranking = response_data.get("ranking", [])
                # --- Ranking Normalization and Validation ---
                ranking_raw = response_data.get("ranking", [])
                if not isinstance(ranking_raw, list): ranking_raw = []

                # Create a normalization (lowercase) map for comparison without formatting errors
                norm_map = {opt.strip().lower(): opt for opt in item['shuffled_options']}
                ranking = []
                for r_item in ranking_raw:
                    if not isinstance(r_item, str): continue
                    norm_r = r_item.strip().lower()
                    if norm_r in norm_map:
                        ranking.append(norm_map[norm_r])
                
                correct = item['correct_answer']
                
                # Detailed Analysis
                rank_of_correct = (ranking.index(correct) + 1) if correct in ranking else None
                predicted_top_1 = ranking[0] if ranking else None
                
                error_type = None
                if rank_of_correct != 1 and predicted_top_1:
                    error_type = get_error_type(predicted_top_1, item['distractors'])

                results["samples"][i][mode] = {
                    "rank_of_correct": rank_of_correct,
                    "predicted_top_1": predicted_top_1,
                    "error_type": error_type,
                    "ranking": ranking,
                    "reasoning": response_data.get("reasoning", ""),
                    "is_valid_format": len(ranking) == len(item['shuffled_options'])
                }
                
                # Recalculate metrics and Save final result for this step
                if (i + 1) % 1 == 0 or (i + 1) == len(benchmark):
                    # Fair dynamic comparison
                    has_classic = any(s.get("classic_rag") is not None for s in results.get("samples", []))
                    if has_classic:
                        common = [idx for idx, s in enumerate(results["samples"]) 
                                  if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                                     (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"]) and
                                     (isinstance(s.get("classic_rag"), dict) and "ranking" in s["classic_rag"])]
                        modes = ["base", "graph_rag", "classic_rag"]
                    else:
                        common = [idx for idx, s in enumerate(results["samples"]) 
                                  if (isinstance(s.get("base"), dict) and "ranking" in s["base"]) and 
                                     (isinstance(s.get("graph_rag"), dict) and "ranking" in s["graph_rag"])]
                        modes = ["base", "graph_rag"]

                    # Update individual metrics
                    for m_key in modes:
                        results["metrics"][m_key] = calculate_metrics_for_mode(
                            results["samples"], benchmark, m_key, HITS_LIST, common_indices=common)
                    
                    save_json(results, run_dir / "results.json")
                    save_json(metadata, run_dir / "metadata.json")

        # 4. Save and Report (Now correctly indented inside run_link_prediction_eval)
        ExecutionMeasures.calculate_final_times(execution_measures)
        metadata["execution_measures"] = execution_measures
        save_json(metadata, run_dir / "metadata.json")
        
        print("\n=== EVALUATION RESULTS ===")
        for mode in results["metrics"].keys():
            m = results["metrics"].get(mode, {})
            if not m:
                continue
            processed = m.get("processed_samples", 0)
            skipped = m.get("skipped_samples", 0)
            
            if processed == 0 and skipped == 0:
                print(f"\nMode: {mode} - No metrics available (0 samples processed).")
                continue
                
            fair_total = m.get('fair_comparison_samples', 0)
            print(f"\nMode: {mode} (Processed: {processed}, Denominator: {fair_total}, Skipped: {skipped})")
            for k in HITS_LIST:
                val = m.get(f"hits@{k}", 0.0)
                count = m.get(f"hits@{k}_count", 0)
                print(f"  Hits@{k}: {val:.{FLOAT_PRECISION}f} ({count}/{fair_total})")
            print(f"  MRR:    {m.get('mrr', 0.0):.{FLOAT_PRECISION}f}")
            _print_error_dist(m)

        print(f"\nResults and metadata saved in: {run_dir}")

if __name__ == "__main__":
    run_link_prediction_eval()
