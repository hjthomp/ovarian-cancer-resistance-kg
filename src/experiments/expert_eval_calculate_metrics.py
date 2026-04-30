"""
Calculates agreement metrics and consensus for expert annotations.
Processes CSV evaluations to compute majority voting, distribution of variables,
and Inter-Annotator Agreement (Gwet's AC1 and Fleiss' Kappa).

Note: This script expects annotated CSV files to be located in subfolders of:
data/experiments/expert_evaluation/annotations/
Each subfolder should correspond to a specific evaluation session (timestamp).
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import Counter, defaultdict

from src.utils.io import load_json, save_json
from src.utils.constants import DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Configuration
CALCULATE_GWET_AC1 = True
CALCULATE_FLEISS_KAPPA = True
# Optional: enable statsmodels-based cross-validation (Fleiss' Kappa)
# Requires: numpy, statsmodels
# Disabled by default to avoid warnings if dependencies are missing
# Set to True to enable
USE_STATSMODELS = False
                        
sm_ir = None
if USE_STATSMODELS:
    try:
        import numpy as np
        import statsmodels.stats.inter_rater as sm_ir
        pass
    except ImportError:
        # If libraries are not installed, we gracefully disable the feature
        logging.warning("statsmodels or numpy not found. Cross-validation metrics will be skipped.")
        USE_STATSMODELS = False

# Paths
ANNOTATIONS_DIR = DATA_DIR / "experiments" / "expert_evaluation" / "annotations"
SPLITS_BASE_DIR = DATA_DIR / "experiments" / "expert_evaluation" / "splits"
RESULTS_BASE_DIR = DATA_DIR / "experiments" / "expert_evaluation" / "results"

VALID_TAGS = {
    "coherent": {"YES", "NO"},
    "plausible": {"YES", "NO", "UNKNOWN"},
    "verification": {"OK", "ERROR"}
}


# --- UTILS & HELPERS ---

def get_user_choice(options: List[Any], prompt_text: str) -> Any:
    """Helper to select an option from a list."""
    print(f"\n{prompt_text}")
    for i, option in enumerate(options, 1):
        display = option.relative_to(DATA_DIR) if isinstance(option, Path) else str(option)
        print(f"[{i}] {display}")
    
    while True:
        try:
            choice = int(input(f"Select an option (1-{len(options)}): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def calculate_consensus(votes: List[str], valid_set: Set[str], dispute_label: str = "DISPUTE") -> Tuple[Optional[str], int]:
    """Calculates majority vote. Returns (consensus_value, strength)."""
    clean_votes = [str(v).strip().upper() for v in votes if v and str(v).strip().upper() in valid_set]
    
    if not clean_votes: return None, 0

    counts = Counter(clean_votes)
    top = counts.most_common()
    
    if len(top) > 1 and top[0][1] == top[1][1]:
        return dispute_label, top[0][1]
    
    return top[0][0], top[0][1]

def get_agreement_status(strength: int, num_annotators: int, value: str, dispute_label: str = "DISPUTE") -> str:
    """Categorizes the level of agreement for a specific sample."""
    if value == dispute_label: return "DISPUTE"
    if num_annotators > 1 and strength == num_annotators: return "UNANIMOUS"
    if num_annotators > 1 and strength > num_annotators / 2: return "MAJORITY"
    if num_annotators == 1: return "SINGLE_ANNOTATOR"
    return "WEAK_AGREEMENT"


# --- METRICS LOGIC ---

def calculate_fleiss_kappa(matrix: List[List[int]], n_annotators: int) -> float:
    """
    Computes Fleiss' Kappa for a matrix of counts (N x k).
    N: number of subjects, k: number of categories.
    """
    if not matrix or n_annotators < 2: return 0.0

    n_subjects = len(matrix)
    
    # 1. P_i: Proportion of agreeing pairs for subject i
    #   P_i = (1 / (n*(n-1))) * (sum(c_ij^2) - n)
    p_rows = [(sum(c**2 for c in row) - n_annotators) / (n_annotators * (n_annotators - 1)) for row in matrix]
    p_observed = sum(p_rows) / n_subjects # Avg

    # 2. P_e: Proportion of pairs agreeing by chance
    #   P_j = (1 / (N*n)) * sum(c_ij)
    total_votes = n_subjects * n_annotators
    p_categories = [sum(row[j] for row in matrix) / total_votes for j in range(len(matrix[0]))]
    p_expected = sum(p_j**2 for p_j in p_categories)

    return (p_observed - p_expected) / (1 - p_expected) if p_expected < 1 else 1.0

def calculate_gwet_ac1(matrix: List[List[int]], n_annotators: int) -> float:
    """
    Computes Gwet's AC1 for a matrix of counts (N x k).
    N: number of subjects, k: number of categories.
    Gwet's AC1 is more robust than Kappa when dealing with unbalanced data.
    """
    if not matrix or n_annotators < 2: return 0.0

    n_subjects = len(matrix)
    n_categories = len(matrix[0])

    # 1. p_a: Overall agreement (Same as Fleiss' P_observed avg)
    #   P_i = (1 / (n*(n-1))) * (sum(c_ij^2) - n)
    p_rows = [(sum(c**2 for c in row) - n_annotators) / (n_annotators * (n_annotators - 1)) for row in matrix]
    p_a = sum(p_rows) / n_subjects

    # 2. p_e: Chance agreement for AC1 (Based on category prevalence)
    #   pi_q = (1 / (N*n)) * sum(c_iq)
    #   p_e = (1 / (k-1)) * sum(pi_q * (1 - pi_q))
    # pi_q is the proportion of all ratings that fall into category q
    total_ratings = n_subjects * n_annotators
    pi_q = [sum(row[j] for row in matrix) / total_ratings for j in range(n_categories)]
        
    p_e = (1 / (n_categories - 1)) * sum(p * (1 - p) for p in pi_q)

    return (p_a - p_e) / (1 - p_e) if p_e < 1 else 1.0

def get_kappa_interpretation(k: float) -> str:
    """Landis and Koch (1977) scale."""
    if k < 0: return "Poor agreement"
    if k <= 0.20: return "Slight agreement"
    if k <= 0.40: return "Fair agreement"
    if k <= 0.60: return "Moderate agreement"
    if k <= 0.80: return "Substantial agreement"
    return "Almost perfect agreement"


# --- REPORT GENERATION ---

def generate_text_report(output_path: Path, timestamp: str, data: Dict[str, Any]) -> None:
    """Generates a human-readable summary of the evaluation results."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== EXPERT EVALUATION REPORT: {timestamp} ===\n\n")
        f.write(f"Total Samples Annotated: {data['metadata']['total_samples']}\n")
        f.write("-" * 40 + "\n\n")

        # 1. IAA Block
        f.write("[INTER-ANNOTATOR AGREEMENT (IAA)]\n")
        for var, k_info in data['inter_annotator_agreement'].items():
            f.write(f"- {var.capitalize()}:\n")
            # Gwet's AC1 (Manual) displayed first as per request
            if "gwet_ac1_score" in k_info:
                f.write(f"  * Gwet's AC1 (Manual): {k_info['gwet_ac1_score']:.4f} ({k_info['gwet_ac1_interpretation']})\n")
            if "score" in k_info:
                f.write(f"  * Fleiss' Kappa (Manual): {k_info['score']:.4f} ({k_info['interpretation']})\n")
                if "statsmodels_score" in k_info:
                    f.write(f"  * Fleiss' Kappa (Statsmodels): {k_info['statsmodels_score']:.4f} ({k_info['statsmodels_interpretation']})\n")
            f.write(f"  (Based on {k_info['samples_used']} shared samples)\n")
        f.write("\n")

        # 2. Metrics Block
        def _write_stats_block(title: str, stats: Dict[str, Counter], origin: Optional[str] = None):
            f.write(f"[{title}]\n")
            for var in ["coherent", "plausible", "verification"]:
                counts = stats[var]
                total = sum(counts.values())
                if total == 0: continue
                
                f.write(f"  {var.capitalize()}:\n")
                for val, count in sorted(counts.items()):
                    if val == "DISPUTE": continue
                    f.write(f"    - {val}: {count} ({count/total:.1%})\n")
                
                # Agreement Strength
                status_counts = stats[f"{var}_status"]
                f.write(f"    Agreement Strength:\n")
                for status, count in sorted(status_counts.items()):
                    f.write(f"      * {status}: {count}")
                    if status in ["MAJORITY", "DISPUTE"]:
                        ids = [sid for sid, res in data['details'].items() 
                               if (not origin or res['origin'] == origin) and res[var]['status'] == status]
                        if ids: f.write(f" -> {', '.join(ids[:10])}{'...' if len(ids)>10 else ''}")
                    f.write("\n")
            f.write("\n")

        _write_stats_block("GLOBAL METRICS", data['summary']['global'])
        _write_stats_block("ORIGIN: TOP-N (Core)", data['summary']['by_origin']['top-n'], origin="top-n")
        _write_stats_block("ORIGIN: RANDOM (Tail)", data['summary']['by_origin']['random'], origin="random")


# --- MAIN PROCESS ---

def process_metrics() -> None:
    logging.info("--- Expert Evaluation Metrics Tool ---")

    # 1. Selection
    if not ANNOTATIONS_DIR.exists():
        logging.error(f"Annotations directory not found: {ANNOTATIONS_DIR}")
        print(f"\n[!] ERROR: Annotations directory missing.")
        print(f"Please ensure you have created the following path and added your expert CSVs:")
        print(f"  {ANNOTATIONS_DIR}/<timestamp_folder>/")
        return

    timestamp_dirs = sorted([d for d in ANNOTATIONS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not timestamp_dirs:

        logging.error(f"No annotation subdirectories found in {ANNOTATIONS_DIR}")
        print(f"\n[!] WARNING: No evaluation folders found in:")
        print(f"  {ANNOTATIONS_DIR}")
        print("Make sure to place your annotated CSVs inside a timestamped subfolder.")
        return

    selected_dir = get_user_choice(timestamp_dirs, "Select an evaluation folder to process:")
    timestamp = selected_dir.name

    # 2. Load Registry & Annotations
    registry_path = SPLITS_BASE_DIR / timestamp / "samples_registry.json"
    if not registry_path.exists():
        logging.error(f"Registry not found: {registry_path}")
        return
    
    registry = {str(k): v for k, v in load_json(registry_path).items()}
    csv_files = list(selected_dir.glob("*.csv"))
    logging.info(f"Loading {len(csv_files)} annotation files...")
    
    all_annotations = defaultdict(list)
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if sid := str(row.get("ID", "")):
                    all_annotations[sid].append(row)

    # 3. Compute Consensus and Aggregate
    results, summary_stats, iaa_raw = perform_analysis(all_annotations, registry)
    
    # 4. Compute IAA (Fleiss' Kappa)
    max_annotators = max(len(a) for a in all_annotations.values())
    kappa_results = compute_iaa_metrics(iaa_raw, max_annotators)

    # 5. Save & Export
    output_dir = RESULTS_BASE_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = {
        "metadata": {
            "timestamp": timestamp, "total_samples": len(all_annotations),
            "max_annotators": max_annotators, "files": [f.name for f in csv_files]
        },
        "summary": summary_stats,
        "inter_annotator_agreement": kappa_results,
        "details": results
    }
    
    save_json(final_output, output_dir / "metrics_summary.json")
    generate_text_report(output_dir / "report_summary.txt", timestamp, final_output)

    logging.info(f"Results saved to {output_dir}")
    print(f"\nDone! Report: {output_dir / 'report_summary.txt'}")


def perform_analysis(
    all_annotations: Dict[str, List[Dict[str, Any]]], 
    registry: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[List[int]]]]:
    """Iterates over samples to calculate consensus and global counters."""
    results = {}
    summary_stats = {
        "global": defaultdict(Counter),
        "by_origin": {"top-n": defaultdict(Counter), "random": defaultdict(Counter)}
    }
    iaa_raw = {"coherent": [], "plausible": []}
    max_annotators = max(len(a) for a in all_annotations.values())

    for sample_id, annotations in all_annotations.items():
        reg_info = registry.get(sample_id, {})
        origin = reg_info.get("origin", "unknown")
        n_experts = len(annotations)

        # Extract Votes
        c_votes = [r.get("Coherent? (YES/NO)") for r in annotations]
        p_votes = [r.get("Plausible? (YES/NO/UNKNOWN)") for r in annotations]
        verification_votes = [r.get("Source Verification (OK/ERROR)") for r in annotations 
                              if str(r.get("Source Verification (OK/ERROR)", "")).strip().upper() in VALID_TAGS["verification"]]

        # Consensos
        c_val, c_str = calculate_consensus(c_votes, VALID_TAGS["coherent"])
        p_val, p_str = calculate_consensus(p_votes, VALID_TAGS["plausible"])
        v_val, v_str = calculate_consensus(verification_votes, VALID_TAGS["verification"])

        results[sample_id] = {
            "coherent": {"value": c_val, "strength": c_str, "status": get_agreement_status(c_str, n_experts, c_val)},
            "plausible": {"value": p_val, "strength": p_str, "status": get_agreement_status(p_str, n_experts, p_val)},
            "verification": {"value": v_val, "strength": v_str, "status": get_agreement_status(v_str, len(verification_votes), v_val)},
            "origin": origin, "num_annotators": n_experts, "votes_raw": {"coherent": c_votes, "plausible": p_votes}
        }

        # Update Summary Stats
        for var in ["coherent", "plausible", "verification"]:
            val = results[sample_id][var]["value"]
            if val:
                summary_stats["global"][var][val] += 1
                summary_stats["by_origin"][origin][var][val] += 1
                summary_stats["global"][f"{var}_status"][results[sample_id][var]["status"]] += 1
                summary_stats["by_origin"][origin][f"{var}_status"][results[sample_id][var]["status"]] += 1

        # Collect data for IAA (only full overlap)
        if n_experts == max_annotators and max_annotators > 1:
            for var in ["coherent", "plausible"]:
                votes = c_votes if var == "coherent" else p_votes
                counts = Counter([str(v).strip().upper() for v in votes if str(v).strip().upper() in VALID_TAGS[var]])
                iaa_raw[var].append([counts.get(tag, 0) for tag in sorted(list(VALID_TAGS[var]))])

    return results, summary_stats, iaa_raw


def compute_iaa_metrics(iaa_raw: Dict[str, List[List[int]]], max_annotators: int) -> Dict[str, Any]:
    """Computes IAA metrics (Fleiss Kappa and Gwet AC1) for each variable."""
    kappa_results = {}
    for var in ["coherent", "plausible"]:
        kappa_results[var] = {"samples_used": len(iaa_raw[var])}

        if CALCULATE_FLEISS_KAPPA:
            k_val = calculate_fleiss_kappa(iaa_raw[var], max_annotators)
            kappa_results[var].update({
                "score": round(k_val, 4), "interpretation": get_kappa_interpretation(k_val)
            })
            # Statsmodels note: statsmodels does not have AC1, only Fleiss and G-index.
            if USE_STATSMODELS and iaa_raw[var] and max_annotators > 1:
                try:
                    sm_k = sm_ir.fleiss_kappa(np.array(iaa_raw[var]), method='fleiss')
                    kappa_results[var].update({"statsmodels_score": round(sm_k, 4), "statsmodels_interpretation": get_kappa_interpretation(sm_k)})
                except Exception as e:
                    logging.error(f"Error statsmodels for {var}: {e}")

        if CALCULATE_GWET_AC1:
            ac1_val = calculate_gwet_ac1(iaa_raw[var], max_annotators)
            kappa_results[var].update({
                "gwet_ac1_score": round(ac1_val, 4), "gwet_ac1_interpretation": get_kappa_interpretation(ac1_val)
            })

    return kappa_results


if __name__ == "__main__":
    process_metrics()