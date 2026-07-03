"""
Ablation study for global node normalization. 

It evaluates random ordering and semantic ordering based on AgglomerativeClustering.

It considers:
    - batch composition (number of batches, avg nodes per batch),
    - block semantics (cohesion and variance), 
    - continuous semantics of nodes within blocks (similarity and distance of adjacent nodes).
"""
import logging
import pandas as pd
import numpy as np
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.abstract_augmenter.normalize_graph import get_all_unique_nodes, sort_nodes_semantically, create_dynamic_batches
from src.abstract_augmenter.augment_graph import select_graph
from src.utils.io import load_json, load_txt, save_json, load_yaml
from src.utils.constants import DATA_DIR, PROMPT_DIR, CONFIG_DIR, ABSTRACT_AUGMENTER_DIR

# METRICS
def batch_cohesion(batch, emb_dict):
    """
    Measures semantic cohesion within a batch.

    Defined as the average pairwise cosine similarity between all node embeddings
    inside a batch.

    Higher values indicate that nodes in the batch are semantically closer,
    implying better semantic grouping quality.

    This is a GLOBAL intra-batch similarity metric.
    """
    if len(batch) < 2:
        return 0.0

    sims = []
    for i in range(len(batch)):
        for j in range(i + 1, len(batch)):
            sims.append(np.dot(emb_dict[batch[i]], emb_dict[batch[j]]))

    return float(np.mean(sims))

def batch_variance(batch, emb_dict):
    """
    Measures semantic dispersion within a batch.

    Defined as the average L2 distance of nodes to the batch centroid
    in embedding space.

    Higher values indicate more dispersed (less compact) semantic groups.

    This is a GLOBAL intra-batch dispersion metric.
    """
    vectors = np.array([emb_dict[n] for n in batch])
    if len(vectors) < 2:
        return 0.0

    centroid = np.mean(vectors, axis=0)
    return float(np.mean(np.linalg.norm(vectors - centroid, axis=1)))

def sequential_smoothness(batch, emb_dict):
    """
    Measures how semantically smooth a sequence is.

    Defined as the average cosine similarity between consecutive nodes
    in the ordered batch.

    Higher values indicate smoother semantic transitions,
    meaning adjacent nodes are more semantically similar.

    This metric evaluates LOCAL ordering quality (not global clustering).
    """
    if len(batch) < 2:
        return 0.0

    sims = []
    for i in range(len(batch) - 1):
        sims.append(np.dot(emb_dict[batch[i]], emb_dict[batch[i + 1]]))

    return float(np.mean(sims))

def embedding_jump_distance(batch, emb_dict):
    """
    Measures semantic discontinuity in a sequence.

    Defined as the average L2 distance between consecutive node embeddings.

    Higher values indicate more abrupt semantic changes between adjacent nodes,
    meaning the ordering is less semantically smooth.

    This metric captures LOCAL semantic disruption in the sequence.
    """
    if len(batch) < 2:
        return 0.0

    distances = []
    for i in range(len(batch) - 1):
        v1 = emb_dict[batch[i]]
        v2 = emb_dict[batch[i + 1]]
        distances.append(np.linalg.norm(v1 - v2))

    return float(np.mean(distances))

# EXPERIMENT RUNNER
def run_experiment(sorted_nodes, label, threshold, embeddings_dict, prompt_template, batch_size, num_ctx):
    batches = create_dynamic_batches(
        node_list=sorted_nodes,
        prompt_template=prompt_template,
        max_batch_size=batch_size,
        num_ctx=num_ctx
    )

    batch_sizes = [len(b) for b in batches]
    cohesions = [batch_cohesion(b, embeddings_dict) for b in batches]
    variances = [batch_variance(b, embeddings_dict) for b in batches]
    
    smoothness = [sequential_smoothness(b, embeddings_dict) for b in batches]
    jump_dist = [embedding_jump_distance(b, embeddings_dict) for b in batches]

    result = {
        "setting": label,
        "threshold": threshold,

        "num_batches": len(batches),

        "avg_batch_size": round(np.mean(batch_sizes), 3),
        "std_batch_size": round(np.std(batch_sizes), 3),

        "avg_cohesion": round(np.mean(cohesions), 4),
        "std_cohesion": round(np.std(cohesions), 4),

        "avg_variance": round(np.mean(variances), 4),
        "std_variance": round(np.std(variances), 4),

        "avg_smoothness": round(np.mean(smoothness), 4),
        "std_smoothness": round(np.std(smoothness), 4),

        "avg_jump_distance": round(np.mean(jump_dist), 4),
        "std_jump_distance": round(np.std(jump_dist), 4),
    }

    print({k: result[k] for k in result if k != "batches"})
    return result

def run_ablation_study() -> None:

    # --- 1. Load Configuration ---
    config_path = CONFIG_DIR / "experiments" / "ablation.yaml"
    config = load_yaml(config_path)
    if not config:
        logging.error(f"Could not load configuration from {config_path}. Exiting.")
        exit()

    THRESHOLDS = config.get("thresholds", [])
    EMBEDDING_MODEL = config.get("embedding_model")
    BATCH_SIZE = config.get("batch_size")
    NUM_CTX = config.get("num_ctx")

    PROMPT_TEMPLATE = load_txt(PROMPT_DIR / "abstract_augmenter" / "normalize_graph.txt") or "dummy"

    # --- 2. Select Graph ---
    print("\n--- Ablation Study: Node Sorting Strategies ---")
    base_graph_path = select_graph(ABSTRACT_AUGMENTER_DIR, require_completed=True)
    if not base_graph_path:
        print("No graph selected. Exiting.")
        exit()

    kg_data = load_json(base_graph_path)
    graph = kg_data["graph"]

    all_nodes = get_all_unique_nodes(graph)
    print(f"Total nodes: {len(all_nodes)}")

    print("Computing embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(all_nodes, normalize_embeddings=True)
    embeddings_dict = {n: e for n, e in zip(all_nodes, embeddings)}

    OUTPUT_DIR = DATA_DIR / "experiments" / "ablation"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n=== BASELINE: RANDOM ORDERING ===")
    sorted_nodes_random = all_nodes.copy()
    random.seed(42)
    random.shuffle(sorted_nodes_random)

    results.append(
        run_experiment(
            sorted_nodes_random,
            label="random",
            threshold="none",
            embeddings_dict=embeddings_dict,
            prompt_template=PROMPT_TEMPLATE,
            batch_size=BATCH_SIZE,
            num_ctx=NUM_CTX
        )
    )

    # SEMANTIC THRESHOLDS
    print("\n=== SEMANTIC THRESHOLDS ===")

    for t in THRESHOLDS:

        sorted_nodes = sort_nodes_semantically(
            nodes=all_nodes,
            model_name=EMBEDDING_MODEL,
            distance_threshold=t
        )

        results.append(
            run_experiment(
                sorted_nodes,
                f"agglomerative",
                t,
                embeddings_dict,
                prompt_template=PROMPT_TEMPLATE,
                batch_size=BATCH_SIZE,
                num_ctx=NUM_CTX
            )
        )

    # SAVE OUTPUTS
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    save_json(results, OUTPUT_DIR / "results.json")

    settings = {
        "base_graph_path": str(base_graph_path),
        "embedding_model": EMBEDDING_MODEL,
        "batch_size": BATCH_SIZE,
        "num_ctx": NUM_CTX,
        "thresholds": THRESHOLDS,
        "results_path": str(OUTPUT_DIR / "results.csv"),
        "results_json_path": str(OUTPUT_DIR / "results.json")
    }
    save_json(settings, OUTPUT_DIR / "settings.json")

    print("\nDone.")

if __name__ == "__main__":
    run_ablation_study()