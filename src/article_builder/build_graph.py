"""
Builds the initial Knowledge Graph for an article.
Categorizes extracted triplets into themes using LLMs and enriches 
metadata by fetching official publication details from the CrossRef API.
"""

from collections import defaultdict
import datetime
import json
import logging
from string import Template
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

import requests

from ..utils.io import load_txt, load_yaml, load_json, save_json
from ..utils.constants import CONFIG_DIR, PROMPT_DIR, ARTICLE_BUILDER_DIR, SUFFIX_BASE_KG
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import BaseLLM, get_llm_client

def fetch_crossref_metadata(doi: str, api_base_url: str) -> Optional[Dict[str, Any]]:
    """Gets CrossRef metadata for a given DOI."""
    api_url = f"{api_base_url}{doi}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises an exception for 4xx/5xx responses
        data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error contacting API at {api_url}: {e}")
        return None

    # Process the response
    # Metadata
    item = data["message"]
    metadata = {
        "doi": doi,
        "title": item.get("title", [""])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])],
        "journal": item.get("container-title", [""])[0],
        "published": "-".join(str(p) for p in (item.get("issued", {}).get("date-parts", [[]])[0] + [1,1])[:3]),
    }
    
    if not metadata["title"]:
        logging.warning("⚠️ Metadata not found (please complete manually)")
        # Initial empty fields
        metadata = {
            "doi": "",
            "title": "",
            "authors": [],
            "year": "",
            "journal": "",
            "references": {}
        }
    else:
        logging.info(f"📄 Metadata obtained for DOI {doi}")

    # Control published date
    if not "published" in metadata or not metadata["published"]:
        logging.warning(f"Could not parse year from date '{metadata.get('published')}'. Using 'N/A'.")
        metadata["published"] = "N/A"
    
    return metadata

def build_prompt(topic_set: List[str], prompt_template: str) -> str:
    """Constructs the prompt for topic categorization."""
    topics = "\n".join([f"- {t}" for t in topic_set ])
    template = Template(prompt_template)
    prompt = template.substitute(topic_list=topics)
    return prompt

def categorize_topics_by_llm(
    llm_client: BaseLLM, 
    topic_set: List[str], 
    prompt_template: str, 
    retries: int, 
    delay: float
) -> Optional[Dict[str, Any]]:
    """Uses the LLM to group unique triplet topics into broader biological categories."""
    prompt = build_prompt(topic_set, prompt_template)

    # Estimate tokens (approx 4 chars per token)
    est_tokens = int(len(prompt) / 4)
    logging.info(f"  Prompt size: {len(prompt)} chars (~{est_tokens} tokens)")

    # Call to LLM with retries
    response = llm_client.generate(prompt, retries=retries, delay=delay)
    
    try:
        result_json = json.loads(response)
        return result_json
    except (json.JSONDecodeError, TypeError):
        logging.error(f"JSON parsing error in LLM response. Response received: {response}")
        return None # Devuelve una estructura válida vacía

def categorize_triplets(
    triplets: List[Dict[str, Any]], 
    categorized_topics: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Maps each triplet to its assigned category based on its topic."""
    mapping_topic_cat = {}
    for category, topics in categorized_topics.items():
        for t in topics:
            mapping_topic_cat[t] = category
    
    for t in triplets:
        if "triplet_topic" in t and t["triplet_topic"] in mapping_topic_cat:
            t["category"] = mapping_topic_cat[t["triplet_topic"]]
        else:
            t["category"] = t.get("triplet_topic", None)

    return triplets

def sort_keys(items: Dict[str, Any], sorted_keys: List[str]) -> Dict[str, Any]:
    """Reorders dictionary keys according to a specified sequence."""
    # Reorder keys
    sorted_items = { k: items[k] for k in sorted_keys if k in items }
    return sorted_items

def run_graph_building(
    article_id: str, 
    article_data: Dict[str, Any], 
    output_dir: Optional[Path] = None
) -> None:
    """
    Executes the graph building pipeline for a single article.
    """
    logging.info(f"Processing article: {article_id}")
    
    # Load configuration and prompt
    try:
        config_path = CONFIG_DIR / "article_builder/settings.yaml"
        config = load_yaml(config_path)["build_graph"]

        prompt_path = PROMPT_DIR / "article_builder/build_graph.txt" 
        prompt_template = load_txt(prompt_path)
        logging.info(f"Configuration and prompt loaded.")
    except Exception:
        sys.exit(1)
    
    run_settings = config["run_settings"]
    model_settings = config["model_settings"]

    # 2. Define input and output paths
    base_root = output_dir if output_dir else ARTICLE_BUILDER_DIR
    base_output_dir = base_root / article_id
    output_kg_file = base_output_dir / f"kg_{SUFFIX_BASE_KG}.json" # "kg_base.json"
    input_triplet_file = base_output_dir / "extract_triplets" / "triplets.json"

    build_output_dir = base_output_dir / "build_graph"
    build_output_dir.mkdir(parents=True, exist_ok=True)
    output_summary_file = build_output_dir / "summary.json"
    
    input_doi = article_data["doi"]    
    
    # Load triplets and inherit metadata
    triplets_data = load_json(input_triplet_file)
    triplets = triplets_data.get("triplets", [])

    final_kg = {
        "graph": {},
        "metadata": {
            "kg_name": f"{article_id}",
            "kg_creation_date": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            "kg_type": "base",
            "doi": input_doi,
            "execution_measures": triplets_data.get("execution_measures", {}),
        }
    }

    # 3. Initialize LLM client
    llm_client = get_llm_client(model_settings)

    # 4. Execute pipeline with Timer
    with ExecutionMeasures(final_kg["metadata"]["execution_measures"], "graph_construction", module_name="build_graph") as em:
        # Fetching more information by using crossref
        logging.info("Fetching metadata and references...")
        works_api_url = run_settings.get("works_api_url")
        if not works_api_url:
            logging.warning("The 'works_api_url' key is not configured in extract_evidence_config.json")
        final_kg["metadata"].update(fetch_crossref_metadata(input_doi, works_api_url))
        
        logging.info(f"Loaded {len(triplets)} triplets from {input_triplet_file}.")
        
        topics_set = sorted(list({t["triplet_topic"] for t in triplets if "triplet_topic" in t}))
        logging.info(f"Found {len(topics_set)} unique topics to categorize.")

        logging.info("Categorizing topics using the LLM...")
        categorized_topics = categorize_topics_by_llm(llm_client, topics_set, prompt_template, retries=run_settings["llm_retries"]["max_attempts"], delay=run_settings["llm_retries"]["delay_seconds"])
        # 5. Post-processing and saving  
        if not categorized_topics:
            logging.warning(f"Error graph building - categorize_topics_by_llm: {categorized_topics}")
            return None
    
        new_triplets = categorize_triplets(triplets, categorized_topics)    

        categorized_triplets = defaultdict(list)
        triplets_readable = []
        for t in new_triplets:
            category = t.pop("category", "Uncategorized") # Default in case something fails
            t = sort_keys(t, ["subject", "relation", "object", "triplet_topic", "evidence_doi", "evidence_page", "evidence_text"])
            categorized_triplets[category].append(t)
            triplets_readable.append(f"{t['subject']} -> {t['relation']} -> {t['object']} [{category}]")

        logging.info(f"Process finished. {len(categorized_triplets)} categories were created for the graph.")

        final_kg["graph"] = categorized_triplets
        final_kg["metadata"]["triplets_readable"] = triplets_readable

        # Register LLM stats
        em.add_llm_stats(llm_client.get_stats())

    # Calculate final times
    ExecutionMeasures.calculate_final_times(final_kg["metadata"]["execution_measures"])

    # Save graph
    save_json(final_kg, output_kg_file)

    output_data = {
        "total_categories": len(categorized_triplets),
        "total_triplets": len(triplets),
        "categories": list(categorized_triplets.keys()),
        "settings": {
            "prompt": prompt_template,
            "model": model_settings,
            "run": run_settings,
        },
    }
    # Save execution summary
    save_json(output_data, output_summary_file)

    logging.info(f"✔ Graph built and saved to {output_kg_file}.")

if __name__ == '__main__':
    logging.warning("This script is not intended to be run directly. Please use 'main.py'.")