"""
Extracts textual evidence from PDF articles using a multi-step pipeline.
Handles PDF parsing, text cleaning, sentence segmentation (via spaCy), 
semantic chunking, and LLM-based evidence extraction.
"""

import json
import logging
import re
import sys
from string import Template
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import fitz  # PyMuPDF
import spacy

from ..utils.io import load_yaml, load_txt, save_json, load_json
from ..utils.constants import CONFIG_DIR, PROMPT_DIR, ARTICLE_BUILDER_DIR
from ..utils.execution_measures import ExecutionMeasures
from ..utils.llm_calls import BaseLLM, get_llm_client

def clean_text(raw_text: str) -> str:
    """
    Basic text cleaning: Unicode normalization and hyphen fixing.
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", raw_text)

    # Standardize quotes and dashes
    replacements = {
        '\u2010': '-',  # hyphen
        '\u2011': '-',  # non-breaking hyphen
        '\u2012': '-',  # figure dash
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quote
        '\u2019': "'",  # right single quote
        '\u201c': '"',  # left double quote
        '\u201d': '"',  # right double quote
        '\xa0': ' ',    # non-breaking space
        '\u0000': ' '   # NULL
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Fix words broken across lines by hyphen
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Collapse multiple spaces (preserving tabs)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def is_valid_block(text: str) -> bool:
    """
    Determines if a text block is likely valid content (paragraph) or noise (table, header).
    """
    text = text.strip()
    if not text:
        return False

    # 1. Check for explicit Table/Figure headers
    # Matches "Table 1", "Fig. 2", "Figure 3" at the start
    if re.match(r'^(Table|Fig\.?|Figure)\s*\d+', text, re.IGNORECASE):
        return False

    words = re.findall(r'\w+', text)
    n_words = len(words)
    
    if n_words == 0:
        return False

    # 2. Numeric density check
    # If > 50% of tokens are numbers, it's likely a table row or data
    numbers = re.findall(r'\d+', text)
    if len(numbers) > n_words * 0.5:
        return False

    return True

def extract_pages_from_pdf(
    doc: Any, 
    chunk_settings: Dict[str, Any], 
    nlp_model: Any
) -> List[Dict[str, Any]]:
    """
    Iterates through PDF pages, extracts text blocks, cleans them, 
    and segments them into valid sentences.
    """
    margin_top = chunk_settings.get("margin_top", 0.0)
    margin_bottom = chunk_settings.get("margin_bottom", 0.0)
    remove_illegible = chunk_settings.get("remove_illegible_text", False)

    # Keywords that denote the start of references (anchored to start of line/block)
    ref_headers = [
        r'^references[:.]?$', 
        r'^bibliography[:.]?$', 
        r'^literature cited[:.]?$', 
        r'^works cited[:.]?$',
        r'^\d+\.?\s*references[:.]?$' # e.g. "12. References"
    ]
    
    # Check the last 15 pages to avoid false positives in TOC
    num_pages = len(doc)
    start_check_ref = max(0, num_pages - 15)

    pages = []
    stop_parsing = False

    for i, page in enumerate(doc):
        if stop_parsing:
            break
        
        # Define clip rect if margins are set
        clip_rect = None
        if margin_top > 0 or margin_bottom > 0:
            rect = page.rect
            clip_rect = fitz.Rect(rect.x0, rect.y0 + (margin_top * rect.height), rect.x1, rect.y1 - (margin_bottom * rect.height))

        # Extract blocks (x0, y0, x1, y1, "text", block_no, block_type)
        # block_type=0 is text, 1 is image
        blocks = page.get_text("blocks", clip=clip_rect)
        
        page_text_parts = []
        
        for b in blocks:
            if b[6] != 0: continue # Skip images
            
            # Check for references header if we are deep enough in the document
            if i >= start_check_ref:
                raw_text = b[4].strip()
                # Check the first line of the block to handle cases where header and content are merged
                first_line = raw_text.split('\n', 1)[0].strip()
                
                is_ref = False
                for pattern in ref_headers:
                    if re.match(pattern, first_line, re.IGNORECASE):
                        is_ref = True
                        break
                if is_ref:
                    logging.info(f"Stopping PDF parsing at page {i+1} (References section reached: '{first_line}').")
                    stop_parsing = True
                    break

            block_text = clean_text(b[4]) # Basic clean per block
            
            if remove_illegible and not is_valid_block(block_text):
                continue
                
            # Smart Join: Check if we should append with newline or space
            if not page_text_parts:
                page_text_parts.append(block_text)
            else:
                prev_text = page_text_parts[-1]
                # If previous block ends with sentence terminator, it's likely a paragraph break
                if prev_text and prev_text[-1] in ['.', '!', '?']:
                    page_text_parts.append("\n\n" + block_text)
                else:
                    # Otherwise it might be a sentence continuing on next line/column
                    page_text_parts.append(" " + block_text)

        full_page_text = "".join(page_text_parts)
        
        # Extract valid sentences using Spacy
        sentences = extract_valid_sentences(full_page_text, nlp_model)
        
        pages.append({
            "page": i + 1,
            "text": full_page_text,
            "sentences": sentences
        })

    return pages

def extract_valid_sentences(text: str, nlp_model: Any) -> List[str]:
    """Splits text into sentences and filters out malformed ones."""
    doc = nlp_model(text)
    sentences = []
    
    for sent in doc.sents:
        s_text = sent.text.strip()
                
        if not s_text:
            continue
        
        tokens = s_text.split()

        # 1. minimal length
        if len(tokens) < 5:
            continue

        # Low POS diversity (tables lack syntax) → triplets cannot be constructed afterwards!
        pos_types = {t.pos_ for t in sent}
        if len(pos_types) < 3:
            continue

        sentences.append(s_text)

    return sentences

def initialize_spacy_model() -> Any:
    """Loads and returns the spacy model."""
    logging.info("Loading spacy model 'en_core_web_sm'...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("Model 'en_core_web_sm' not found. Downloading it automatically...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    logging.info("Spacy model loaded.")
    return nlp

def create_chunks(
    sentences: List[str], 
    page: int, 
    sent_per_chunk: int, 
    max_chars: int
) -> List[Dict[str, Any]]:
    """
    Groups sentences into chunks based on the number of sentences and the maximum number of characters.
    """
    chunks = []
    i = 0
    chunk_counter = 0
    while i < len(sentences):
        # Make sure sent_per_chunk is at least 1 to avoid infinite loops
        num_sents_to_take = max(1, sent_per_chunk)
        chunk_sents = sentences[i:i + num_sents_to_take]
        chunk_text = " ".join(chunk_sents)

        while len(chunk_text) > max_chars and len(chunk_sents) > 1:
            chunk_sents = chunk_sents[:-1]
            chunk_text = " ".join(chunk_sents)

        chunks.append({
            "chunk_id": f"p{page}_c{chunk_counter}",
            "page": page,
            "text_block": chunk_text
        })
        # Advance the index based on the actual number of sentences added to the chunk
        i += len(chunk_sents)
        chunk_counter += 1
    return chunks
           
# ---- Extract evidence functions (through LLMs) ----
def extract_evidence_item_from_chunk(
    llm_client: BaseLLM, 
    chunk: Dict[str, Any], 
    prompt_template: str, 
    retries: int, 
    delay: float
) -> Optional[List[Dict[str, Any]]]:
    template = Template(prompt_template)
    prompt = template.substitute(text_block = chunk['text_block'])

    # Estimate tokens (approx 4 chars per token)
    est_tokens = int(len(prompt) / 4)
    logging.info(f"  Prompt size: {len(prompt)} chars (~{est_tokens} tokens)")

    # Call to LLM with retries
    response = llm_client.generate(prompt, retries=retries, delay=delay)
    
    if response is None:
        logging.error(f"LLM call returned None for chunk {chunk['chunk_id']}. Skipping.")
        return None

    try:
        evidence_item_json = json.loads(response)
        if not isinstance(evidence_item_json, list):
            evidence_item_json = [evidence_item_json]
        return evidence_item_json
    except json.JSONDecodeError as e:
        # This error occurs if the LLM response is not valid JSON.
        logging.error(f"JSON parsing error for chunk {chunk['chunk_id']}. Details: {e}")
        return None # Return None explicitly on error

# ---- Invoking functions ----
def parse_pdf(
    path_pdf: Path, 
    path_chunks: Path, 
    chunk_settings: Dict[str, Any], 
    nlp_model: Any
) -> List[Dict[str, Any]]:
    doc = fitz.open(path_pdf)
        
    logging.info(f"Parsing PDF: {path_pdf.name}")
    
    # Extract and process pages
    pages = extract_pages_from_pdf(doc, chunk_settings, nlp_model)

    # Create chunks from the pre-validated sentences on each page
    all_chunks = []
    sent_per_chunk = chunk_settings.get("sentences_per_chunk", 10)
    max_chars_per_chunk = chunk_settings.get("max_chars_per_chunk", 2000)
    for p in pages:
        # Sentences are already extracted and cleaned in preprocess_pages
        # The 'sentences' key is guaranteed to exist here because it was added by preprocess_pages
        sentences = p["sentences"]
        chunks = create_chunks(sentences, p["page"], sent_per_chunk, max_chars_per_chunk)
        all_chunks.extend(chunks)

    # Save intermediate chunks (optional)
    save_json(all_chunks, path_chunks)
    logging.info(f"Chunk file created - {len(all_chunks)} chunks saved to {path_chunks}")
    
    return all_chunks

def extract_evidence(
    llm_client: BaseLLM, 
    all_chunks: List[Dict[str, Any]], 
    prompt_template: str, 
    run_settings: Dict[str, Any], 
    temp_file: Optional[Path] = None, 
    em: Optional[ExecutionMeasures] = None
) -> List[Dict[str, Any]]:
    all_evidence = []
    processed_ids = set()

    # 1. Try to load previous progress
    if temp_file and temp_file.exists():
        try:
            logging.info(f"Found temporary file: {temp_file}. Loading progress...")
            temp_data = load_json(temp_file)
            if temp_data:
                all_evidence = temp_data.get("evidence", [])
                processed_ids = set(temp_data.get("processed_ids", []))
                logging.info(f"Resuming: {len(processed_ids)} chunks already processed.")
        except Exception as e:
            logging.warning(f"Error loading temporary file: {e}. Starting from scratch.")

    for idx, chunk in enumerate(all_chunks):
        if chunk["chunk_id"] in processed_ids:
            continue

        logging.info(f"Processing chunk {idx+1} of {len(all_chunks)}...")
        evidence_item = extract_evidence_item_from_chunk(llm_client, chunk, prompt_template, run_settings["llm_retries"]["max_attempts"], run_settings["llm_retries"]["delay_seconds"])
        
        if evidence_item is not None and len(evidence_item) > 0:
            formatted_batch = []
            for e in evidence_item:
                if isinstance(e, str):
                    formatted_batch.append({"evidence_text": e, "evidence_page": chunk['page']})
                elif isinstance(e, dict):
                    e['evidence_page'] = chunk['page']
                    formatted_batch.append(e)
            all_evidence.extend(formatted_batch)
        
        # Save progress
        processed_ids.add(chunk["chunk_id"])
        if temp_file:
            elapsed = em.get_current_duration() if em else 0.0
            save_json({
                "processed_ids": list(processed_ids), 
                "evidence": all_evidence,
                "elapsed_time": elapsed
            }, temp_file)
            
    return all_evidence

def run_extract_evidence(
    article_id: str, 
    article_data: Dict[str, Any], 
    output_dir: Optional[Path] = None
) -> None:
    """
    Executes the evidence extraction pipeline for a single article.
    """
    logging.info(f"Processing article: {article_id}")

    try:
        config_path = CONFIG_DIR / "article_builder" / "settings.yaml"
        config = load_yaml(config_path)["extract_evidence"]
        
        prompt_path = PROMPT_DIR / "article_builder" / "extract_evidence.txt"
        prompt_template = load_txt(prompt_path)
        logging.info(f"Configuration and prompt loaded.")
    except Exception:
        sys.exit(1)
        
    run_settings = config["run_settings"]
    model_settings = config["model_settings"]

    # 2. Define input and output paths
    input_pdf_path = Path(article_data["pdf_path"])
    input_doi = article_data["doi"]
    
    logging.info(f"- pdf_path: {input_pdf_path}")
    logging.info(f"- doi:      {input_doi}")

    # Use provided output_dir (run specific) or fallback to default constant
    base_dir = output_dir if output_dir else ARTICLE_BUILDER_DIR
    
    article_dir = base_dir / article_id
    evidence_dir = article_dir / "extract_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # kg_file = article_dir / f"kg_{input_pdf_path.stem}.json"
    chunks_file = evidence_dir / "chunks.json"
    evidence_file = evidence_dir / "evidence.json"
    temp_file = evidence_dir / "temp_progress.json"

    # Initialize models
    nlp_model = initialize_spacy_model()

    # Initialize metadata for the pipeline
    execution_measures = {}

    # Check for previous execution time to resume timer correctly
    previous_duration = 0.0
    if temp_file.exists():
        try:
            t_data = load_json(temp_file)
            if t_data:
                previous_duration = t_data.get("elapsed_time", 0.0)
        except Exception:
            pass

    # ***Execute pipeline with Timer***
    with ExecutionMeasures(execution_measures, "evidence_extraction", module_name="extract_evidence", initial_duration=previous_duration) as em:
        logging.info("Parsing PDF to chunks...")

        all_chunks = parse_pdf(input_pdf_path, chunks_file, run_settings, nlp_model)
        
        # Limit chunks for testing if configured
        n_chunks_to_process = run_settings.get("chunks_to_process_limit")
        if n_chunks_to_process is not None and n_chunks_to_process > 0:
            all_chunks = all_chunks[:n_chunks_to_process]
            logging.info(f"Limiting processing to the first {n_chunks_to_process} chunks.")

        llm_client = get_llm_client(model_settings)
        logging.info(f"Extracting evidence from {len(all_chunks)} chunks...")
        all_evidence = extract_evidence(llm_client, all_chunks, prompt_template, run_settings, temp_file, em=em)

        # Add doi to all evidence items (for traceability)
        for e in all_evidence:
            e["evidence_doi"] = input_doi

        # Register LLM stats in the timer
        em.add_llm_stats(llm_client.get_stats())
        
    output_data = {
        "execution_measures": execution_measures,
        "total_evidence": len(all_evidence),
        "evidence_list": all_evidence,
        "settings": {
            "prompt": prompt_template,
            "model": model_settings,
            "run": run_settings
        }
    }
    # Save evidence
    save_json(output_data, evidence_file)

    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()

    logging.info(f"✔ Evidence extraction finished for {article_id}")
    logging.info(f"  Evidence file created - {len(all_evidence)} evidence items saved to: {evidence_file}")

if __name__ == "__main__":
    logging.warning("This script is not intended to be run directly. Please use 'main.py'.")