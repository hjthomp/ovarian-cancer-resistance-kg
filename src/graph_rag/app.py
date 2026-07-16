"""
Streamlit-based dashboard for interacting with the GraphRAG pipeline.
Allows users to upload KGs, configure retrieval/LLM settings, and visualize 
the resulting graph neighborhoods and evidence.
"""

import gc
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- Path Setup ---
# Ensure rag_pipeline and lib can be imported from the project root
CURRENT_DIR = Path(__file__).parent.resolve()
_project_root_for_path = CURRENT_DIR.parents[1]
sys.path.append(str(_project_root_for_path))

from src.utils.constants import PROJECT_ROOT
from src.graph_rag.rag_pipeline import GraphRAGPipeline

DATA_PATH = PROJECT_ROOT / "data" / "graph_rag"

# --- Load Configuration ---
config_path = PROJECT_ROOT / "config" / "graph_rag" / "settings.yaml"
app_config = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        app_config = yaml.safe_load(f) or {}

DEFAULT_DB_PATH = DATA_PATH / "vectorstores"
DEFAULT_EMBEDDING_MODEL = app_config.get("embedding_model", "all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = app_config.get("llm_model", "alibayram/medgemma:27b")
DEFAULT_TEMPERATURE = app_config.get("temperature", 0.0)
DEFAULT_TOP_P = app_config.get("top_p", 1)
DEFAULT_TOP_K = app_config.get("top_k", 30)
DEFAULT_RERANK_K = app_config.get("rerank_top_k", 100)
DEFAULT_USE_RERANK = app_config.get("use_reranking", True)
DEFAULT_NUM_CTX = app_config.get("num_ctx", 4096)

st.set_page_config(layout="wide", page_title="Reasoning framework for E-KGs")

st.title("Reasoning framework for E-KGs")
st.markdown("Experiment interactively with retrieval, prompts, and LLM configurations using `rag_pipeline.py`.")

# --- Load Prompts (with fallback) ---
prompts_dir = PROJECT_ROOT / "prompts" / "graph_rag"
base_prompt_path = prompts_dir / "default_base.txt"
graph_prompt_path = prompts_dir / "default_graphrag.txt"

# Priority: 1. settings.yaml (config) -> 2. Hardcoded fallback
fallback_base = "Answer the question based on your internal knowledge.\n\nQuestion: $question\nAnswer:"
fallback_rag = "Answer based on context.\n\nContext:\n$context_text\n\nQuestion: $question\nAnswer:"

PROMPT_BASE = base_prompt_path.read_text(encoding="utf-8") if base_prompt_path.exists() else fallback_base
PROMPT_RAG = graph_prompt_path.read_text(encoding="utf-8") if graph_prompt_path.exists() else fallback_rag

if not base_prompt_path.exists() or not graph_prompt_path.exists():
    st.toast(f"Using default prompts (files not found in {prompts_dir})", icon="ℹ️")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader("📂 Upload Graph (.json)", type="json", help="Upload a JSON file exported from the pipeline.")

if uploaded_file is None:
    st.info("👈 Please upload a Knowledge Graph JSON file in the sidebar to start.")
    st.stop()

# Save uploaded file to a temp directory so the pipeline can read it by path
temp_dir = DATA_PATH / "temp_uploads"
temp_dir.mkdir(parents=True, exist_ok=True)

graph_path_obj = temp_dir / uploaded_file.name
with open(graph_path_obj, "wb") as f:
    f.write(uploaded_file.getbuffer())

# Convert to string relative to project root for cleaner logs/config
try:
    graph_path = str(graph_path_obj.relative_to(PROJECT_ROOT)).replace("\\", "/")
except ValueError:
    graph_path = str(graph_path_obj).replace("\\", "/")

with st.sidebar:

    # Embedding Model (Fixed from Config)
    embedding_model = DEFAULT_EMBEDDING_MODEL
    rerank_model_display = app_config.get("rerank_model", "None")
    safe_model_name = embedding_model.replace("/", "_").replace(":", "_")
    db_path = f"{DEFAULT_DB_PATH}/{safe_model_name}"

    # --- INFO & RESET ---
    with st.expander("ℹ️ System Info & Cache", expanded=False):
        st.markdown(f"**Graph:** `{uploaded_file.name}`")
        st.markdown(f"**Embedding:** `{embedding_model}`")
        st.markdown(f"**Rerank Model:** `{rerank_model_display}`")
        st.markdown(f"**LLM:** `{DEFAULT_LLM_MODEL}`")
        st.markdown(f"**DB Path:** `{db_path}`")
        
        if st.button("♻️ Reset Cache", help="Clears the vector DB and reloads configuration.", use_container_width=True):
            st.cache_resource.clear()
            gc.collect()
            
            db_root = PROJECT_ROOT / db_path
            if db_root.exists():
                try:
                    shutil.rmtree(db_root)
                    st.toast("Database successfully deleted.", icon="🗑️")
                except Exception as e:
                    st.error(f"Error deleting DB: {e}")
            
            st.rerun()

    # --- RETRIEVAL SETTINGS ---
    with st.expander("🔍 Retrieval Settings", expanded=True):
        def on_mode_change():
            if st.session_state.pipeline_mode == "GraphRAG":
                st.session_state.prompt_template = PROMPT_RAG
            else:
                st.session_state.prompt_template = PROMPT_BASE

        pipeline_mode = st.radio(
            "Pipeline Mode", 
            ["GraphRAG", "LLM-only"], 
            index=0, 
            key="pipeline_mode", 
            on_change=on_mode_change,
            horizontal=True,
            help="Choose whether to use the graph context (RAG) or just the LLM knowledge."
        )
        use_rag = (pipeline_mode == "GraphRAG")

        if use_rag:
            top_k = st.number_input(
                "Final Context Size (Top-K)", 
                min_value=1, max_value=1000, value=DEFAULT_TOP_K, 
                help="Number of the most relevant triplets that will be included in the final LLM prompt."
            )
            
            use_reranking = st.checkbox("Enable Re-ranking", value=DEFAULT_USE_RERANK, help="Use a Cross-Encoder to refine search results.")
            if use_reranking:
                rerank_top_k = st.number_input(
                    "Initial Retrieval Size", 
                    min_value=top_k, max_value=500, value=DEFAULT_RERANK_K,
                    help="How many candidates to fetch from the vector database before refining them with the re-ranker."
                )
                st.caption(f"🔍 **Two-stage flow:** Fetch {rerank_top_k} → Rerank → Keep top {top_k}")
            
            use_evidence = st.checkbox("Use Evidence", value=False, help="Include evidence text in the context sent to LLM")
            enable_eval = st.checkbox("🛡️ Evaluate Answer Quality (Faithfulness & Relevance)", value=False, help="Run an LLM-as-a-judge pass to evaluate the answer's Faithfulness (groundedness in context) and Relevance (relation to the question).")
        else:
            top_k = DEFAULT_TOP_K
            use_evidence = False
            use_reranking = False
            enable_eval = False

    # --- LLM SETTINGS ---
    with st.expander("🤖 LLM Parameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.01, help="Controls randomness: higher values (closer to 1) make output more creative, lower values (closer to 0) make it more deterministic.")
        top_p = st.slider("Top P", 0.0, 1.0, DEFAULT_TOP_P, 0.01, help="Nucleus sampling: limits token selection to the top cumulative probability. Lower values reduce diversity.")
        llm_top_k = st.slider("Top K (Gen)", 1, 100, DEFAULT_TOP_K, 1, help="Limits the next token selection to the K most likely tokens. Lower values reduce the chance of rare words.")
        
        num_ctx = st.slider("Context Window", 4096, 16384, DEFAULT_NUM_CTX, 4096, help="Determine how much of your conversation the LLMs can remember and use it to generate responses")

# --- Pipeline Initialization ---

@st.cache_resource(show_spinner="Loading Graph & Embeddings...")
def get_base_pipeline(
    graph_p: str, 
    db_p: str, 
    embed_model: str, 
    rerank_model: Optional[str], 
    k: int, 
    rag_enabled: bool
) -> GraphRAGPipeline:
    """Initializes and caches the GraphRAG pipeline core components."""
    config = {
        "name": "Playground",
        "retrieval": {
            "enabled": rag_enabled,
            "graph_path": graph_p,
            "db_path": db_p,
            "embedding_model": embed_model,
            "rerank_model": rerank_model,
            "top_k": k
        },
        "prompt": {"use_evidence": False}, # Placeholder, updated dynamically
        "prompt_template": "", # Placeholder, updated dynamically
        "llm": {} # Placeholder
    }
    return GraphRAGPipeline(config)

# Initialize Base Pipeline
try:
    rerank_model_name = app_config.get("rerank_model")
    pipeline = get_base_pipeline(graph_path, db_path, embedding_model, rerank_model_name, top_k, use_rag)
except Exception as e:
    st.error(f"Error initializing pipeline: {e}")
    st.stop()

# Current LLM Configuration
current_llm_settings = {
    "model_name": DEFAULT_LLM_MODEL,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": llm_top_k,
    "num_ctx": num_ctx,
    "format": "json"
}

# Re-initialize LLM client only if settings changed
if not hasattr(pipeline, '_last_llm_settings') or pipeline._last_llm_settings != current_llm_settings:
    pipeline.llm = current_llm_settings
    pipeline.llm_client = pipeline._setup_llm()
    pipeline._last_llm_settings = current_llm_settings

# --- Main Interface ---

# Initialize prompt in session state if not present
if "prompt_template" not in st.session_state:
    st.session_state["prompt_template"] = PROMPT_RAG

# Prompt Engineering Section (Moved to Main Area)
with st.expander("📝 Prompt Engineering", expanded=False):
    prompt_template = st.text_area("Prompt Template", key="prompt_template", height=250, help="Edit the prompt sent to the LLM. Use $context_text and $question as placeholders.")

# Update dynamic configurations in the pipeline
pipeline.prompt_template = prompt_template
pipeline.use_evidence = use_evidence
pipeline.temperature = temperature
pipeline.top_p = top_p 
pipeline.top_k = top_k
pipeline.use_reranking = use_reranking
pipeline.rerank_top_k = rerank_top_k if use_reranking else top_k
pipeline.num_ctx = num_ctx 


# st.markdown("---")

# --- Input Section (Full Width) ---
st.subheader("💬 Question")
col_input, col_btn = st.columns([6, 1], vertical_alignment="bottom")
with col_input:
    question = st.text_area("💬 Question", height=68, value="What are the mechanisms of cisplatin resistance?", label_visibility="collapsed", placeholder="Type your question here...")
with col_btn:
    run_btn = st.button("🚀 Run", type="primary", use_container_width=True)

if run_btn and question:
    with st.spinner("Running RAG Pipeline..."):
        try:
            start_time = time.time()
            stats, result = pipeline.run(question, evaluate=enable_eval)
            end_time = time.time()
            stats['total_duration'] = end_time - start_time
            logging.info(f"Pipeline execution finished successfully in {stats['total_duration']:.2f}s")
            st.toast("Response generated successfully!", icon="✅")
            st.session_state['last_run'] = result
            st.session_state['stats'] = stats
            st.session_state['last_run_config'] = {'use_evidence': use_evidence}
        except Exception as e:
            st.error(f"Error during execution: {e}")

# --- Results Display ---
if 'last_run' in st.session_state:
    result = st.session_state['last_run']
    stats = st.session_state.get('stats', {})
    
    # st.markdown("---")
    
    st.subheader("🤖 Answer")
    answer_text = result.get("answer")
    if answer_text:
        st.success(answer_text)
    else:
        # Fallback: Si el modelo no devolvió 'answer' (ej. Link Prediction Ranking), 
        # mostramos el JSON omitiendo los datos pesados del grafo.
        st.info("The model returned a structured response:")
        internal_keys = ["used_triplets", "context_triplets", "raw_llm_response", "evaluation"]
        st.json({k: v for k, v in result.items() if k not in internal_keys})
    
    # Display Evaluation Score AFTER the answer
    if "evaluation" in result:
        eval_res = result["evaluation"]
        faith_score = eval_res.get("faithfulness_score")
        faith_reason = eval_res.get("faithfulness_reasoning", "No reasoning provided.")
        relevance_score = eval_res.get("relevance_score")
        relevance_reason = eval_res.get("relevance_reasoning", "No reasoning provided.")

        # Display Faithfulness Score
        if faith_score is not None:
            color = "green" if faith_score >= 0.8 else "orange" if faith_score >= 0.5 else "red"
            st.markdown(f"**🛡️ Faithfulness:** :{color}[{faith_score:.2f}]  \n*{faith_reason}*")
        
        # Display Relevance Score
        if relevance_score is not None:
            color = "green" if relevance_score >= 0.8 else "orange" if relevance_score >= 0.5 else "red"
            st.markdown(f"**🎯 Relevance:** :{color}[{relevance_score:.2f}]  \n*{relevance_reason}*")

    # 2. Split View: Graph vs Evidence
    col_res_graph, col_res_text = st.columns([3, 2], gap="large")
    
    with col_res_graph:
        st.subheader("🕸️ Graph Visualization")
        
        # Legend
        st.markdown(
            """
            <div style="display: flex; gap: 15px; font-size: 0.9em; margin-bottom: 10px;">
                <div><span style="color: #37BE2B; font-weight: bold;">●</span> Evidence (Used)</div>
                <div><span style="color: #97C2FC; font-weight: bold;">●</span> Context (Neighborhood)</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Visualization logic
        nodes, edges = [], []
        node_ids = set()
        context_triplets = result.get("context_triplets", [])
        used_triplets = result.get("used_triplets", [])
        
        # Identify used nodes
        used_tuples = {(u['subject'], u['relation'], u['object']) for u in used_triplets if isinstance(u, dict)}
        used_nodes_set = {n for t in used_tuples for n in (t[0], t[2])}
        
        for t in context_triplets:
            sub, rel, obj = t['subject'], t['relation'], t['object']
            is_used = (sub, rel, obj) in used_tuples
            color = "#37BE2B" if is_used else "#97C2FC"
            
            if sub not in node_ids:
                is_used_node = sub in used_nodes_set
                nodes.append(Node(id=sub, label=sub, size=15, color="#37BE2B" if is_used_node else "#97C2FC"))
                node_ids.add(sub)
            if obj not in node_ids:
                is_used_node = obj in used_nodes_set
                nodes.append(Node(id=obj, label=obj, size=15, color="#37BE2B" if is_used_node else "#97C2FC"))
                node_ids.add(obj)
            edges.append(Edge(source=sub, target=obj, label=rel, color=color, font={'size': 10}))
        
        config_graph = Config(width="100%", height=600, directed=True, physics=True, hierarchy=False)
        if nodes:
            agraph(nodes=nodes, edges=edges, config=config_graph)
        else:
            st.info("No graph data.")

    with col_res_text:
        run_config = st.session_state.get('last_run_config', {})
        was_evidence_used = run_config.get('use_evidence', False)
        
        st.subheader("📚 Evidence & Context")
        tab_ev, tab_ctx = st.tabs(["Used Evidence", "Full Context JSON"])
        
        with tab_ev:
            with st.container(height=500):
                if not was_evidence_used:
                    st.caption("⚠️ Evidence was excluded from the LLM prompt.")

                used_triplets = result.get("used_triplets", [])
                if used_triplets:
                    for i, t in enumerate(used_triplets):
                        if isinstance(t, dict):
                            label = f"**{t.get('subject')}** → *{t.get('relation')}* → **{t.get('object')}**"
                            evidence = t.get('evidence_list', [])
                            original_id = t.get('id', i)
                            
                            if evidence:
                                with st.expander(f"{i+1}. [ID {original_id}] {t.get('subject')} → {t.get('relation')} → {t.get('object')} ({len(evidence)} refs)"):
                                    for j, ev in enumerate(evidence):
                                        st.markdown(f"**{j+1}.** {ev.get('evidence_text', 'No text')}")
                            else:
                                st.markdown(f"{i+1}. [ID {original_id}] {label}")
                else:
                    st.info("No triplets were explicitly used.")
        
        with tab_ctx:
            with st.container(height=500):
                st.json(result.get("context_triplets", []))

    # st.markdown("---")

    # 1. Metrics Row (Moved to bottom)
    st.markdown("### 📊 Execution Stats")
    m1, m2, m3, m4, m5 = st.columns(5)
    n_retrieved = len(result.get("context_triplets", []))
    n_used = len(result.get("used_triplets", []))
    
    m1.metric("⏱️ Duration", f"{stats.get('total_duration', 0):.2f}s")
    m2.metric("📥 Input Tokens", stats.get('mean_input_tokens', 0))
    m3.metric("📤 Output Tokens", stats.get('mean_output_tokens', 0))
    m4.metric("🔗 Retrieved Triplets", n_retrieved)
    m5.metric("✅ Used Triplets", n_used)
