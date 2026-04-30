"""
Core pipeline for GraphRAG (Knowledge Graph Augmented Generation).
Handles graph indexing in ChromaDB, semantic retrieval, re-ranking, 
and LLM-based answering with faithfulness evaluation.
"""

import json
from pathlib import Path
from string import Template
from typing import Dict, Any, List, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils.constants import PROJECT_ROOT
from src.utils.llm_calls import get_llm_client

class GraphRAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pipeline with a configuration dictionary.
        
        Args:
        config (dict): Configuration dictionary.
        """
        self.exp_name = config.get("name", "untitled")
        self.exp_description = config.get("description", "")
        
        # Retrieval configuration
        retrieval = config.get("retrieval", {})
        self.use_retrieval = retrieval.get("enabled", True)

        if self.use_retrieval:
            self.graph_path = PROJECT_ROOT / retrieval["graph_path"]
            self.collection_name = self.graph_path.stem
            self.db_path = PROJECT_ROOT / retrieval["db_path"] / self.collection_name
            self.embedding_model = retrieval.get("embedding_model")
            self.rerank_model_name = retrieval.get("rerank_model")
            self.top_k = retrieval.get("top_k", 30)
            self.rerank_top_k = retrieval.get("rerank_top_k", 100)
            self.use_reranking = retrieval.get("use_reranking", False)
        
        # Prompt template
        self.prompt_template = config.get("prompt_template", "")
        self.use_evidence = config.get("prompt", {}).get("use_evidence", False)
        
        # LLM 
        self.llm = config.get("llm", {})
        
        # Component init
        print(f"\n🚀 Starting GraphRAG Pipeline: {self.exp_name}")
        if self.use_retrieval:
            self.embed_model = SentenceTransformer(self.embedding_model)
            if self.rerank_model_name:
                self.reranker = CrossEncoder(self.rerank_model_name)
            self.collection = self._setup_chroma_db()
            self._index_graph()
        self.llm_client = self._setup_llm()

    def _setup_chroma_db(self) -> Any:
        """Initializes the persistent ChromaDB client and collection."""
        client = chromadb.PersistentClient(path=str(self.db_path))
        return client.get_or_create_collection(self.collection_name)

    def _index_graph(self) -> None:
        """Loads the KG JSON and indexes triplets into the vector database if empty."""
        if self.collection.count() > 0:
            print(f"[DB] Collection '{self.collection_name}' active ({self.collection.count()} triplets).")
            return
        
        print(f"Indexing graph from {self.graph_path}...")
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                kg_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Graph file not found at {self.graph_path}")
            return

        graph = kg_data.get("graph", {})
        documents = []
        metadatas = []
        ids = []
        doc_id = 0

        for theme, triplets in graph.items():
            for t in triplets:
                doc_text = f"Triplet: {t['subject']} -> {t['relation']} -> {t['object']}. Theme: {theme}."
                documents.append(doc_text)
                
                metadata = {
                    "subject": t.get("subject", ""),
                    "relation": t.get("relation", ""),
                    "object": t.get("object", ""),
                    "theme": theme,
                    "source_type": t.get("source_type", "N/A"),
                    "evidence_list_str": json.dumps(t.get("evidence_list", []))
                }
                metadatas.append(metadata)
                ids.append(f"triplet_{doc_id}")
                doc_id += 1

        if documents:
            print(f"Encoding and adding {len(documents)} documents to Vector DB...")
            embeddings = self.embed_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            
            # Batch addition to avoid ChromaDB/SQLite limits (max 5461 vars)
            batch_size = 5000
            for i in range(0, len(documents), batch_size):
                end = i + batch_size
                self.collection.add(
                    embeddings=embeddings[i:end].tolist(),
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                    ids=ids[i:end]
                )
            print("Indexing complete.")

    def _setup_llm(self) -> Any:
        """Configures the LLM client based on pipeline settings."""
        llm_settings = {
            "model_name": self.llm.get("model_name", "alibayram/medgemma:27b"),
            "temperature": self.llm.get("temperature", 0.0),
            "top_p": self.llm.get("top_p", 1.0),
            "top_k": self.llm.get("top_k", 1),
            "num_ctx": self.llm.get("num_ctx", 4096),
            "format": self.llm.get("format", "json")
        }
        return get_llm_client(llm_settings)

    def run(self, question: str, evaluate: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes the full RAG flow: retrieval, re-ranking, prompting, and generation."""
        # 1. Retrieval
        retrieved_metadatas = []
        retrieved_documents = []
        
        if self.use_retrieval:
            print(f"\n[Retrieval] Searching graph for: '{question[:50]}...'")
            q_emb = self.embed_model.encode([question], convert_to_numpy=True)
            
            # Si usamos re-ranking, pedimos más candidatos inicialmente (rerank_top_k)
            fetch_k = self.rerank_top_k if self.use_reranking else self.top_k
            results = self.collection.query(query_embeddings=q_emb.tolist(), n_results=fetch_k)
            
            retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []
            retrieved_documents = results["documents"][0] if results["documents"] else []

            # 1.1 Re-ranking step
            if self.use_reranking and retrieved_metadatas and hasattr(self, 'reranker'):
                # Formatear pares (pregunta, documento) para el Cross-Encoder
                pairs = [[question, doc] for doc in retrieved_documents]
                scores = self.reranker.predict(pairs)
                
                # Combinar, ordenar por score y recortar al top_k final
                scored_results = sorted(zip(scores, retrieved_metadatas), key=lambda x: x[0], reverse=True)
                retrieved_metadatas = [meta for score, meta in scored_results[:self.top_k]]
                
                print(f"[Rerank] Refined {len(scored_results)} candidates down to {len(retrieved_metadatas)} most relevant triplets.")
            else:
                print(f"[Retrieval] Fetched {len(retrieved_metadatas)} context triplets.")
        
        # 2. Prompt Construction
        print(f"[Prompt] Constructing context (Evidence: {self.use_evidence}).")
        context_parts = []
        for i, meta in enumerate(retrieved_metadatas):
            triplet_str = f"ID {i}: Triplet: {meta['subject']} -> {meta['relation']} -> {meta['object']}"
            
            if self.use_evidence:
                try:
                    evidence_list = json.loads(meta.get("evidence_list_str", "[]"))
                    if evidence_list:
                        evidence_texts = [f"  - Evidence: {ev.get('evidence_text', 'N/A')}" for ev in evidence_list]
                        triplet_str += "\n" + "\n".join(evidence_texts)
                except (json.JSONDecodeError, KeyError):
                    pass
            
            context_parts.append(triplet_str)

        context_text = "\n\n".join(context_parts)

        template = Template(self.prompt_template)
        prompt = template.substitute(context_text=context_text, question=question)
        # 3. Generation
        print(f"[LLM] Generating answer...")
        try:
            response = self.llm_client.generate(prompt=prompt)
            data = json.loads(response)
            data["raw_llm_response"] = response
        except Exception as e:
            data = {"answer": f"Error: {str(e)}", "used_triplets": [], "raw_response": str(e)}
        
        # Reconstruct full context for visualization/output
        all_retrieved_triplets = []
        for i, meta in enumerate(retrieved_metadatas):
            full_triplet = meta.copy()
            full_triplet['id'] = i
            try:
                full_triplet['evidence_list'] = json.loads(full_triplet.pop('evidence_list_str', '[]'))
            except (json.JSONDecodeError, KeyError):
                full_triplet['evidence_list'] = []
            all_retrieved_triplets.append(full_triplet)

        # Map used IDs to full triplets
        used_triplets = []
        if "used_triplet_ids" in data and isinstance(data["used_triplet_ids"], list):
            for idx in data["used_triplet_ids"]:
                if isinstance(idx, int) and 0 <= idx < len(all_retrieved_triplets):
                    used_triplets.append(all_retrieved_triplets[idx])

        data["used_triplets"] = used_triplets
        data["context_triplets"] = all_retrieved_triplets

        # 4. Evaluation (Faithfulness)
        if evaluate and "answer" in data:
            print(f"[Eval] Auditing faithfulness score...")
            eval_data = self._evaluate_faithfulness(question, context_text, data["answer"])
            data["evaluation"] = eval_data

        print(f"[Pipeline] Run completed successfully.")
        return self.llm_client.get_stats(), data

    def _evaluate_faithfulness(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """Evaluates if the LLM answer is grounded in the provided graph context."""
        eval_prompt_path = PROJECT_ROOT / "prompts/graph_rag/evaluate_faithfulness.txt"
        
        try:
            if not eval_prompt_path.exists():
                raise FileNotFoundError(f"Evaluation prompt file not found at {eval_prompt_path}")

            template_str = eval_prompt_path.read_text(encoding="utf-8")
            template = Template(template_str)
            eval_prompt = template.substitute(context=context, question=question, answer=answer)
            
            eval_response = self.llm_client.generate(prompt=eval_prompt)
            return json.loads(eval_response)
        except Exception as e:
            return {"faithfulness_score": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
