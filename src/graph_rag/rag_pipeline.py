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
            
            # If re-ranking is used, more candidates are initially requested (rerank_top_k)
            fetch_k = self.rerank_top_k if self.use_reranking else self.top_k
            results = self.collection.query(query_embeddings=q_emb.tolist(), n_results=fetch_k)
            
            retrieved_metadatas = results["metadatas"][0] if results["metadatas"] else []
            retrieved_documents = results["documents"][0] if results["documents"] else []

            # 1.1 Re-ranking step
            if self.use_reranking and retrieved_metadatas and hasattr(self, 'reranker'):
                # Format pairs (question, document) for the Cross-Encoder
                pairs = [[question, doc] for doc in retrieved_documents]
                scores = self.reranker.predict(pairs)
                
                # Combine, sort by score, and trim to the final top_k
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
            print(f"[Eval] Auditing Faithfulness and Relevance scores...")
            faithfulness_data = self._evaluate_faithfulness(question, context_text, data["answer"])
            relevance_data = self._evaluate_relevance(question, data["answer"])
            
            # Merge evaluation results with distinct keys to avoid overwriting 'reasoning'
            data["evaluation"] = {
                "faithfulness_score": faithfulness_data.get("faithfulness_score"), "faithfulness_reasoning": faithfulness_data.get("reasoning"),
                "relevance_score": relevance_data.get("relevance_score"), "relevance_reasoning": relevance_data.get("reasoning")
            }

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

    def _evaluate_relevance(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluates if the LLM answer is relevant to the original question."""
        eval_prompt_path = PROJECT_ROOT / "prompts/graph_rag/evaluate_relevance.txt"
        
        try:
            if not eval_prompt_path.exists():
                raise FileNotFoundError(f"Evaluation prompt file not found at {eval_prompt_path}")

            template_str = eval_prompt_path.read_text(encoding="utf-8")
            template = Template(template_str)
            # Note: Context is not needed for relevance
            eval_prompt = template.substitute(question=question, answer=answer)
            
            eval_response = self.llm_client.generate(prompt=eval_prompt)
            return json.loads(eval_response)
        except Exception as e:
            return {"relevance_score": 0.0, "reasoning": f"Relevance evaluation failed: {str(e)}"}


class ClassicRAGPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classic flat-text RAG pipeline.
        
        Args:
        config (dict): Configuration dictionary.
        """
        self.exp_name = config.get("name", "untitled")
        self.exp_description = config.get("description", "")
        
        # Retrieval configuration
        retrieval = config.get("retrieval", {})
        self.use_retrieval = retrieval.get("enabled", True)

        if self.use_retrieval:
            self.collection_name = "flat_text_documents"
            self.db_path = PROJECT_ROOT / retrieval["db_path"] / self.collection_name
            self.embedding_model = retrieval.get("embedding_model")
            self.top_k = retrieval.get("top_k", 15)
            
            data_sources = config.get("data_sources", {})
            self.abstracts_file = PROJECT_ROOT / data_sources.get("abstracts_file", "")
            self.articles_runs_dir = PROJECT_ROOT / data_sources.get("articles_runs_dir", "")
        
        # Prompt template
        self.prompt_template = config.get("prompt_template", "")
        
        # LLM 
        self.llm = config.get("llm", {})
        
        # Component init
        print(f"\n🚀 Starting Classic RAG Pipeline: {self.exp_name}")
        if self.use_retrieval:
            self.embed_model = SentenceTransformer(self.embedding_model)
            self.collection = self._setup_chroma_db()
            self._index_documents()
        self.llm_client = self._setup_llm()

    def _setup_chroma_db(self) -> Any:
        """Initializes the persistent ChromaDB client and collection."""
        client = chromadb.PersistentClient(path=str(self.db_path))
        return client.get_or_create_collection(self.collection_name)

    def _index_documents(self) -> None:
        """Loads and chunks source documents (abstracts and articles) and indexes them."""
        if self.collection.count() > 0:
            print(f"[DB] Collection '{self.collection_name}' active ({self.collection.count()} chunks).")
            return
        
        print("Indexing raw text documents (abstracts & articles)...")
        documents = []
        metadatas = []
        ids = []
        doc_id = 0

        # 1. Load PubMed Abstracts
        if self.abstracts_file.exists() and self.abstracts_file.is_file():
            print(f"Loading abstracts from {self.abstracts_file}...")
            try:
                with open(self.abstracts_file, "r", encoding="utf-8") as f:
                    pubmed_data = json.load(f)
                
                results = pubmed_data.get("results", {})
                for key, val in results.items():
                    abstract_text = val.get("abstract", "")
                    if abstract_text and isinstance(abstract_text, str) and abstract_text.strip():
                        meta = {
                            "source_type": "abstract",
                            "doi": val.get("doi") or "",
                            "title": val.get("title") or "",
                            "year": val.get("year") or ""
                        }
                        meta_clean = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                        
                        documents.append(abstract_text.strip())
                        metadatas.append(meta_clean)
                        ids.append(f"abstract_{doc_id}")
                        doc_id += 1
            except Exception as e:
                print(f"Error reading abstracts file: {e}")
        else:
            print(f"Warning: Abstracts file not found or is invalid at {self.abstracts_file}")

        # 2. Load Article Chunks
        if self.articles_runs_dir.exists() and self.articles_runs_dir.is_dir():
            print(f"Loading article chunks from {self.articles_runs_dir}...")
            try:
                for paper_dir in self.articles_runs_dir.iterdir():
                    if paper_dir.is_dir():
                        chunks_file = paper_dir / "extract_evidence" / "chunks.json"
                        if chunks_file.exists():
                            with open(chunks_file, "r", encoding="utf-8") as f:
                                chunks_data = json.load(f)
                            
                            for chunk in chunks_data:
                                text_block = chunk.get("text_block", "")
                                if text_block and isinstance(text_block, str) and text_block.strip():
                                    meta = {
                                        "source_type": "article_chunk",
                                        "paper": paper_dir.name,
                                        "chunk_id": chunk.get("chunk_id") or "",
                                        "page": int(chunk.get("page") or 0)
                                    }
                                    meta_clean = {k: (v if v is not None else "") for k, v in meta.items()}
                                    
                                    documents.append(text_block.strip())
                                    metadatas.append(meta_clean)
                                    ids.append(f"article_{doc_id}")
                                    doc_id += 1
            except Exception as e:
                print(f"Error reading article chunks: {e}")
        else:
            print(f"Warning: Articles runs directory not found or is invalid at {self.articles_runs_dir}")

        # 3. Add to ChromaDB
        if documents:
            print(f"Encoding and adding {len(documents)} text chunks to Vector DB...")
            embeddings = self.embed_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            
            # Batch addition
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
        else:
            print("No documents found to index.")

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

    def run(self, question: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Executes the classic RAG flow: retrieval, flat prompting, and generation."""
        retrieved_documents = []
        
        # 1. Retrieval
        if self.use_retrieval:
            print(f"\n[Classic Retrieval] Searching text for: '{question[:50]}...'")
            q_emb = self.embed_model.encode([question], convert_to_numpy=True)
            results = self.collection.query(query_embeddings=q_emb.tolist(), n_results=self.top_k)
            retrieved_documents = results["documents"][0] if results["documents"] else []
            print(f"[Classic Retrieval] Fetched {len(retrieved_documents)} context chunks.")
            
        # 2. Prompt Construction
        context_text = "\n\n".join(retrieved_documents)
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

        # Backfill standard keys for compatibility with the evaluation logic
        data["used_triplets"] = []
        data["context_triplets"] = []
        
        return self.llm_client.get_stats(), data
