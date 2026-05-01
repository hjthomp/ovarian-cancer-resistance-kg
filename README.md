# Ovarian Cancer Resistance Knowledge Graph

This repository presents an evidence-based knowledge graph (KG) framework for studying resistance mechanisms in ovarian cancer. It integrates information extraction from scientific literature with large language models (LLMs), combined with human-in-the-loop (HITL) validation, enabling the construction, enrichment, and exploration of structured biomedical knowledge.

## 📋 Table of Contents

-   [Key Features](#-key-features)
-   [Project Architecture](#-project-architecture)
-   [Prerequisites](#-prerequisites)
-   [Installation](#-installation)
-   [Usage and Execution](#-usage-and-execution)
-   [Article Builder Module](#-article-builder-module)
-   [Abstract Augmenter Module](#-abstract-augmenter-module)
-   [Interactive Visualization](#-interactive-visualization)
-   [GraphRAG (Chat with the Graph)](#-graphrag-chat-with-the-graph)
-   [Experiments & Analysis Module](#-experiments--analysis-module)
- [Citation](#-citation)
- [License](#-license)



---

## 🚀 Key Features

*   **Evidence Extraction**: Processes PDFs using `PyMuPDF` and `spaCy` for intelligent sentence segmentation, filtering out tables and references. Uses LLMs to extract evidence from text chunks.
*   **Triplet Generation**: Uses LLMs to extract relationships (Subject, Predicate, Object) with traceability to the source (DOI, page, original text).
*   **Semantic Normalization**: Clustering algorithms (`scikit-learn`) and embeddings (`SentenceTransformers`) to unify similar nodes (e.g., "Cisplatin" vs. "CDDP").
*   **Advanced Visualization**: Interactive web interface based on **D3.js** with community detection, pathfinding (shortest paths), and dynamic filters.
*   **GraphRAG**: A Question & Answering (QA) system grounded in the generated graph, allowing interaction with the KG knowledge.

---

## 📂 Project Architecture

```text
ovarian_cancer_resistance_kg/
├── data/                  # Project data storage
│   ├── raw/               # Raw input data (PDFs and scraped abstracts)
│   └── releases/          # Versioned Knowledge Graph releases
├── src/                   # Main project source code
│   ├── article_builder/       # KG construction module from PDFs
│   ├── abstract_augmenter/    # KG augmenter module with PubMed abstracts
│   ├── experiments/           # Analysis and evaluation module
│   ├── visualization/         # Interactive web interface to explore the KG
│   ├── graph_rag/             # Streamlit RAG chatbot to interact with the KG
│   └── utils/                 # I/O utilities, LLM calls, logging, constants (general paths)
├── config/                # Configuration files for modules
├── prompts/               # LLM prompt templates
├── requirements.txt       # Python dependencies
├── license                # MIT license
└── README.md              # This file
```

---

## ✅ Prerequisites

Ensure that Python (3.9 or higher) and Git are installed.

### Requirement: Ollama
This project uses local language model inference via **Ollama**. Before running the system, ensure that Ollama is installed and the service is running. You can download it from https://ollama.ai.

We recommend using the MedGemma model (https://ollama.com/alibayram/medgemma).
The MedGemma 27B model may require significant computational resources (≈12–24GB of GPU VRAM depending on quantization). However, it can also run in quantized form via Ollama on CPU, with reduced speed.

#### Model setup — pull the MedGemma model:

```bash
ollama pull alibayram/medgemma:27b
```

#### Start service — run Ollama service in the background:

```bash
ollama serve
```

---

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/hjthomp/ovarian-cancer-resistance-kg.git
    cd ovarian_cancer_resistance_kg
    ```

2.  **Create a virtual environment**

    **Windows:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the environment**

    **Windows:**
    ```bash
    .\venv\Scripts\activate
    ```

    **Mac/Linux:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage and Execution

The system is executed through a centralized interactive interface. Ensure that the virtual environment is activated and that Ollama is running (`ollama serve`).

Start the main menu:

```bash
python -m src.main
```
This will open the interactive interface:

```
============================================================
 OVARIAN CANCER KNOWLEDGE GRAPH 
============================================================
  [1] Article Builder (PDFs -> Graph)
  [2] Abstract Augmenter (PubMed -> Augmentation)
  [3] Release Manager (Publish to 'data/releases')
  [4] Experiments & Analysis
  [5] Interactive Visualization (Web)
  [6] GraphRAG Interface (Streamlit)
  ---
  [Q] Exit Application
============================================================
```

Select the corresponding option to execute each module.

---

## 🧩 Article Builder Module

This module (`src/article_builder`) is responsible for constructing the initial KG from full-text PDF articles defined in `config/articles.yaml`. It extracts structured information from scientific articles and integrates it into the KG as the first stage of the pipeline.

```text
===================== MODULE: BUILD_GRAPH ======================
      Articles are loaded from 'config/articles.yaml'

  [1] Create a graph from all articles
  [2] Create a graph from an article
  [3] Unify processed graphs
  [4] Resume interrupted run
  ---
  [R] Return to principal menu
  [Q] Exit
============================================================
```


---

## 🔄 Abstract Augmenter Module

This module (`src/abstract_augmenter`) enriches the existing KG by retrieving, augmenting, and normalizing related abstracts from PubMed.

```text
====================== MODULE: AUGMENT_GRAPH =======================
  PubMed Augmentation Runs
  [1] Start new run from Article Graph
  [2] Start new run from Augmented Graph
  [3] Resume an incomplete run
  [4] Generate a partial graph from an incomplete run
  ---
  Single-Article Augmentation
  [5] Add a single abstract from Article Graph
  [6] Add a single abstract from Augmented Graph
  ---
  Post-processing (only for complete runs)
  [7] Normalize an Augmented Graph
  [8] Refine a Graph
  ---
  [R] Return to principal menu
  [Q] Exit
====================================================================
```

---

## 🌐 Interactive Visualization

This module (`src/visualization`) provides an interactive web interface to explore the generated KGs. It runs entirely in the browser and loads data directly from JSON files, without requiring a backend server.

### Access
1.  Start the main menu (`python -m src.main`).
2.  Select option `[5] Interactive Visualization (Web)`. This will automatically open the visualization in your default browser. Alternatively, you can directly open `visualization/index.html` in your browser.
3. Load a graph from `data/releases/`.

![Tool demo](assets/tool_demo.gif)

---

## 🤖 GraphRAG (Chat with the KG)

This module (`src/graph_rag`) provides an interactive Streamlit-based interface for quering the KG using natural language. It enables graph-aware QA over the extracted biomedical knowledge.

### Access
1.  Ensure Ollama is running (`ollama serve`).
2.  Start the main menu (`python -m src.main`).
3.  Select option `[6] GraphRAG Interface (Streamlit)`. This will launch the Streamlit application in your browser.


![GraphRAG demo](assets/graphrag_demo.gif)


---

## 🔬 Experiments & Analysis Module

This module (`src/experiments`) provides tools for evaluating and analyzing the constructed KG, including experimental pipelines and analysis utilities.

### Access
1.  Start the main menu (`python -m src.main`).
2.  Select option `[4] Experiments & Analysis`. 
3. Select the corresponding option to run the desired experiment.

```
============================================================
 MODULE: EXPERIMENTS 
============================================================
  [1] Calculate Statistics
  [2] Expert Evaluation: Generate Splits
  [3] Expert Evaluation: Calculate Metrics (requires manually annotated data)
  [4] Link Prediction Evaluation
  [5] NER Pipeline (Extraction & Normalization)
  [6] Plot NER Results
  [7] Plot WordClouds
  ---
  [R] Return to principal menu
  [Q] Exit Application
============================================================
```

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{thompson2026,
  title={Title},
  author={Name},
  journal={...},
  year={2026}
}
```

## 📬 Contact
[hjthompson@unsl.edu.ar](mailto:hjthompson@unsl.edu.ar)

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
