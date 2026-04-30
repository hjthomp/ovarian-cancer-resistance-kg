# Ovarian Cancer Resistance Knowledge Graph (KG)

This project implements a complete pipeline for the construction, enrichment, analysis, and visualization of a Knowledge Graph (KG) focused on resistance mechanisms in ovarian cancer.

The system extracts information from full-text scientific articles (PDFs) and PubMed abstracts using local Large Language Models (LLMs) via **Ollama**, normalizes entities, and provides interactive interfaces for exploration and biological pathway discovery.

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
- [Contact](#-contact)



---

## 🚀 Key Features

*   **Evidence Extraction**: Processes PDFs using `PyMuPDF` and `spaCy` for intelligent sentence segmentation, filtering out tables and references. Uses LLMs to extract evidence from text chunks.
*   **Triplet Generation**: Uses LLMs to extract relationships (Subject, Predicate, Object) with traceability to the source (DOI, page, original text).
*   **Semantic Normalization**: Clustering algorithms (`scikit-learn`) and embeddings (`SentenceTransformers`) to unify similar nodes (e.g., "Cisplatin" vs. "Cisplatino").
*   **Advanced Visualization**: Interactive web interface based on **D3.js** with community detection, pathfinding (shortest paths), and dynamic filters.
*   **GraphRAG**: A Question & Answering (QA) system grounded in the generated graph, allowing interaction with the KG knowledge.

---

## 📂 Project Architecture

```text
ovarian_cancer_resistance_kg/
├── data/                  # Project data storage
│   ├── raw/               # Raw input data (PDFs and scraped abstracts)
│   ├── releases/          # Versioned Knowledge Graph releases
│   └── experiments/       # Evaluation results and analysis data
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
└── README.md              # This file
```

---

## ✅ Prerequisites

Ensure you have Python (version 3.9 or higher) and `git` installed.

### 2.0. Requirement: Ollama
This project uses local language models via Ollama. Before running the scripts, ensure that Ollama is installed and the service is active. You can download it from https://ollama.ai.

#### Model Setup - Pull the MedGemma model:

```bash
ollama pull alibayram/medgemma:27b
```

#### Start Service - Run the Ollama service in the background:

```bash
ollama serve
```

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_user/ovarian_cancer_resistance_kg.git
    cd ovarian_cancer_resistance_kg
    ```

2.  **Create a virtual environment:**
    This isolates the project libraries. Open the terminal in the project root folder:

    **On Windows:**
    ```bash
    python -m venv venv
    ```

    **On Mac/Linux:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the environment:**
    Do this every time you open a new terminal to work on the project.

    **On Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
    *(You should see `(venv)` at the beginning of the line).*

    **On Mac/Linux:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    With the environment active, install the necessary libraries:

    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage and Execution

**IMPORTANT:** Always run commands from the **project root** (the folder containing this README) so that Python can find all files.

The project is executed via a main interactive menu. Ensure your virtual environment is active and Ollama is running (`ollama serve`).

```bash
python -m src.main
```

Once executed, a menu with the following options will be presented:

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

Select the number corresponding to the functionality you wish to execute.

---

## 🧩 Article Builder Module

This module (`src/article_builder`) manages the initial construction of the Knowledge Graph from full-text PDF articles defined in `config/articles.yaml`.

To access:
1.  Start the main menu (`python -m src.main`).
2.  Select option `[1] Article Builder (PDFs -> Graph)`. You will see the following sub-menu:

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

This module (`src/abstract_augmenter`) enriches the existing Knowledge Graph by fetching and processing related abstracts from PubMed.

To access:
1.  Start the main menu (`python -m src.main`).
2.  Select option `[2] Abstract Augmenter (PubMed -> Augmentation)`. You will see the following sub-menu:

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

This module (`visualization/`) provides an interactive web interface to explore the generated graphs. It does not require an additional backend server, as it loads data directly from JSON files.

To access:
1.  Start the main menu (`python -m src.main`).
2.  Select option `[5] Interactive Visualization (Web)`. This will automatically open the visualization in your default browser.
3.  Alternatively, you can directly open `visualization/index.html` in your browser and load a graph from `data/releases/`.

---

## 🤖 GraphRAG (Chat with the Graph)

This module (`src/graph_rag/app.py`) offers an interactive Streamlit-based chatbot to perform natural language queries about the knowledge extracted from the graph.

To access:
1.  Ensure Ollama is running (`ollama serve`).
2.  Start the main menu (`python -m src.main`).
3.  Select option `[6] GraphRAG Interface (Streamlit)`. This will launch the Streamlit application and open it in your browser.

---

## 🔬 Experiments & Analysis Module

This module (`src/experiments`) contains various tools to analyze and evaluate the knowledge graph.

To access:
1.  Start the main menu (`python -m src.main`).
2.  Select option `[4] Experiments & Analysis`. You will be presented with a sub-menu:

```
============================================================
 MODULE: EXPERIMENTS 
============================================================
  [1] Calculate Statistics
  [2] Expert Evaluation: Generate Splits
  [3] Expert Evaluation: Calculate Metrics
  [4] Link Prediction Evaluation
  [5] NER Pipeline (Extraction & Normalization)
  [6] Plot NER Results
  [7] Plot WordClouds
  ---
  [R] Return to principal menu
  [Q] Exit Application
============================================================
```
Select the number corresponding to the experiment you wish to execute.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{yourpaper2026,
  title={Your Paper Title},
  author={Your Name},
  journal={...},
  year={2026}
}
```

## 📬 Contact
[yourname@university.edu](mailto:yourname@university.edu)
