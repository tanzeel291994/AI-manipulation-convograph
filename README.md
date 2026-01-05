# GNN-HGT: Auditable Hybrid GNN for Retrieval and Ranking

This repository implements a multi-hop retrieval and ranking pipeline using a **Heterogeneous Graph Transformer (HGT)**. The system is designed for complex question-answering tasks (like MuSiQue) where information is spread across multiple documents. It combines dense retrieval with graph-based reasoning to provide high-precision ranking and auditable evidence extraction.

## 🚀 Overview

The pipeline consists of:
1.  **Global Knowledge Graph Construction**: Building a graph of entities, sentences, and passages.
2.  **Self-Supervised Pre-training**: Learning structural representations without labels.
3.  **Subgraph Extraction**: Using Beam Search to find relevant context windows.
4.  **Supervised Ranking**: Fine-tuning the GNN to identify supporting evidence.
5.  **Evaluation**: End-to-end metrics for recall and precision.

## 📁 Core Components

### 1. KG Construction (`construct_kg.py`)
Builds a global Heterogeneous Knowledge Graph from a document corpus.
-   **Nodes**: `entity`, `sentence`, `passage`.
-   **Edges**: `entity-re-entity` (OpenIE triples), `entity-in-sentence`, `sentence-in-passage`.
-   **Embeddings**: Uses `BAAI/bge-small-en-v1.5` for node features.

### 2. Self-Supervised Pre-training (`PretrainGNN.py`)
Pre-trains the `PretrainableHeteroGNN` using contrastive learning (**NT-Xent loss**).
-   Uses graph augmentations (node masking and edge dropping).
-   Encourages the model to learn semantic and structural relationships in the graph before seeing any labels.

### 3. Subgraph Extraction (`BeamSearchExtractor.py`)
Implements a multi-hop **Beam Search** over the Global KG to retrieve candidate subgraphs.
-   Starts with entity-based retrieval using query embeddings.
-   Expands through graph edges to find relevant neighboring entities and their parent passages.
-   Aligns with the query space using raw embeddings for high-recall retrieval.

### 4. Ranking Fine-tuning (`train_ranking.py`)
Fine-tunes the pre-trained GNN for supervised ranking using **MarginRankingLoss**.
-   **Hard Negative Mining**: Compares positive passages against the hardest negatives in the retrieved subgraph.
-   **Hybrid Architecture**: Combines GNN-based message passing with query-specific interaction layers.
-   Loads backbone weights from the pre-training stage.

### 5. Evaluation Pipeline (`EvalPipeline.py`)
Measures the end-to-end performance of the retrieval and reranking system.
-   Calculates **Recall@K** for the Beam Search retrieval.
-   Calculates **Hit@1**, **Recall@5**, and **Recall@10** for the GNN Reranker.

## 📊 Dataset

The system is trained and evaluated on the **MuSiQue** dataset:
-   **Training**: 2,000 samples used for fine-tuning the ranking head.
-   **Evaluation**: 1,000 samples used for pipeline assessment.
-   **Corpus**: Multi-hop documents processed into a global KG.

## 🛠️ Installation & Setup

This project uses **Poetry** for dependency management.

```bash
# Install dependencies
poetry install

# Setup Jupyter kernel (optional)
poetry run python -m ipykernel install --user --name gnn-hgt --display-name "Python (gnn-hgt)"
```

## 🏃 Workflow

To run the full pipeline:

1.  **Build the Graph**:
    ```bash
    python construct_kg.py
    ```
2.  **Pre-train the Model**:
    ```bash
    python PretrainGNN.py
    ```
3.  **Generate Ranking Data**:
    ```bash
    python generate_training_data.py
    ```
4.  **Fine-tune for Ranking**:
    ```bash
    python train_ranking.py
    ```
5.  **Evaluate**:
    ```bash
    python EvalPipeline.py
    ```

## 🧠 Model Architecture

-   **Backbone**: `HGTConv` (Heterogeneous Graph Transformer) for multi-relational message passing.
-   **Global Reasoning**: `SGFormer` layer to capture long-range entity dependencies.
-   **Interaction**: `EnhancedQueryInteraction` for fusing query embeddings with passage/entity representations.
-   **Scoring**: Multi-layer MLP for final relevance calculation.
