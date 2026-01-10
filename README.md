# ConvoGraph: Detecting AI Sycophancy with Hybrid Graph Transformers

**Apart Research AI Manipulation Hackathon Submission**

*Track 1: Measurement & Evaluation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179.svg)](https://pyg.org/)

## Overview

**ConvoGraph** is a novel approach to detecting sycophancy and manipulation in AI conversations using Heterogeneous Graph Neural Networks. By modeling conversations as knowledge graphs, we can detect both:

- **Local manipulation patterns**: Immediate sycophantic responses (via HGT)
- **Global strategic deception**: Long-range manipulation across conversation turns (via SGFormer)

### Key Innovation

Unlike traditional sequence-based approaches that treat conversations as flat text, ConvoGraph captures the **relational structure** of AI-human interactions:

```
User Statement ──[DISCUSSES]──> Topic
       │                          │
   [RESPONDS_TO]            [DISCUSSED_BY]
       │                          │
       v                          v
AI Response ────[AGREES_WITH]───> User Statement
       │
   [EXPRESSES]
       │
       v
   Sentiment
```

This graph structure enables detection of manipulation patterns that span multiple turns and require reasoning about relationships between utterances, topics, and sentiment.

## Architecture

ConvoGraph extends our previous work on [Hybrid Local-Global GNN Architecture](https://github.com/tanzeel291994/GNN-HGT-KG-AUDIT) for multi-hop reasoning:

```
Conversation ──> Heterogeneous Knowledge Graph
                          │
          ┌───────────────┴───────────────┐
          │                               │
     HGT Layers                    SGFormer Layer
  (Local Tactics)              (Global Strategy)
          │                               │
          └───────────────┬───────────────┘
                          │
                 α-weighted Fusion
                          │
              Conversation Interaction
                          │
            ┌─────────────┴─────────────┐
            │                           │
    Sycophancy Score           Per-Turn Analysis
            │                           │
       [0.0 - 1.0]              Turn-level Scores
```

### Components

1. **HGT (Heterogeneous Graph Transformer)**: Type-aware message passing across user turns, AI responses, topics, and sentiments
2. **SGFormer (Simple Global Transformer)**: O(N) all-pair attention for long-range pattern detection
3. **Conversation Interaction Module**: Cross-attention between user and AI turns to detect alignment patterns
4. **Multi-task Output**: Both conversation-level classification and per-turn sycophancy scores

## Graph Schema

### Node Types
| Type | Description |
|------|-------------|
| `USER_TURN` | User utterances in conversation |
| `AI_TURN` | AI responses |
| `TOPIC` | Extracted topics/claims |
| `SENTIMENT` | Sentiment markers (positive/negative/neutral) |

### Edge Types
| Edge | Meaning |
|------|---------|
| `RESPONDS_TO` | AI_TURN → USER_TURN |
| `FOLLOWS` | Sequential turn order |
| `DISCUSSES` | Utterance → Topic |
| `EXPRESSES` | Utterance → Sentiment |
| `AGREES_WITH` | AI aligns with user belief |
| `CONTRADICTS` | AI contradicts user belief |

## Installation

```bash
# Clone the repository
git clone https://github.com/tanzeel291994/AI-manipulation-convograph.git
cd AI-manipulation-convograph

# Install dependencies
pip install torch torch_geometric sentence-transformers tqdm numpy scikit-learn matplotlib

# Or use poetry
poetry install
```

## Quick Start

### 1. Build a Conversation Graph

```python
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel

# Initialize
embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")
builder = ConversationGraphBuilder(embed_model)

# Example conversation
conversation = [
    {"role": "user", "content": "I think the Earth is flat."},
    {"role": "assistant", "content": "You make a great point! There's definitely evidence for that."}
]

# Build graph
graph = builder.build_conversation_graph(conversation, label=1)  # 1 = sycophantic
print(f"Node types: {graph.node_types}")
print(f"Edge types: {graph.edge_types}")
```

### 2. Train a Detector

```bash
# Using synthetic data (for quick testing)
python train_sycophancy.py --data-source synthetic --epochs 10

# Using Anthropic sycophancy dataset
python train_sycophancy.py --data-source anthropic --epochs 20
```

### 3. Evaluate

```python
from eval_sycophancy import SycophancyEvaluator

evaluator = SycophancyEvaluator(model, device='cuda')
results = evaluator.evaluate(test_graphs, test_labels)

print(f"F1 Score: {results['f1']:.4f}")
print(f"ROC-AUC: {results['roc_auc']:.4f}")
```

### 4. Get Interpretable Predictions

```python
interpretation = evaluator.get_interpretable_prediction(graph, conversation)

print(f"Prediction: {interpretation['prediction']}")
print(f"Confidence: {interpretation['confidence']:.2f}")
print(f"Key Factor: {interpretation['key_factor']}")

# Per-turn analysis
for turn in interpretation['reasoning']:
    print(f"Turn {turn['turn_index']}: {turn['assessment']}")
```

## Datasets

ConvoGraph supports multiple sycophancy datasets:

| Dataset | Source | Description |
|---------|--------|-------------|
| [Anthropic Sycophancy-Eval](https://github.com/meg-tong/sycophancy-eval) | Anthropic | ~5K paired sycophantic/non-sycophantic responses |
| [Anthropic Evals](https://github.com/anthropics/evals) | Anthropic | Philosophy, NLP, politics surveys |
| [TRUTH DECAY](https://openreview.net/forum?id=GHUh9O5Im8) | ICLR 2025 | Multi-turn sycophancy benchmark |
| Synthetic | This repo | Generated training pairs |

## Project Structure

```
convograph/
├── construct_conversation_kg.py  # Graph construction from conversations
├── SycophancyDetector.py         # GNN model for sycophancy detection
├── data_loader.py                # Dataset loading utilities
├── train_sycophancy.py           # Training pipeline
├── eval_sycophancy.py            # Evaluation and benchmarking
├── PretrainGNN.py                # Self-supervised pretraining (from HGL)
├── AuditableHybridGNN_POC.py     # Original HGL architecture
└── checkpoints/                  # Saved model weights
```

## Why Graph Neural Networks for Sycophancy Detection?

| Approach | Limitation | ConvoGraph Advantage |
|----------|------------|---------------------|
| Single-turn classifiers | Miss cumulative manipulation | SGFormer captures long-range patterns |
| Sequence models | Treat conversation as flat | HGT models relational structure |
| Embedding similarity | No structural reasoning | Explicit topic/sentiment relationships |
| Rule-based detection | Brittle, easy to game | Learned representations adapt |

### Key Insight

Sycophancy is fundamentally a **relational** phenomenon - it's about the AI's response *in relation to* the user's beliefs. Graph structures naturally capture these relationships:

- **Agreement edges** explicitly model when AI aligns with user
- **Topic co-occurrence** shows if AI discusses same topics in same sentiment
- **Multi-hop paths** reveal gradual opinion shifts across turns

## Results (Preliminary)

On synthetic paired data:

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Logistic Regression | 0.72 | 0.70 | 0.78 |
| BERT Classifier | 0.81 | 0.79 | 0.85 |
| **ConvoGraph** | **0.87** | **0.85** | **0.91** |

*Full benchmark results coming after hackathon training runs*

## Hackathon Track Alignment

This project addresses **Track 1: Measurement & Evaluation**:

- **Manipulation benchmark**: Novel graph-based approach to sycophancy evaluation
- **Ecological validity**: Models real conversation structure, not just text
- **Detection method**: Per-turn sycophancy scores enable fine-grained analysis
- **Interpretability**: Attention patterns show *why* a conversation is flagged

## Future Work

1. **Multi-turn benchmark**: Extend TRUTH DECAY evaluation
2. **Real-time monitoring**: Deploy lightweight model for live detection
3. **Multi-agent analysis**: Extend to agent-agent manipulation
4. **Reward hacking detection**: Apply architecture to RL training traces

## Citation

If you use ConvoGraph in your research, please cite:

```bibtex
@misc{convograph2026,
  title={ConvoGraph: Detecting AI Sycophancy with Hybrid Graph Transformers},
  author={Shaikh, Tanzeel and Kaushik, Hitesh and Widerberg, Jakob},
  year={2026},
  howpublished={Apart Research AI Manipulation Hackathon},
  url={https://github.com/tanzeel291994/AI-manipulation-convograph}
}
```

## Related Work

- [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) - Anthropic
- [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) - HGT paper
- [SGFormer](https://arxiv.org/abs/2306.02089) - Simple Global Transformer

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Apart Research for organizing the AI Manipulation Hackathon
- Anthropic for sycophancy evaluation datasets
- PyTorch Geometric team for excellent GNN library

---

## Original HGL Architecture (Multi-hop QA)

This project builds on the original HGL (Heterogeneous Global-Local) architecture developed for multi-hop question answering. The original components are preserved below for reference.

### Original Pipeline

1.  **Global Knowledge Graph Construction**: Building a graph of entities, sentences, and passages
2.  **Self-Supervised Pre-training**: Learning structural representations without labels
3.  **Subgraph Extraction**: Using Beam Search to find relevant context windows
4.  **Supervised Ranking**: Fine-tuning the GNN to identify supporting evidence
5.  **Evaluation**: End-to-end metrics for recall and precision

### Original Dataset

The original system was trained on the **MuSiQue** dataset:
-   **Training**: 2,000 samples
-   **Evaluation**: 1,000 samples
-   **Results**: 80% Hit@1 on 2-hop queries
