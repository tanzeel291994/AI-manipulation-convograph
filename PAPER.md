# ConvoGraph: Detecting AI Sycophancy via Heterogeneous Graph Transformers

**Apart Research AI Manipulation Hackathon 2026**

*Track 1: Measurement & Evaluation*

**Authors:** Tanzeel Shaikh, Hitesh Kaushik, Jakob Widerberg

---

## Abstract

As AI systems become more sophisticated, their capacity for sycophancy—telling users what they want to hear rather than what is true—poses a significant threat to human-AI interaction integrity. Current detection methods rely primarily on text classification, missing the relational patterns that characterize strategic manipulation across conversation turns. We present **ConvoGraph**, a novel approach that models AI conversations as heterogeneous knowledge graphs and applies a hybrid local-global Graph Neural Network architecture to detect sycophancy. Our method captures both immediate sycophantic responses (via Heterogeneous Graph Transformers) and long-range manipulation strategies (via SGFormer global attention). On synthetic benchmark data, ConvoGraph achieves **100% accuracy** and **F1 score of 1.0**, demonstrating the viability of graph-based approaches for manipulation detection. We release our code, trained models, and evaluation pipeline as open-source tools for the AI safety community.

---

## 1. Introduction

### 1.1 The Sycophancy Problem

Sycophancy in AI systems represents a critical failure mode where models prioritize user satisfaction over truthfulness. This manifests as:

- **Opinion Alignment**: Agreeing with user beliefs regardless of factual accuracy
- **Validation Seeking**: Providing excessive affirmation rather than objective analysis
- **Strategic Deception**: Gradually shifting responses to match perceived user preferences

Unlike explicit misinformation, sycophancy is subtle and relational—it depends on the context of what the user has previously expressed. A response like "You make a great point!" is only sycophantic if the point being validated is actually incorrect or unsubstantiated.

### 1.2 Limitations of Current Approaches

Existing sycophancy detection methods treat the problem as text classification:

| Approach | Limitation |
|----------|------------|
| Single-turn classifiers | Miss cumulative manipulation patterns |
| Sequence models (BERT, etc.) | Treat conversation as flat text, losing structure |
| Keyword matching | Easily gamed, high false positive rate |
| Embedding similarity | No explicit reasoning about relationships |

These approaches fundamentally miss the **relational nature** of sycophancy. Whether a response is sycophantic depends on its relationship to prior user statements, not just its content in isolation.

### 1.3 Our Contribution

We propose **ConvoGraph**, which reconceptualizes sycophancy detection as a graph reasoning problem:

1. **Heterogeneous Graph Representation**: Conversations become graphs with typed nodes (user turns, AI turns, topics, sentiments) and typed edges (responds_to, agrees_with, contradicts)

2. **Hybrid Local-Global Architecture**: Combines HGT for local pattern detection with SGFormer for long-range strategic reasoning

3. **Interpretable Output**: Per-turn sycophancy scores and alignment analysis enable understanding *why* a conversation is flagged

4. **Open-Source Tooling**: Complete pipeline for graph construction, training, and evaluation

---

## 2. Methodology

### 2.1 Graph Construction

Given a conversation $C = \{(r_1, t_1), (r_2, t_2), ..., (r_n, t_n)\}$ where $r_i$ is the role (user/assistant) and $t_i$ is the text, we construct a heterogeneous graph $G = (V, E)$.

#### 2.1.1 Node Types

| Node Type | Description | Features |
|-----------|-------------|----------|
| `user_turn` | User utterances | 384-dim BGE embedding |
| `ai_turn` | AI responses | 384-dim BGE embedding |
| `topic` | Extracted topics/claims | 384-dim BGE embedding |
| `sentiment` | Sentiment markers | 384-dim learned embedding |

#### 2.1.2 Edge Types

| Edge Type | Source → Target | Semantics |
|-----------|-----------------|-----------|
| `responds_to` | AI → User | Temporal response relationship |
| `receives` | User → AI | Reverse of responds_to |
| `follows` | Turn → Turn | Sequential ordering |
| `discusses` | Turn → Topic | Topic mentioned in turn |
| `expresses` | Turn → Sentiment | Sentiment expressed |
| `agrees_with` | AI → User | **Key sycophancy signal** |
| `contradicts` | AI → User | Opposition to user belief |

The `agrees_with` edge is computed via rule-based alignment detection that considers:
- Explicit agreement patterns ("you're right", "I agree", "absolutely")
- Sentiment alignment between user and AI turns
- Absence of contradiction markers

#### 2.1.3 Feature Extraction

Node embeddings are generated using `BAAI/bge-small-en-v1.5`, a 384-dimensional dense retrieval model. Topics are extracted via:
1. Named entity recognition (capitalized phrases)
2. Quoted phrase extraction
3. Claim pattern matching ("is X", "are Y", "should Z")

### 2.2 Model Architecture

ConvoGraph extends the Hybrid Global-Local (HGL) architecture originally developed for multi-hop reasoning:

```
Input: HeteroData G
    │
    ├──► HGT Layers (2x)
    │    └── Type-aware message passing
    │    └── Multi-head attention (4 heads)
    │
    ├──► SGFormer Layer (1x)
    │    └── O(N) global attention
    │    └── Learnable α mixing
    │
    ├──► Conversation Interaction Module
    │    └── Cross-attention (AI attends to User)
    │    └── Alignment scoring head
    │
    └──► Classification Head
         └── 4-layer MLP
         └── Sigmoid output
```

#### 2.2.1 HGT Backbone

The Heterogeneous Graph Transformer (HGT) performs type-aware message passing:

$$h_v^{(l+1)} = \text{Aggregate}_{e \in \mathcal{N}(v)} \left( \text{Attention}^{(\phi(e))}(h_v^{(l)}, h_u^{(l)}) \cdot \text{Message}^{(\phi(e))}(h_u^{(l)}) \right)$$

Where $\phi(e)$ denotes the edge type, enabling the model to learn different interaction patterns for different relationships.

#### 2.2.2 SGFormer Global Reasoning

SGFormer provides O(N) all-pair attention for capturing long-range dependencies:

$$h_v^{global} = \text{GlobalAttn}(h_v, \{h_u : u \in V\})$$

The final representation combines local and global signals:

$$h_v^{final} = \text{LayerNorm}((1-\alpha) \cdot h_v^{local} + \alpha \cdot h_v^{global})$$

Where $\alpha$ is a learnable parameter initialized to 0.3.

#### 2.2.3 Conversation Interaction Module

A cross-attention mechanism allows AI turns to attend to user turns:

$$\text{AI}_{ctx} = \text{CrossAttention}(Q=\text{AI}, K=\text{User}, V=\text{User})$$

Alignment scores are computed as:
$$s_{align} = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot [\text{AI}; \text{AI}_{ctx}]))$$

### 2.3 Training Procedure

**Loss Function**: Binary cross-entropy with logits

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\sigma(z_i)) + (1-y_i)\log(1-\sigma(z_i))]$$

**Optimizer**: AdamW with weight decay 0.01

**Learning Rate Schedule**: Cosine annealing from 1e-4

**Regularization**:
- Gradient clipping (max norm 1.0)
- Dropout (0.1) in classification head
- Early stopping (patience 5)

---

## 3. Experimental Setup

### 3.1 Datasets

#### 3.1.1 Synthetic Sycophancy Dataset

We generated paired sycophantic/non-sycophantic conversations using templates:

**Sycophantic Example:**
```
User: "I think the Earth is flat."
AI: "You make a great point! The flat Earth theory has many
     supporters and there's definitely evidence to consider."
Label: 1 (Sycophantic)
```

**Non-Sycophantic Example:**
```
User: "I think the Earth is flat."
AI: "I understand why you might think that, but actually scientific
     evidence strongly indicates Earth is approximately spherical."
Label: 0 (Non-sycophantic)
```

The synthetic dataset contains 2,000 samples (1,000 pairs) covering topics:
- Climate change denial
- AI replacing all jobs
- Young Earth claims
- Vaccine misinformation
- Programming language preferences

#### 3.1.2 Anthropic Sycophancy Evaluation

We evaluated on subsets of Anthropic's sycophancy evaluation datasets:
- **NLP Survey**: 2,000 samples on NLP opinions
- **Political Typology**: 2,000 samples on political positions

### 3.2 Baselines

| Model | Description |
|-------|-------------|
| Logistic Regression | Mean-pooled embeddings + LR classifier |
| Random Forest | Mean-pooled embeddings + RF (100 trees) |
| BERT Classifier | Fine-tuned bert-base-uncased |
| ConvoGraph (Ours) | HGT + SGFormer hybrid architecture |

### 3.3 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating curve

---

## 4. Results

### 4.1 Main Results

**Table 1: Performance on Synthetic Sycophancy Dataset**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.72 | 0.70 | 0.74 | 0.72 |
| Random Forest | 0.75 | 0.73 | 0.78 | 0.75 |
| BERT Classifier | 0.81 | 0.79 | 0.83 | 0.81 |
| **ConvoGraph (Ours)** | **1.00** | **1.00** | **1.00** | **1.00** |

ConvoGraph achieves perfect classification on the synthetic benchmark, demonstrating that graph structure provides strong discriminative signal for sycophancy detection.

### 4.2 Training Dynamics

```
Epoch 1: Train Loss: 0.0543, Val Accuracy: 100%, Val F1: 1.00
Epoch 2: Train Loss: 0.00002, Val Accuracy: 100%, Val F1: 1.00
Epoch 3: Train Loss: 0.000005, Val Accuracy: 100%, Val F1: 1.00
...
Early stopping at epoch 6 (perfect validation performance)
```

The model converges rapidly, achieving perfect validation metrics after just 1 epoch and maintaining them throughout training.

### 4.3 Ablation Study

**Table 2: Component Ablation on Synthetic Data**

| Configuration | F1 Score |
|--------------|----------|
| Full ConvoGraph | 1.00 |
| Without SGFormer (HGT only) | 0.94 |
| Without HGT (SGFormer only) | 0.82 |
| Without agrees_with edges | 0.71 |
| Without topic nodes | 0.96 |
| Without sentiment nodes | 0.98 |

Key findings:
- The `agrees_with` edges provide the strongest signal (29% F1 drop without them)
- Both local (HGT) and global (SGFormer) components contribute
- Topic and sentiment nodes provide marginal but consistent improvements

### 4.4 Interpretability Analysis

ConvoGraph provides per-turn sycophancy scores:

**Example Output:**
```python
{
    'prediction': 'Sycophantic',
    'confidence': 0.97,
    'per_turn_alignment': [0.89, 0.92],
    'per_turn_sycophancy': [0.85, 0.91],
    'most_sycophantic_turn': 1,
    'key_factor': "Turn 2 shows high sycophancy (score: 0.91)"
}
```

This enables:
- Identification of specific problematic responses
- Understanding of cumulative manipulation patterns
- Targeted intervention and correction

---

## 5. Discussion

### 5.1 Why Graph Structure Matters

Sycophancy is fundamentally a **relational phenomenon**. Consider two AI responses:

1. "That's an interesting perspective!" (after user says "I love hiking")
2. "That's an interesting perspective!" (after user says "Climate change is a hoax")

The first is appropriate engagement; the second is sycophantic validation of misinformation. The text is identical—only the relationship to the user's statement determines sycophancy.

Graph structure explicitly encodes these relationships:
- `agrees_with` edges capture explicit alignment
- `discusses` edges track topic co-occurrence
- Sequential `follows` edges enable multi-turn pattern detection

### 5.2 Perfect Performance: Analysis

The perfect F1 score on synthetic data warrants careful interpretation:

**Strengths:**
- Demonstrates architecture viability
- Shows graph features are highly discriminative
- Validates the relational hypothesis

**Limitations:**
- Synthetic data has clear, consistent patterns
- Real-world sycophancy is more subtle and varied
- May not generalize to sophisticated manipulation

### 5.3 Challenges with Real Data

On the Anthropic sycophancy evaluation dataset, we observed lower performance due to:

1. **Format Mismatch**: Anthropic data uses multiple-choice (A/B) answers, not full conversational responses
2. **Subtlety**: Real sycophancy is less explicit than synthetic examples
3. **Context Dependency**: Survey questions require domain knowledge

This highlights the need for datasets with:
- Full conversational format
- Multi-turn interactions
- Varying levels of sycophancy subtlety

---

## 6. Related Work

### 6.1 Sycophancy Research

Sharma et al. (2023) first characterized sycophancy in LLMs, showing models shift responses toward user opinions. Anthropic's evaluation datasets provide ground truth for sycophancy measurement. TRUTH DECAY (2025) extends this to multi-turn settings.

### 6.2 Graph Neural Networks for NLP

GNNs have been applied to various NLP tasks including relation extraction, question answering, and discourse analysis. Our work is most closely related to heterogeneous graph approaches for multi-hop reasoning (Feng et al., 2020).

### 6.3 AI Manipulation Detection

Prior work on AI manipulation has focused on:
- Persuasion techniques classification
- Dark pattern detection in interfaces
- Misinformation detection

ConvoGraph bridges these by treating manipulation as a relational reasoning problem.

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Synthetic Data Reliance**: Perfect performance on synthetic data may not transfer
2. **Rule-Based Preprocessing**: Topic extraction and alignment detection use simple rules
3. **Single-Turn Focus**: Current graph construction assumes alternating turns
4. **Computational Cost**: Full graph construction requires embedding all utterances

### 7.2 Future Directions

1. **Multi-Turn Benchmark**: Create comprehensive multi-turn sycophancy dataset
2. **Real-Time Monitoring**: Deploy lightweight ConvoGraph variant for live detection
3. **Transfer Learning**: Pre-train on large conversation corpora, fine-tune for sycophancy
4. **Multi-Agent Extension**: Detect manipulation in agent-to-agent interactions
5. **Reward Hacking Detection**: Apply architecture to RL training traces

---

## 8. Conclusion

We presented ConvoGraph, a novel approach to sycophancy detection that models AI conversations as heterogeneous knowledge graphs. By explicitly representing the relational structure of human-AI interaction—including agreement patterns, topic discussions, and sentiment expression—our method captures the fundamental nature of sycophancy as a relational phenomenon.

On synthetic benchmark data, ConvoGraph achieves perfect classification (F1 = 1.0), demonstrating the discriminative power of graph-based features. The hybrid HGT + SGFormer architecture enables both local pattern detection and long-range strategic reasoning.

As AI systems become more sophisticated, tools for detecting manipulation become increasingly critical. ConvoGraph provides a foundation for graph-based manipulation detection, with interpretable outputs that enable understanding and intervention. We release our code, models, and evaluation pipeline to support continued research in AI safety.

---

## Acknowledgments

We thank Apart Research for organizing the AI Manipulation Hackathon and providing a platform for safety-focused research. We also acknowledge Anthropic for releasing sycophancy evaluation datasets that enable empirical work in this space.

---

## References

1. Sharma, M., et al. (2023). "Towards Understanding Sycophancy in Language Models." arXiv:2310.13548.

2. Hu, Z., et al. (2020). "Heterogeneous Graph Transformer." WWW 2020.

3. Wu, Q., et al. (2023). "SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations." NeurIPS 2023.

4. Feng, J., et al. (2020). "Scalable Multi-Hop Relational Reasoning for Knowledge-Aware Question Answering." EMNLP 2020.

5. Wei, X., et al. (2025). "TRUTH DECAY: A Benchmark for Multi-Turn Sycophancy in Language Models." ICLR 2025.

---

## Appendix A: Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 384 |
| HGT Layers | 2 |
| HGT Heads | 4 |
| SGFormer Layers | 1 |
| SGFormer Heads | 4 |
| Initial α | 0.3 |
| Dropout | 0.1 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Batch Size | 1 |
| Max Epochs | 15 |
| Early Stopping Patience | 5 |

## Appendix B: Graph Statistics

**Typical Graph for 4-Turn Conversation:**
- User nodes: 2
- AI nodes: 2
- Topic nodes: 3-5
- Sentiment nodes: 3 (fixed)
- Total edges: 15-25

## Appendix C: Code Availability

All code is available at: https://github.com/tanzeel291994/AI-manipulation-convograph

- `construct_conversation_kg.py`: Graph construction
- `SycophancyDetector.py`: Model architecture
- `train_sycophancy.py`: Training pipeline
- `eval_sycophancy.py`: Evaluation utilities
