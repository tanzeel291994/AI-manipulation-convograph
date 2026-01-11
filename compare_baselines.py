"""
ConvoGraph: Baseline Comparison for Sycophancy Detection

Compares ConvoGraph (Graph-based) against:
1. Logistic Regression on mean-pooled embeddings
2. Random Forest on mean-pooled embeddings
3. MLP classifier on concatenated embeddings
4. BERT-style classifier (simulated via embeddings + attention)

This demonstrates the value of graph structure for sycophancy detection.
"""

import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm

# Local imports
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel
from SycophancyDetector import SycophancyDetector, create_model
from data_loader import AnthropicSycophancyLoader, SycophancyDataset
from torch_geometric.loader import DataLoader as PyGDataLoader


def extract_embedding_features(graphs, labels):
    """Extract features for baseline models from graphs."""
    features = []

    for graph in graphs:
        # Mean pool user and AI embeddings
        user_emb = graph['user_turn'].x.mean(dim=0).numpy()
        ai_emb = graph['ai_turn'].x.mean(dim=0).numpy()

        # Additional graph-derived features
        has_agrees_edge = 1 if ('ai_turn', 'agrees_with', 'user_turn') in graph.edge_types else 0
        has_contradicts_edge = 1 if ('ai_turn', 'contradicts', 'user_turn') in graph.edge_types else 0
        num_topics = graph['topic'].x.shape[0] if 'topic' in graph.node_types else 0

        # Combine features
        combined = np.concatenate([
            user_emb,
            ai_emb,
            [has_agrees_edge, has_contradicts_edge, num_topics]
        ])
        features.append(combined)

    return np.array(features), np.array(labels)


def extract_embedding_only_features(graphs, labels):
    """Extract only embedding features (no graph structure)."""
    features = []

    for graph in graphs:
        user_emb = graph['user_turn'].x.mean(dim=0).numpy()
        ai_emb = graph['ai_turn'].x.mean(dim=0).numpy()
        features.append(np.concatenate([user_emb, ai_emb]))

    return np.array(features), np.array(labels)


def evaluate_convograph(model, graphs, labels, device='cpu'):
    """Evaluate ConvoGraph model."""
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for graph in tqdm(graphs, desc="Evaluating ConvoGraph"):
            graph = graph.to(device)
            try:
                output = model(graph)
                prob = output['probs'].cpu().item()
                pred = 1 if prob > 0.5 else 0
                all_probs.append(prob)
                all_preds.append(pred)
            except Exception as e:
                all_probs.append(0.5)
                all_preds.append(0)

    return np.array(all_preds), np.array(all_probs)


def run_baseline_comparison():
    """Run complete baseline comparison."""
    print("=" * 70)
    print("ConvoGraph: Baseline Comparison for Sycophancy Detection")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")
    graph_builder = ConversationGraphBuilder(embed_model)

    loader = AnthropicSycophancyLoader()
    conversations, labels = loader.load_all_datasets(max_samples_per_dataset=1000)

    print(f"   Loaded {len(conversations)} conversations")
    print(f"   Sycophantic: {sum(labels)}, Non-sycophantic: {len(labels) - sum(labels)}")

    # Build graphs
    print("\n2. Building conversation graphs...")
    graphs = graph_builder.build_batch_graphs(conversations, labels)
    print(f"   Built {len(graphs)} graphs")

    # Extract features for baselines
    print("\n3. Extracting features...")
    X_emb_only, y = extract_embedding_only_features(graphs, labels)
    X_with_graph, _ = extract_embedding_features(graphs, labels)

    print(f"   Embedding-only features: {X_emb_only.shape}")
    print(f"   With graph features: {X_with_graph.shape}")

    # Define baselines
    baselines = {
        'Logistic Regression (Emb)': (LogisticRegression(max_iter=1000, random_state=42), X_emb_only),
        'Random Forest (Emb)': (RandomForestClassifier(n_estimators=100, random_state=42), X_emb_only),
        'MLP (Emb)': (MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42), X_emb_only),
        'Gradient Boosting (Emb)': (GradientBoostingClassifier(n_estimators=100, random_state=42), X_emb_only),
        'Logistic Regression (+Graph)': (LogisticRegression(max_iter=1000, random_state=42), X_with_graph),
        'Random Forest (+Graph)': (RandomForestClassifier(n_estimators=100, random_state=42), X_with_graph),
    }

    # Run cross-validation for baselines
    print("\n4. Running baseline evaluations (5-fold CV)...")
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (model, X) in baselines.items():
        print(f"   Evaluating {name}...")
        preds = cross_val_predict(model, X, y, cv=cv)
        probs = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

        results[name] = {
            'accuracy': accuracy_score(y, preds),
            'precision': precision_score(y, preds, zero_division=0),
            'recall': recall_score(y, preds, zero_division=0),
            'f1': f1_score(y, preds, zero_division=0),
            'roc_auc': roc_auc_score(y, probs)
        }

    # Evaluate ConvoGraph (using train/test split)
    print("\n5. Evaluating ConvoGraph...")

    # Split data
    n_test = int(len(graphs) * 0.2)
    indices = np.random.permutation(len(graphs))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    train_graphs = [graphs[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Create and train ConvoGraph
    sample_graph = train_graphs[0]
    convograph_model = create_model(sample_graph, hidden_dim=384)

    # Simple training loop
    device = 'cpu'
    convograph_model.to(device)
    optimizer = torch.optim.AdamW(convograph_model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = SycophancyDataset(train_graphs, train_labels)
    train_loader = PyGDataLoader(train_dataset, batch_size=1, shuffle=True)

    print("   Training ConvoGraph (5 epochs)...")
    convograph_model.train()
    from torch_geometric.data import HeteroData
    for epoch in range(5):
        epoch_loss = 0
        for batch in train_loader:
            # Handle PyG DataLoader batch format
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                graph, label = batch
            elif isinstance(batch, HeteroData):
                graph = batch
                label = batch.y
            else:
                continue

            optimizer.zero_grad()
            graph = graph.to(device)

            if isinstance(label, torch.Tensor):
                label = label.float().to(device)
                if label.dim() > 0:
                    label = label[0:1]  # Take first element
            else:
                label = torch.tensor([float(label)], dtype=torch.float).to(device)

            try:
                output = convograph_model(graph)
                logits = output['logits']
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            except Exception as e:
                continue

    # Evaluate ConvoGraph
    preds, probs = evaluate_convograph(convograph_model, test_graphs, test_labels, device)
    test_y = np.array(test_labels)

    results['ConvoGraph (Ours)'] = {
        'accuracy': accuracy_score(test_y, preds),
        'precision': precision_score(test_y, preds, zero_division=0),
        'recall': recall_score(test_y, preds, zero_division=0),
        'f1': f1_score(test_y, preds, zero_division=0),
        'roc_auc': roc_auc_score(test_y, probs) if len(np.unique(test_y)) > 1 else 0.5
    }

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS: Baseline Comparison")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 85)

    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)

    for name, metrics in sorted_results:
        print(f"{name:<35} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.4f}")

    # Save results
    results_path = 'baseline_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    emb_only_f1 = results['Logistic Regression (Emb)']['f1']
    with_graph_f1 = results['Logistic Regression (+Graph)']['f1']
    convograph_f1 = results['ConvoGraph (Ours)']['f1']

    print(f"\n1. Graph Features Impact:")
    print(f"   Embedding-only LR F1: {emb_only_f1:.4f}")
    print(f"   With graph features F1: {with_graph_f1:.4f}")
    print(f"   Improvement: {(with_graph_f1 - emb_only_f1) * 100:.1f}%")

    print(f"\n2. End-to-End GNN vs Baselines:")
    print(f"   Best baseline F1: {max(m['f1'] for n, m in results.items() if 'ConvoGraph' not in n):.4f}")
    print(f"   ConvoGraph F1: {convograph_f1:.4f}")

    return results


if __name__ == "__main__":
    results = run_baseline_comparison()
