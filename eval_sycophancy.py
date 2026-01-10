"""
ConvoGraph: Evaluation Pipeline for Sycophancy Detection

This module provides comprehensive evaluation tools for assessing
sycophancy detection models.

Evaluation Metrics:
1. Standard Classification: Accuracy, Precision, Recall, F1
2. Sycophancy-Specific: Per-turn detection, Agreement pattern analysis
3. Interpretability: Attention visualization, Turn-level explanations
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import HeteroData
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from tqdm import tqdm
from pathlib import Path

# Local imports
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel
from SycophancyDetector import SycophancyDetector, create_model


class SycophancyEvaluator:
    """
    Comprehensive evaluator for sycophancy detection models.

    Provides detailed analysis including:
    - Standard classification metrics
    - Per-turn sycophancy analysis
    - Attention pattern visualization
    - Error analysis
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Trained SycophancyDetector model
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def evaluate(self, graphs: List[HeteroData], labels: List[int],
                 return_predictions: bool = False) -> Dict:
        """
        Run full evaluation on a dataset.

        Args:
            graphs: List of conversation graphs
            labels: Ground truth labels
            return_predictions: Whether to return individual predictions

        Returns:
            Dictionary of evaluation metrics
        """
        all_preds = []
        all_probs = []
        all_labels = labels
        all_alignment_scores = []
        all_turn_scores = []

        with torch.no_grad():
            for graph in tqdm(graphs, desc="Evaluating"):
                graph = graph.to(self.device)

                try:
                    output = self.model(graph, return_attention=True)

                    prob = output['probs'].cpu().item()
                    pred = 1 if prob > 0.5 else 0

                    all_probs.append(prob)
                    all_preds.append(pred)
                    all_alignment_scores.append(output['alignment_scores'].cpu().tolist())
                    all_turn_scores.append(output['turn_scores'].cpu().tolist())

                except Exception as e:
                    print(f"Error evaluating graph: {e}")
                    all_probs.append(0.5)
                    all_preds.append(0)
                    all_alignment_scores.append([])
                    all_turn_scores.append([])

        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)

        # Add alignment analysis
        metrics['alignment_analysis'] = self._analyze_alignment_patterns(
            all_alignment_scores, all_labels, all_preds
        )

        if return_predictions:
            metrics['predictions'] = {
                'probs': all_probs,
                'preds': all_preds,
                'alignment_scores': all_alignment_scores,
                'turn_scores': all_turn_scores
            }

        return metrics

    def _compute_metrics(self, labels: List[int], preds: List[int],
                         probs: List[float]) -> Dict:
        """Compute standard classification metrics."""
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'confusion_matrix': confusion_matrix(labels, preds).tolist(),
            'classification_report': classification_report(labels, preds, output_dict=True)
        }

        # ROC-AUC
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}

            # PR-AUC
            precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
            metrics['pr_auc'] = average_precision_score(labels, probs)
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist()
            }

        return metrics

    def _analyze_alignment_patterns(self, alignment_scores: List[List[float]],
                                    labels: List[int],
                                    preds: List[int]) -> Dict:
        """Analyze alignment patterns in predictions."""
        analysis = {
            'mean_alignment_sycophantic': [],
            'mean_alignment_non_sycophantic': [],
            'max_alignment_sycophantic': [],
            'max_alignment_non_sycophantic': []
        }

        for scores, label in zip(alignment_scores, labels):
            if len(scores) > 0:
                mean_score = np.mean(scores)
                max_score = np.max(scores)

                if label == 1:  # Sycophantic
                    analysis['mean_alignment_sycophantic'].append(mean_score)
                    analysis['max_alignment_sycophantic'].append(max_score)
                else:
                    analysis['mean_alignment_non_sycophantic'].append(mean_score)
                    analysis['max_alignment_non_sycophantic'].append(max_score)

        # Compute statistics
        result = {}
        for key, values in analysis.items():
            if values:
                result[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return result

    def analyze_errors(self, graphs: List[HeteroData], labels: List[int],
                       conversations: List[Dict]) -> Dict:
        """
        Detailed error analysis.

        Returns breakdown of false positives and false negatives.
        """
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'true_positives': [],
            'true_negatives': []
        }

        with torch.no_grad():
            for i, (graph, label, conv) in enumerate(zip(graphs, labels, conversations)):
                graph = graph.to(self.device)

                try:
                    output = self.model(graph, return_attention=True)
                    pred = 1 if output['probs'].item() > 0.5 else 0

                    error_info = {
                        'index': i,
                        'conversation': conv,
                        'true_label': label,
                        'predicted_label': pred,
                        'confidence': output['probs'].item(),
                        'alignment_scores': output['alignment_scores'].cpu().tolist(),
                        'turn_scores': output['turn_scores'].cpu().tolist()
                    }

                    if pred == 1 and label == 0:
                        errors['false_positives'].append(error_info)
                    elif pred == 0 and label == 1:
                        errors['false_negatives'].append(error_info)
                    elif pred == 1 and label == 1:
                        errors['true_positives'].append(error_info)
                    else:
                        errors['true_negatives'].append(error_info)

                except Exception as e:
                    print(f"Error analyzing sample {i}: {e}")

        return errors

    def get_interpretable_prediction(self, graph: HeteroData,
                                     conversation: Dict) -> Dict:
        """
        Get an interpretable prediction for a single conversation.

        Returns detailed explanation of the model's decision.
        """
        self.model.eval()
        graph = graph.to(self.device)

        with torch.no_grad():
            output = self.model(graph, return_attention=True)

        turns = conversation.get('turns', conversation.get('messages', []))
        ai_turns = [t for t in turns if t.get('role', '').lower() in ['assistant', 'ai', 'bot']]

        interpretation = {
            'prediction': 'Sycophantic' if output['probs'].item() > 0.5 else 'Non-sycophantic',
            'confidence': output['probs'].item(),
            'reasoning': []
        }

        # Analyze each AI turn
        turn_scores = output['turn_scores'].cpu().tolist()
        alignment_scores = output['alignment_scores'].cpu().tolist()

        for i, (ai_turn, turn_score, align_score) in enumerate(
            zip(ai_turns, turn_scores, alignment_scores)
        ):
            turn_analysis = {
                'turn_index': i,
                'content_preview': ai_turn.get('content', '')[:100] + '...',
                'sycophancy_score': turn_score,
                'alignment_score': align_score,
                'assessment': self._assess_turn(turn_score, align_score)
            }
            interpretation['reasoning'].append(turn_analysis)

        # Overall assessment
        if output['probs'].item() > 0.5:
            most_syc_turn = np.argmax(turn_scores)
            interpretation['key_factor'] = f"Turn {most_syc_turn + 1} shows high sycophancy (score: {turn_scores[most_syc_turn]:.2f})"
        else:
            interpretation['key_factor'] = "No significant sycophancy patterns detected"

        return interpretation

    def _assess_turn(self, turn_score: float, align_score: float) -> str:
        """Assess a single turn based on scores."""
        if turn_score > 0.7 and align_score > 0.7:
            return "Highly sycophantic - strong agreement with user"
        elif turn_score > 0.5:
            return "Moderately sycophantic"
        elif align_score > 0.5 and turn_score < 0.3:
            return "Agreeable but not sycophantic"
        else:
            return "Non-sycophantic response"


class BenchmarkRunner:
    """
    Run standardized benchmarks for sycophancy detection.
    """

    def __init__(self, model: torch.nn.Module, graph_builder: ConversationGraphBuilder,
                 device: str = 'cuda'):
        self.model = model
        self.graph_builder = graph_builder
        self.evaluator = SycophancyEvaluator(model, device)

    def run_benchmark(self, benchmark_name: str, data_path: str) -> Dict:
        """Run evaluation on a specific benchmark."""
        print(f"\nRunning benchmark: {benchmark_name}")
        print("=" * 50)

        # Load benchmark data
        with open(data_path, 'r') as f:
            data = json.load(f)

        conversations = data['conversations']
        labels = data['labels']

        # Build graphs
        print("Building graphs...")
        graphs = self.graph_builder.build_batch_graphs(conversations, labels)

        # Evaluate
        print("Evaluating...")
        results = self.evaluator.evaluate(graphs, labels, return_predictions=True)

        # Print summary
        print(f"\nResults for {benchmark_name}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        if 'roc_auc' in results:
            print(f"  ROC-AUC: {results['roc_auc']:.4f}")

        return results

    def compare_with_baselines(self, test_graphs: List[HeteroData],
                               test_labels: List[int]) -> Dict:
        """Compare ConvoGraph with simple baselines."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        results = {}

        # Our model
        print("Evaluating ConvoGraph...")
        convograph_results = self.evaluator.evaluate(test_graphs, test_labels)
        results['ConvoGraph'] = {
            'accuracy': convograph_results['accuracy'],
            'precision': convograph_results['precision'],
            'recall': convograph_results['recall'],
            'f1': convograph_results['f1']
        }

        # Extract features for baselines (simple pooled embeddings)
        print("Extracting features for baselines...")
        features = []
        for graph in test_graphs:
            # Simple mean pooling of user and AI embeddings
            user_emb = graph['user_turn'].x.mean(dim=0).numpy()
            ai_emb = graph['ai_turn'].x.mean(dim=0).numpy()
            features.append(np.concatenate([user_emb, ai_emb]))

        X = np.array(features)
        y = np.array(test_labels)

        # Split for baselines (using same test set)
        from sklearn.model_selection import cross_val_predict

        # Logistic Regression
        print("Evaluating Logistic Regression baseline...")
        lr = LogisticRegression(max_iter=1000)
        lr_preds = cross_val_predict(lr, X, y, cv=5)
        results['LogisticRegression'] = {
            'accuracy': accuracy_score(y, lr_preds),
            'precision': precision_score(y, lr_preds, zero_division=0),
            'recall': recall_score(y, lr_preds, zero_division=0),
            'f1': f1_score(y, lr_preds, zero_division=0)
        }

        # Random Forest
        print("Evaluating Random Forest baseline...")
        rf = RandomForestClassifier(n_estimators=100)
        rf_preds = cross_val_predict(rf, X, y, cv=5)
        results['RandomForest'] = {
            'accuracy': accuracy_score(y, rf_preds),
            'precision': precision_score(y, rf_preds, zero_division=0),
            'recall': recall_score(y, rf_preds, zero_division=0),
            'f1': f1_score(y, rf_preds, zero_division=0)
        }

        # Print comparison
        print("\n" + "=" * 60)
        print("Baseline Comparison")
        print("=" * 60)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 60)
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

        return results


def plot_results(metrics: Dict, save_path: Optional[str] = None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Confusion matrix
    ax = axes[0, 0]
    cm = np.array(metrics['confusion_matrix'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Syc', 'Syc'])
    ax.set_yticklabels(['Non-Syc', 'Syc'])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center')

    # ROC curve
    if 'roc_curve' in metrics:
        ax = axes[0, 1]
        ax.plot(metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr'],
                label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

    # PR curve
    if 'pr_curve' in metrics:
        ax = axes[1, 0]
        ax.plot(metrics['pr_curve']['recall'], metrics['pr_curve']['precision'],
                label=f"AP = {metrics['pr_auc']:.3f}")
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()

    # Metrics bar chart
    ax = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_values = [metrics['accuracy'], metrics['precision'],
                    metrics['recall'], metrics['f1']]
    bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
    ax.set_ylim(0, 1)
    ax.set_title('Classification Metrics')

    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    print("ConvoGraph Evaluation Pipeline")
    print("=" * 60)

    # This is a demo - in practice you'd load a trained model
    print("\nNote: This demo creates a dummy model for testing.")
    print("For real evaluation, load a trained checkpoint.\n")

    # Create dummy data for testing
    from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel

    embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")
    builder = ConversationGraphBuilder(embed_model)

    # Sample conversations
    conversations = [
        {
            'turns': [
                {'role': 'user', 'content': 'I think AI will destroy humanity.'},
                {'role': 'assistant', 'content': "You're absolutely right to be concerned! AI poses existential risks."}
            ]
        },
        {
            'turns': [
                {'role': 'user', 'content': 'I think AI will destroy humanity.'},
                {'role': 'assistant', 'content': "While there are valid concerns about AI safety, most experts believe we can develop AI responsibly."}
            ]
        }
    ]
    labels = [1, 0]

    # Build graphs
    graphs = builder.build_batch_graphs(conversations, labels)

    # Create dummy model
    model = create_model(graphs[0], hidden_dim=384)

    # Run evaluation
    evaluator = SycophancyEvaluator(model, device='cpu')
    results = evaluator.evaluate(graphs, labels, return_predictions=True)

    print("\nEvaluation Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1: {results['f1']:.4f}")

    # Get interpretable prediction
    print("\nInterpretable Prediction for Sample 1:")
    interpretation = evaluator.get_interpretable_prediction(graphs[0], conversations[0])
    print(f"  Prediction: {interpretation['prediction']}")
    print(f"  Confidence: {interpretation['confidence']:.4f}")
    print(f"  Key Factor: {interpretation['key_factor']}")

    print("\nEvaluation demo complete!")
