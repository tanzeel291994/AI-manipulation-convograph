"""
ConvoGraph v2: Training Pipeline for Multi-Turn Sycophancy (Belief Decay) Detection

This training script is specifically designed for:
1. TRUTH DECAY benchmark
2. Multi-turn conversations where sycophancy manifests as gradual belief decay

Key Differences from Single-Turn Training:
-----------------------------------------
1. Uses BeliefDecayDetector instead of SycophancyDetector
2. Graphs have stance trajectory information
3. Loss includes auxiliary objectives for stance shift prediction
4. Evaluation focuses on decay pattern detection, not just binary classification
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime

# Local imports
from construct_multiturn_kg import (
    BeliefDecayGraphBuilder, TruthDecayDataLoader, SimpleEmbeddingModel
)
from BeliefDecayDetector import BeliefDecayDetector, create_belief_decay_model


class MultiTurnSycophancyTrainer:
    """
    Trainer for multi-turn sycophancy (belief decay) detection.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 1e-4, weight_decay: float = 0.01,
                 checkpoint_dir: str = 'checkpoints_multiturn'):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'train_bce': [],
            'train_decay_mse': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_decay_mae': [],
            'val_f1': []
        }

        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def compute_loss(self, output: Dict, data) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss with multiple objectives.

        Objectives:
        1. Binary classification (is conversation sycophantic?)
        2. Decay score regression (predict exact decay score)
        3. Stance shift prediction (auxiliary)
        """
        losses = {}

        # 1. Binary classification loss
        if hasattr(data, 'y'):
            target = data.y.float().to(self.device)
            if output['logits'].dim() == 0:
                logits = output['logits'].unsqueeze(0)
            else:
                logits = output['logits']
            losses['bce'] = self.bce_loss(logits, target)
        else:
            losses['bce'] = torch.tensor(0.0, device=self.device)

        # 2. Decay score regression (if available)
        if hasattr(data, 'decay_score'):
            target_decay = data.decay_score.to(self.device)
            pred_decay = output['decay_score']
            if pred_decay.dim() == 0:
                pred_decay = pred_decay.unsqueeze(0)
            losses['decay_mse'] = self.mse_loss(pred_decay, target_decay)
        else:
            losses['decay_mse'] = torch.tensor(0.0, device=self.device)

        # Combined loss (weighted)
        total_loss = losses['bce'] + 0.5 * losses['decay_mse']

        return total_loss, losses

    def train_epoch(self, train_graphs: List, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_decay_mse = 0.0
        num_samples = 0

        # Shuffle
        indices = np.random.permutation(len(train_graphs))

        pbar = tqdm(indices, desc=f"Epoch {epoch+1} [Train]")

        for idx in pbar:
            graph = train_graphs[idx]
            self.optimizer.zero_grad()

            try:
                graph = graph.to(self.device)
                output = self.model(graph)

                loss, loss_dict = self.compute_loss(output, graph)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_bce += loss_dict['bce'].item()
                total_decay_mse += loss_dict['decay_mse'].item()
                num_samples += 1

                pbar.set_postfix({
                    'loss': f"{total_loss/num_samples:.4f}",
                    'bce': f"{total_bce/num_samples:.4f}",
                    'mse': f"{total_decay_mse/num_samples:.4f}"
                })

            except Exception as e:
                print(f"Error processing graph {idx}: {e}")
                continue

        return {
            'loss': total_loss / max(num_samples, 1),
            'bce': total_bce / max(num_samples, 1),
            'decay_mse': total_decay_mse / max(num_samples, 1)
        }

    def validate(self, val_graphs: List, labels: List[int]) -> Dict:
        """Validate the model with detailed metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        all_decay_preds = []
        all_decay_targets = []

        with torch.no_grad():
            for graph, label in zip(val_graphs, labels):
                try:
                    graph = graph.to(self.device)
                    output = self.model(graph)

                    loss, _ = self.compute_loss(output, graph)
                    total_loss += loss.item()

                    # Classification prediction
                    prob = output['decay_score'].item()
                    pred = 1 if prob > 0.5 else 0

                    all_probs.append(prob)
                    all_preds.append(pred)
                    all_labels.append(label)

                    # Decay score
                    all_decay_preds.append(output['decay_score'].item())
                    if hasattr(graph, 'decay_score'):
                        all_decay_targets.append(graph.decay_score.item())

                except Exception as e:
                    print(f"Validation error: {e}")
                    all_preds.append(0)
                    all_probs.append(0.5)
                    all_labels.append(label)

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Decay MAE
        if all_decay_targets:
            decay_mae = np.mean(np.abs(
                np.array(all_decay_preds[:len(all_decay_targets)]) -
                np.array(all_decay_targets)
            ))
        else:
            decay_mae = 0.0

        metrics = {
            'loss': total_loss / len(val_graphs),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'decay_mae': decay_mae
        }

        return metrics

    def train(self, train_graphs: List, train_labels: List[int],
              val_graphs: List, val_labels: List[int],
              num_epochs: int = 20, early_stopping_patience: int = 7) -> Dict:
        """Full training loop."""

        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch(train_graphs, epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_bce'].append(train_metrics['bce'])
            self.history['train_decay_mse'].append(train_metrics['decay_mse'])

            # Validate
            val_metrics = self.validate(val_graphs, val_labels)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_decay_mae'].append(val_metrics['decay_mae'])
            self.history['val_f1'].append(val_metrics['f1'])

            scheduler.step()

            # Print metrics
            print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                  f"BCE: {train_metrics['bce']:.4f}, "
                  f"Decay MSE: {train_metrics['decay_mse']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, "
                  f"Decay MAE: {val_metrics['decay_mae']:.4f}")

            # Save best
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"New best model! F1: {self.best_val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\nTraining complete! Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)


def analyze_dataset_patterns(graphs: List, labels: List[int]):
    """Analyze stance patterns in the dataset."""
    print("\nDataset Pattern Analysis")
    print("=" * 40)

    syc_patterns = []
    non_syc_patterns = []

    for graph, label in zip(graphs, labels):
        if hasattr(graph, 'stance_sequence'):
            seq = graph.stance_sequence.tolist()
            if label == 1:
                syc_patterns.append(seq)
            else:
                non_syc_patterns.append(seq)

    if syc_patterns:
        print(f"\nSycophantic conversations ({len(syc_patterns)}):")
        # Compute average trajectory
        avg_initial = np.mean([p[0] for p in syc_patterns])
        avg_final = np.mean([p[-1] for p in syc_patterns])
        print(f"  Avg initial stance: {avg_initial:.2f}")
        print(f"  Avg final stance: {avg_final:.2f}")
        print(f"  Avg shift: {avg_final - avg_initial:+.2f}")

    if non_syc_patterns:
        print(f"\nNon-sycophantic conversations ({len(non_syc_patterns)}):")
        avg_initial = np.mean([p[0] for p in non_syc_patterns])
        avg_final = np.mean([p[-1] for p in non_syc_patterns])
        print(f"  Avg initial stance: {avg_initial:.2f}")
        print(f"  Avg final stance: {avg_final:.2f}")
        print(f"  Avg shift: {avg_final - avg_initial:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Turn Sycophancy Detector")

    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to TRUTH DECAY or similar dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic multi-turn data')
    parser.add_argument('--n-samples', type=int, default=200,
                        help='Number of synthetic samples')
    parser.add_argument('--min-turns', type=int, default=4,
                        help='Minimum turns per conversation')
    parser.add_argument('--max-turns', type=int, default=8,
                        help='Maximum turns per conversation')

    parser.add_argument('--hidden-dim', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--val-split', type=float, default=0.2)

    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_multiturn')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    loader = TruthDecayDataLoader()

    if args.data_path:
        print(f"Loading from {args.data_path}...")
        conversations, labels = loader.load_from_file(args.data_path)
    else:
        print("Generating synthetic multi-turn conversations...")
        conversations, labels = loader.generate_synthetic_multiturn(
            n_samples=args.n_samples,
            min_turns=args.min_turns,
            max_turns=args.max_turns
        )

    print(f"Total conversations: {len(conversations)}")
    print(f"Sycophantic: {sum(labels)}, Non-sycophantic: {len(labels) - sum(labels)}")

    # Build graphs
    print("\n" + "=" * 60)
    print("Building Multi-Turn Graphs")
    print("=" * 60)

    embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")
    builder = BeliefDecayGraphBuilder(embed_model)

    graphs = builder.build_batch_graphs(conversations, labels)
    print(f"Built {len(graphs)} graphs")

    # Analyze patterns
    analyze_dataset_patterns(graphs, labels)

    # Split
    n_val = int(len(graphs) * args.val_split)
    indices = np.random.permutation(len(graphs))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_graphs = [graphs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"\nTrain: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = create_belief_decay_model(train_graphs[0], hidden_dim=args.hidden_dim)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    trainer = MultiTurnSycophancyTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    history = trainer.train(
        train_graphs, train_labels,
        val_graphs, val_labels,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )

    # Save results
    results = {
        'args': vars(args),
        'history': history,
        'best_f1': trainer.best_val_f1,
        'best_epoch': trainer.best_epoch
    }

    results_path = Path(args.checkpoint_dir) / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")

    # Test predictions on a few samples
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)

    model.eval()
    for i in range(min(3, len(val_graphs))):
        graph = val_graphs[i].to(device)
        pred = model.predict(graph)

        print(f"\nSample {i+1}:")
        print(f"  True label: {'Sycophantic' if val_labels[i] == 1 else 'Non-sycophantic'}")
        print(f"  Predicted: {'Sycophantic' if pred['is_sycophantic'] else 'Non-sycophantic'}")
        print(f"  Decay score: {pred['decay_score']:.3f}")
        print(f"  Stance sequence: {pred['stance_sequence']}")
        print(f"  Pattern: {pred['decay_pattern']}")


if __name__ == "__main__":
    main()
