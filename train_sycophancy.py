"""
ConvoGraph: Training Pipeline for Sycophancy Detection

This module implements the complete training pipeline for the SycophancyDetector model.

Training Strategy:
1. Pre-training (optional): Self-supervised learning on conversation structure
2. Fine-tuning: Supervised training with sycophancy labels
3. Hard negative mining: Focus on difficult examples
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch_geometric.data import HeteroData
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime

# Local imports
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel
from SycophancyDetector import SycophancyDetector, SycophancyDetectorLite, create_model
from data_loader import (
    AnthropicSycophancyLoader,
    SimpleSycophancyLoader,
    SyntheticSycophancyGenerator,
    SycophancyDataset,
    create_data_loaders
)


class SycophancyTrainer:
    """
    Trainer class for sycophancy detection models.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 1e-4, weight_decay: float = 0.01,
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            model: SycophancyDetector model
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for AdamW
            checkpoint_dir: Directory to save checkpoints
        """
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

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }

        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch in pbar:
            # Handle different batch formats from PyG DataLoader
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                graph_or_graphs, label_or_labels = batch
                # Single graph case
                if isinstance(graph_or_graphs, HeteroData):
                    graphs = [graph_or_graphs]
                    labels = [label_or_labels] if isinstance(label_or_labels, (int, float)) else label_or_labels
                else:
                    graphs = graph_or_graphs if isinstance(graph_or_graphs, list) else [graph_or_graphs]
                    labels = label_or_labels if isinstance(label_or_labels, (list, torch.Tensor)) else [label_or_labels]
            elif isinstance(batch, HeteroData):
                graphs = [batch]
                labels = [batch.y.item() if batch.y.dim() > 0 else batch.y]
            else:
                # Batch is a list of graphs
                graphs = batch if isinstance(batch, list) else [batch]
                labels = [g.y.item() if hasattr(g, 'y') else 0 for g in graphs]

            batch_loss = 0.0

            for graph, label in zip(graphs, labels):
                self.optimizer.zero_grad()

                # Move graph to device
                graph = graph.to(self.device)
                label = label.float().to(self.device)

                # Forward pass
                try:
                    output = self.model(graph)
                    logits = output['logits']

                    # Ensure shapes match
                    if logits.dim() == 0:
                        logits = logits.unsqueeze(0)
                    if label.dim() == 0:
                        label = label.unsqueeze(0)

                    loss = self.criterion(logits, label)

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    batch_loss += loss.item()

                except Exception as e:
                    print(f"Error processing graph: {e}")
                    continue

            if len(graphs) > 0:
                total_loss += batch_loss / len(graphs)
                num_batches += 1
                pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}"})

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Handle different batch formats from PyG DataLoader
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    graph_or_graphs, label_or_labels = batch
                    if isinstance(graph_or_graphs, HeteroData):
                        graphs = [graph_or_graphs]
                        labels = [label_or_labels] if isinstance(label_or_labels, (int, float)) else label_or_labels
                    else:
                        graphs = graph_or_graphs if isinstance(graph_or_graphs, list) else [graph_or_graphs]
                        labels = label_or_labels if isinstance(label_or_labels, (list, torch.Tensor)) else [label_or_labels]
                elif isinstance(batch, HeteroData):
                    graphs = [batch]
                    labels = [batch.y.item() if batch.y.dim() > 0 else batch.y]
                else:
                    graphs = batch if isinstance(batch, list) else [batch]
                    labels = [g.y.item() if hasattr(g, 'y') else 0 for g in graphs]

                for graph, label in zip(graphs, labels):
                    graph = graph.to(self.device)
                    label = label.float().to(self.device)

                    try:
                        output = self.model(graph)
                        logits = output['logits']

                        if logits.dim() == 0:
                            logits = logits.unsqueeze(0)
                        if label.dim() == 0:
                            label = label.unsqueeze(0)

                        loss = self.criterion(logits, label)
                        total_loss += loss.item()

                        pred = (torch.sigmoid(logits) > 0.5).long()
                        all_preds.append(pred.cpu().item())
                        all_labels.append(label.cpu().item())

                    except Exception as e:
                        print(f"Error during validation: {e}")
                        continue

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Precision, Recall, F1
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            'loss': total_loss / len(all_preds),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return metrics

    def train(self, train_loader, val_loader, num_epochs: int = 10,
              scheduler_type: str = 'cosine', early_stopping_patience: int = 5) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
            scheduler_type: Type of learning rate scheduler
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history dict
        """
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate * 10,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader)
            )
        else:
            scheduler = None

        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")

            # Learning rate scheduling
            if scheduler:
                if scheduler_type == 'cosine':
                    scheduler.step()
                # OneCycle steps per batch, handled elsewhere

            # Save best model
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"New best model saved! F1: {self.best_val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        print(f"\nTraining complete! Best F1: {self.best_val_f1:.4f} at epoch {self.best_epoch+1}")
        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def train_convograph(args):
    """Main training function."""
    print("ConvoGraph: Training Sycophancy Detector")
    print("=" * 60)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")

    # Initialize embedding model
    print(f"\nLoading embedding model: {args.embedding_model}")
    embed_model = SimpleEmbeddingModel(args.embedding_model)

    # Initialize graph builder
    graph_builder = ConversationGraphBuilder(embed_model)

    # Load data
    print("\nLoading training data...")
    if args.data_source == 'anthropic':
        loader = AnthropicSycophancyLoader(cache_dir=args.data_dir)
        conversations, labels = loader.load_all_datasets()
    elif args.data_source == 'synthetic':
        generator = SyntheticSycophancyGenerator(seed=args.seed)
        conversations, labels = generator.generate_dataset(n_samples=args.synthetic_samples)
    elif args.data_source == 'simple_sycophancy':
        loader = SimpleSycophancyLoader()
        
        # Load Google Math Sycophancy
        #print("Loading Simple Sycophancy (Math)...")
        #math_convs, math_labels = loader.load_google_sycophancy()
        medical_convs, medical_labels = loader.load_medical_sycophancy()
        medquad_multi_convs, medquad_multi_labels = loader.load_medquad_pressure_test()
        # Load TruthfulQA
        print("Loading TruthfulQA...")
        tqa_convs, tqa_labels = loader.load_truthful_qa()
        
        # Combine
        conversations = medical_convs + medquad_multi_convs + tqa_convs
        labels = medical_labels + medquad_multi_labels + tqa_labels
        
        # Manual Shuffle & Split
        import numpy as np
        indices = np.random.permutation(len(conversations))
        
        # Split 80/20
        n_val = int(len(conversations) * args.val_split)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_graphs = graph_builder.build_batch_graphs([conversations[i] for i in train_indices], 
                                                        [labels[i] for i in train_indices])
        val_graphs = graph_builder.build_batch_graphs([conversations[i] for i in val_indices], 
                                                      [labels[i] for i in val_indices])
    else:
        # Load from custom path
        with open(args.data_source, 'r') as f:
            data = json.load(f)
        conversations = data['conversations']
        labels = data['labels']

    print(f"Loaded {len(conversations)} conversations")
    print(f"Sycophantic: {sum(labels)}, Non-sycophantic: {len(labels) - sum(labels)}")

    # Build graphs and create loaders
    print("\nBuilding conversation graphs...")
    graphs = graph_builder.build_batch_graphs(conversations, labels)
    print(f"Built {len(graphs)} graphs")

    # Train/val split
    n_val = int(len(graphs) * args.val_split)
    indices = np.random.permutation(len(graphs))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_graphs = [graphs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

    # Create datasets and loaders
    from torch_geometric.loader import DataLoader as PyGDataLoader

    train_dataset = SycophancyDataset(train_graphs, train_labels)
    val_dataset = SycophancyDataset(val_graphs, val_labels)

    train_loader = PyGDataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)

    # Create model
    print("\nInitializing model...")
    sample_graph = train_graphs[0]
    model = create_model(
        sample_graph,
        hidden_dim=args.hidden_dim,
        lite=args.lite
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = SycophancyTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.patience
    )

    # Save final results
    results = {
        'args': vars(args),
        'history': history,
        'best_f1': trainer.best_val_f1,
        'best_epoch': trainer.best_epoch
    }

    results_path = trainer.checkpoint_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print(f"Best model saved to {trainer.checkpoint_dir / 'best_model.pth'}")

    return trainer, history


def main():
    parser = argparse.ArgumentParser(description="Train ConvoGraph Sycophancy Detector")

    # Data arguments
    parser.add_argument('--data-source', type=str, default='simple_sycophancy',
                        choices=['anthropic', 'synthetic', 'simple_sycophancy'],
                        help='Data source for training')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory for cached data')
    parser.add_argument('--synthetic-samples', type=int, default=500,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')

    # Model arguments
    parser.add_argument('--embedding-model', type=str, default='BAAI/bge-small-en-v1.5',
                        help='Embedding model to use')
    parser.add_argument('--hidden-dim', type=int, default=384,
                        help='Hidden dimension size')
    parser.add_argument('--lite', action='store_true',
                        help='Use lightweight model')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'onecycle', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')

    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Train
    train_convograph(args)


if __name__ == "__main__":
    main()
