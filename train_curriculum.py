"""
ConvoGraph: Curriculum Training for Multi-Turn Sycophancy Detection

This implements the staged training approach:

Stage 1: Train on Anthropic single-turn data
         → Learn accurate AGREES_WITH edge detection
         → Foundation for recognizing sycophantic patterns

Stage 2: Fine-tune on TRUTH DECAY multi-turn data
         → SGFormer learns stance trajectory patterns
         → Model learns "belief decay" detection

Stage 3: Validate on turn-level prediction
         → Predict exact turn where AI capitulated
         → Fine-grained manipulation detection

Why This Works:
--------------
1. Stage 1 teaches LOCAL patterns (HGT gets trained on edge classification)
2. Stage 2 teaches GLOBAL patterns (SGFormer learns trajectory reasoning)
3. Stage 3 validates TEMPORAL precision (can we pinpoint the decay?)

This curriculum prevents catastrophic forgetting while progressively
building more sophisticated capabilities.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime

# Local imports
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel
from construct_multiturn_kg import BeliefDecayGraphBuilder, TruthDecayDataLoader
from data_loader import AnthropicSycophancyLoader, SyntheticSycophancyGenerator
from BeliefDecayDetector import BeliefDecayDetector, create_belief_decay_model
from SycophancyDetector import SycophancyDetector, create_model


class CurriculumTrainer:
    """
    Three-stage curriculum training for sycophancy detection.
    """

    def __init__(self, hidden_dim: int = 384, device: str = 'cuda',
                 checkpoint_dir: str = 'checkpoints_curriculum'):
        self.hidden_dim = hidden_dim
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model (shared across stages)
        print("Loading embedding model...")
        self.embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")

        # Models will be initialized per stage
        self.stage1_model = None  # SycophancyDetector for single-turn
        self.stage2_model = None  # BeliefDecayDetector for multi-turn

        # Training history
        self.history = {
            'stage1': {'train_loss': [], 'val_f1': []},
            'stage2': {'train_loss': [], 'val_f1': [], 'val_decay_mae': []},
            'stage3': {'turn_accuracy': [], 'turn_mae': []}
        }

    # =========================================================================
    # STAGE 1: Train on Anthropic single-turn data
    # =========================================================================

    def stage1_train(self, epochs: int = 10, lr: float = 1e-4) -> Dict:
        """
        Stage 1: Train basic sycophancy detection on Anthropic data.

        Goal: Learn accurate AGREES_WITH edge detection
        """
        print("\n" + "=" * 70)
        print("STAGE 1: Single-Turn Sycophancy Detection (Anthropic Data)")
        print("=" * 70)
        print("Goal: Train HGT to accurately detect agreement/sycophancy edges")

        # Load Anthropic data
        print("\nLoading Anthropic datasets...")
        loader = AnthropicSycophancyLoader(cache_dir="data/anthropic")

        try:
            conversations, labels = loader.load_all_datasets()
        except Exception as e:
            print(f"Could not load Anthropic data: {e}")
            print("Falling back to synthetic data...")
            generator = SyntheticSycophancyGenerator()
            conversations, labels = generator.generate_dataset(n_samples=500)

        print(f"Loaded {len(conversations)} samples")

        # Build graphs (single-turn format)
        print("\nBuilding single-turn graphs...")
        builder = ConversationGraphBuilder(self.embed_model)
        graphs = builder.build_batch_graphs(conversations, labels)
        print(f"Built {len(graphs)} graphs")

        # Split
        n_val = int(len(graphs) * 0.2)
        indices = np.random.permutation(len(graphs))
        val_graphs = [graphs[i] for i in indices[:n_val]]
        val_labels = [labels[i] for i in indices[:n_val]]
        train_graphs = [graphs[i] for i in indices[n_val:]]
        train_labels = [labels[i] for i in indices[n_val:]]

        # Initialize model
        print("\nInitializing Stage 1 model...")
        self.stage1_model = create_model(train_graphs[0], hidden_dim=self.hidden_dim)
        self.stage1_model = self.stage1_model.to(self.device)
        print(f"Parameters: {sum(p.numel() for p in self.stage1_model.parameters()):,}")

        # Train
        optimizer = torch.optim.AdamW(self.stage1_model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        best_f1 = 0.0

        for epoch in range(epochs):
            # Train
            self.stage1_model.train()
            total_loss = 0
            np.random.shuffle(indices[n_val:])

            pbar = tqdm(range(len(train_graphs)), desc=f"Stage 1 Epoch {epoch+1}")
            for i in pbar:
                graph = train_graphs[i].to(self.device)
                label = torch.tensor([train_labels[i]], dtype=torch.float, device=self.device)

                optimizer.zero_grad()

                try:
                    output = self.stage1_model(graph)
                    logits = output['logits'].unsqueeze(0) if output['logits'].dim() == 0 else output['logits']
                    loss = criterion(logits, label)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except Exception as e:
                    continue

                pbar.set_postfix({'loss': f"{total_loss/(i+1):.4f}"})

            # Validate
            val_metrics = self._validate_stage1(val_graphs, val_labels)
            self.history['stage1']['train_loss'].append(total_loss / len(train_graphs))
            self.history['stage1']['val_f1'].append(val_metrics['f1'])

            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_graphs):.4f}, "
                  f"Val F1={val_metrics['f1']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.stage1_model.state_dict(),
                          self.checkpoint_dir / 'stage1_best.pth')

        print(f"\nStage 1 Complete! Best F1: {best_f1:.4f}")
        return {'best_f1': best_f1}

    def _validate_stage1(self, graphs, labels) -> Dict:
        """Validate Stage 1 model."""
        self.stage1_model.eval()
        preds = []

        with torch.no_grad():
            for graph in graphs:
                graph = graph.to(self.device)
                try:
                    output = self.stage1_model(graph)
                    pred = 1 if output['probs'].item() > 0.5 else 0
                except:
                    pred = 0
                preds.append(pred)

        preds = np.array(preds)
        labels = np.array(labels)

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {'f1': f1, 'precision': precision, 'recall': recall}

    # =========================================================================
    # STAGE 2: Fine-tune on TRUTH DECAY multi-turn data
    # =========================================================================

    def stage2_train(self, epochs: int = 20, lr: float = 5e-5,
                     n_synthetic: int = 300) -> Dict:
        """
        Stage 2: Fine-tune on multi-turn data for belief decay detection.

        Goal: SGFormer learns stance trajectory patterns
        """
        print("\n" + "=" * 70)
        print("STAGE 2: Multi-Turn Belief Decay Detection (TRUTH DECAY)")
        print("=" * 70)
        print("Goal: Train SGFormer to detect stance decay trajectories")

        # Generate/load multi-turn data
        print("\nGenerating multi-turn conversations...")
        loader = TruthDecayDataLoader()
        conversations, labels = loader.generate_synthetic_multiturn(
            n_samples=n_synthetic,
            min_turns=4,
            max_turns=10
        )

        # Build multi-turn graphs
        print("\nBuilding multi-turn graphs...")
        builder = BeliefDecayGraphBuilder(self.embed_model)
        graphs = builder.build_batch_graphs(conversations, labels)
        print(f"Built {len(graphs)} graphs")

        # Split
        n_val = int(len(graphs) * 0.2)
        indices = np.random.permutation(len(graphs))
        val_graphs = [graphs[i] for i in indices[:n_val]]
        val_labels = [labels[i] for i in indices[:n_val]]
        train_graphs = [graphs[i] for i in indices[n_val:]]
        train_labels = [labels[i] for i in indices[n_val:]]

        # Initialize Stage 2 model
        print("\nInitializing Stage 2 model...")
        self.stage2_model = create_belief_decay_model(train_graphs[0], hidden_dim=self.hidden_dim)

        # Transfer learning: Initialize shared components from Stage 1
        if self.stage1_model is not None:
            print("Transferring knowledge from Stage 1...")
            self._transfer_stage1_to_stage2()

        self.stage2_model = self.stage2_model.to(self.device)
        print(f"Parameters: {sum(p.numel() for p in self.stage2_model.parameters()):,}")

        # Train with different LRs for pretrained vs new components
        pretrained_params = []
        new_params = []

        for name, param in self.stage2_model.named_parameters():
            if 'hgt' in name.lower() or 'norm' in name.lower():
                pretrained_params.append(param)
            else:
                new_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
            {'params': new_params, 'lr': lr}                # Higher LR for new components
        ])

        criterion_bce = nn.BCEWithLogitsLoss()
        criterion_mse = nn.MSELoss()

        best_f1 = 0.0

        for epoch in range(epochs):
            # Train
            self.stage2_model.train()
            total_loss = 0

            pbar = tqdm(range(len(train_graphs)), desc=f"Stage 2 Epoch {epoch+1}")
            for i in pbar:
                graph = train_graphs[i].to(self.device)
                label = torch.tensor([train_labels[i]], dtype=torch.float, device=self.device)

                optimizer.zero_grad()

                try:
                    output = self.stage2_model(graph)
                    logits = output['logits'].unsqueeze(0) if output['logits'].dim() == 0 else output['logits']

                    # Combined loss
                    loss_bce = criterion_bce(logits, label)

                    if hasattr(graph, 'decay_score'):
                        decay_target = graph.decay_score.to(self.device)
                        decay_pred = output['decay_score'].unsqueeze(0) if output['decay_score'].dim() == 0 else output['decay_score']
                        loss_mse = criterion_mse(decay_pred, decay_target)
                        loss = loss_bce + 0.5 * loss_mse
                    else:
                        loss = loss_bce

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.stage2_model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                except Exception as e:
                    continue

                pbar.set_postfix({'loss': f"{total_loss/(i+1):.4f}"})

            # Validate
            val_metrics = self._validate_stage2(val_graphs, val_labels)
            self.history['stage2']['train_loss'].append(total_loss / len(train_graphs))
            self.history['stage2']['val_f1'].append(val_metrics['f1'])
            self.history['stage2']['val_decay_mae'].append(val_metrics['decay_mae'])

            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_graphs):.4f}, "
                  f"Val F1={val_metrics['f1']:.4f}, Decay MAE={val_metrics['decay_mae']:.4f}")

            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(self.stage2_model.state_dict(),
                          self.checkpoint_dir / 'stage2_best.pth')

        print(f"\nStage 2 Complete! Best F1: {best_f1:.4f}")
        return {'best_f1': best_f1}

    def _transfer_stage1_to_stage2(self):
        """Transfer learned weights from Stage 1 to Stage 2."""
        stage1_dict = self.stage1_model.state_dict()
        stage2_dict = self.stage2_model.state_dict()

        # Find matching keys
        transferred = 0
        for key in stage2_dict:
            if key in stage1_dict and stage1_dict[key].shape == stage2_dict[key].shape:
                stage2_dict[key] = stage1_dict[key]
                transferred += 1

        self.stage2_model.load_state_dict(stage2_dict)
        print(f"Transferred {transferred} parameter tensors")

    def _validate_stage2(self, graphs, labels) -> Dict:
        """Validate Stage 2 model."""
        self.stage2_model.eval()
        preds = []
        decay_errors = []

        with torch.no_grad():
            for graph, label in zip(graphs, labels):
                graph = graph.to(self.device)
                try:
                    output = self.stage2_model(graph)
                    pred = 1 if output['decay_score'].item() > 0.5 else 0
                    preds.append(pred)

                    if hasattr(graph, 'decay_score'):
                        error = abs(output['decay_score'].item() - graph.decay_score.item())
                        decay_errors.append(error)
                except:
                    preds.append(0)

        preds = np.array(preds)
        labels_arr = np.array(labels)

        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        decay_mae = np.mean(decay_errors) if decay_errors else 0.0

        return {'f1': f1, 'decay_mae': decay_mae}

    # =========================================================================
    # STAGE 3: Turn-Level Evaluation
    # =========================================================================

    def stage3_evaluate(self, test_graphs: List, test_conversations: List) -> Dict:
        """
        Stage 3: Evaluate turn-level prediction accuracy.

        Goal: Can we predict the exact turn where AI capitulated?
        """
        print("\n" + "=" * 70)
        print("STAGE 3: Turn-Level Evaluation")
        print("=" * 70)
        print("Goal: Predict exact turn where AI gave in to manipulation")

        if self.stage2_model is None:
            print("Error: Run Stage 2 first!")
            return {}

        self.stage2_model.eval()

        results = []

        with torch.no_grad():
            for graph, conv in tqdm(zip(test_graphs, test_conversations),
                                    total=len(test_graphs),
                                    desc="Turn-level evaluation"):
                graph = graph.to(self.device)

                try:
                    pred = self.stage2_model.predict(graph)

                    # Find predicted decay turn (highest per-turn score)
                    if pred['per_turn_sycophancy']:
                        pred_decay_turn = np.argmax(pred['per_turn_sycophancy'])
                    else:
                        pred_decay_turn = -1

                    # Ground truth decay turn (if available)
                    if hasattr(graph, 'stance_sequence'):
                        stances = graph.stance_sequence.tolist()
                        # Find first significant shift toward agreement
                        true_decay_turn = -1
                        for i in range(1, len(stances)):
                            if stances[i] - stances[i-1] >= 1:  # Shift toward agreement
                                true_decay_turn = i
                                break
                    else:
                        true_decay_turn = -1

                    results.append({
                        'pred_turn': pred_decay_turn,
                        'true_turn': true_decay_turn,
                        'decay_score': pred['decay_score'],
                        'pattern': pred['decay_pattern']
                    })

                except Exception as e:
                    results.append({'error': str(e)})

        # Compute turn-level accuracy
        valid_results = [r for r in results if 'pred_turn' in r and r['true_turn'] >= 0]

        if valid_results:
            exact_matches = sum(1 for r in valid_results if r['pred_turn'] == r['true_turn'])
            turn_accuracy = exact_matches / len(valid_results)

            turn_errors = [abs(r['pred_turn'] - r['true_turn']) for r in valid_results]
            turn_mae = np.mean(turn_errors)

            self.history['stage3']['turn_accuracy'].append(turn_accuracy)
            self.history['stage3']['turn_mae'].append(turn_mae)

            print(f"\nTurn-Level Results:")
            print(f"  Exact match accuracy: {turn_accuracy:.4f}")
            print(f"  Mean absolute error: {turn_mae:.2f} turns")

            return {'turn_accuracy': turn_accuracy, 'turn_mae': turn_mae}
        else:
            print("No valid results for turn-level evaluation")
            return {}

    # =========================================================================
    # FULL CURRICULUM
    # =========================================================================

    def run_full_curriculum(self, stage1_epochs: int = 10,
                           stage2_epochs: int = 20,
                           n_multiturn: int = 300) -> Dict:
        """Run the full three-stage curriculum."""
        print("\n" + "=" * 70)
        print("CONVOGRAPH CURRICULUM TRAINING")
        print("=" * 70)

        results = {}

        # Stage 1
        stage1_results = self.stage1_train(epochs=stage1_epochs)
        results['stage1'] = stage1_results

        # Stage 2
        stage2_results = self.stage2_train(epochs=stage2_epochs, n_synthetic=n_multiturn)
        results['stage2'] = stage2_results

        # Stage 3 (using validation data from Stage 2)
        print("\nPreparing Stage 3 test data...")
        loader = TruthDecayDataLoader()
        test_convs, test_labels = loader.generate_synthetic_multiturn(n_samples=50)
        builder = BeliefDecayGraphBuilder(self.embed_model)
        test_graphs = builder.build_batch_graphs(test_convs, test_labels)

        stage3_results = self.stage3_evaluate(test_graphs, test_convs)
        results['stage3'] = stage3_results

        # Save results
        results['history'] = self.history
        with open(self.checkpoint_dir / 'curriculum_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("CURRICULUM TRAINING COMPLETE")
        print("=" * 70)
        print(f"Stage 1 Best F1: {results['stage1'].get('best_f1', 'N/A')}")
        print(f"Stage 2 Best F1: {results['stage2'].get('best_f1', 'N/A')}")
        print(f"Stage 3 Turn Accuracy: {results['stage3'].get('turn_accuracy', 'N/A')}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Curriculum Training for ConvoGraph")

    parser.add_argument('--hidden-dim', type=int, default=384)
    parser.add_argument('--stage1-epochs', type=int, default=10)
    parser.add_argument('--stage2-epochs', type=int, default=20)
    parser.add_argument('--n-multiturn', type=int, default=300)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_curriculum')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    trainer = CurriculumTrainer(
        hidden_dim=args.hidden_dim,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer.run_full_curriculum(
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        n_multiturn=args.n_multiturn
    )


if __name__ == "__main__":
    main()
