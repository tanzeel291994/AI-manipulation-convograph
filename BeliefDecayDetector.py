"""
BeliefDecayDetector: Multi-Turn Sycophancy Detection via Stance Trajectory Analysis

This model is specifically designed for detecting "belief decay" - the gradual
shift of an AI's position toward a user's incorrect belief over multiple turns.

Key Innovation:
--------------
Instead of classifying individual responses, we model the TRAJECTORY of the AI's
stance across the conversation. The SGFormer component can now actually do useful
work - detecting patterns in how stances shift over time.

Architecture:
------------

    ┌─────────────────────────────────────────────────────────────┐
    │                    Multi-Turn Conversation                   │
    └─────────────────────────────────────────────────────────────┘
                                   │
                                   v
    ┌─────────────────────────────────────────────────────────────┐
    │              Heterogeneous Graph Construction                │
    │  (User turns, AI turns, Stances, Pressures, Belief)         │
    └─────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    v                             v
    ┌───────────────────────┐     ┌───────────────────────────────┐
    │     HGT Backbone      │     │    Stance Trajectory Module    │
    │  (Local interactions) │     │  (SGFormer on stance sequence) │
    │                       │     │                                │
    │  - Pressure → Response│     │  - Stance_1 → Stance_2 → ...   │
    │  - Turn context       │     │  - Detects decay patterns      │
    │  - Belief grounding   │     │  - Long-range dependencies     │
    └───────────────────────┘     └───────────────────────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   v
    ┌─────────────────────────────────────────────────────────────┐
    │                  Belief Decay Classifier                     │
    │                                                              │
    │  Features:                                                   │
    │  - Aggregated turn representations                           │
    │  - Stance trajectory embedding                               │
    │  - Initial vs Final stance comparison                        │
    │  - Pressure-response alignment                               │
    └─────────────────────────────────────────────────────────────┘
                                   │
                                   v
                         [Decay Score: 0.0 - 1.0]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv
from torch_geometric.nn.models import SGFormer
from torch_geometric.utils import scatter
from typing import Dict, List, Optional, Tuple


class StanceTrajectoryEncoder(nn.Module):
    """
    Encodes the trajectory of AI stances across a conversation.

    This is where the SGFormer actually becomes useful - it can attend
    to all stances globally to detect decay patterns.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project stance embeddings
        self.stance_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # SGFormer for global stance trajectory analysis
        self.global_trajectory = SGFormer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=2,  # More layers for trajectory
            trans_num_heads=num_heads,
            gnn_num_layers=0,
            graph_weight=0.0
        )

        # Trajectory summarization
        self.trajectory_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # Decay pattern detector (1D conv over stance sequence)
        self.decay_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

    def forward(self, stance_features: torch.Tensor,
                stance_shift_edges: Optional[torch.Tensor] = None,
                shift_attrs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode stance trajectory.

        Args:
            stance_features: [num_stances, hidden_dim]
            stance_shift_edges: [2, num_shifts] edge index for stance transitions
            shift_attrs: [num_shifts, 3] attributes (direction, magnitude, toward_agree)

        Returns:
            trajectory_embedding: [1, hidden_dim]
        """
        num_stances = stance_features.size(0)

        if num_stances == 0:
            return torch.zeros(1, self.hidden_dim, device=stance_features.device)

        # Project stances
        h_stance = self.stance_projector(stance_features)  # [N, D]

        # Apply SGFormer for global context
        batch = torch.zeros(num_stances, dtype=torch.long, device=h_stance.device)
        h_global = self.global_trajectory.trans_conv(h_stance, batch)  # [N, D]

        # Mix local and global
        h_mixed = 0.7 * h_stance + 0.3 * h_global

        # Self-attention to get trajectory summary
        h_seq = h_mixed.unsqueeze(0)  # [1, N, D]
        traj_summary, attn_weights = self.trajectory_attention(h_seq, h_seq, h_seq)
        traj_summary = traj_summary.mean(dim=1)  # [1, D]

        # 1D conv for decay pattern detection
        h_conv_input = h_mixed.t().unsqueeze(0)  # [1, D, N]
        decay_features = self.decay_conv(h_conv_input)  # [1, D//2, N]
        decay_summary = decay_features.mean(dim=2)  # [1, D//2]

        # Combine
        combined = torch.cat([traj_summary, decay_summary], dim=-1)  # [1, D + D//2]
        output = self.output_proj(combined)  # [1, D]

        return output


class PressureResponseModule(nn.Module):
    """
    Models how user pressure affects AI responses.

    Key for understanding sycophancy: does the AI cave under pressure?
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Cross-attention: AI attending to user pressure
        self.pressure_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # Pressure-response alignment scoring
        self.alignment_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_features: torch.Tensor, ai_features: torch.Tensor,
                pressure_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Model pressure-response dynamics.

        Args:
            user_features: [num_user_turns, hidden_dim]
            ai_features: [num_ai_turns, hidden_dim]
            pressure_features: [num_pressures, hidden_dim]

        Returns:
            ai_contextualized: AI features with pressure context
            pressure_response_scores: How much AI yields to each pressure
        """
        # AI attends to user turns (pressure context)
        ai_seq = ai_features.unsqueeze(0)  # [1, N_ai, D]
        user_seq = user_features.unsqueeze(0)  # [1, N_user, D]

        ai_contextualized, _ = self.pressure_attention(ai_seq, user_seq, user_seq)
        ai_contextualized = ai_contextualized.squeeze(0)  # [N_ai, D]

        # Score pressure-response alignment
        # For each AI turn, how aligned is it with the corresponding user pressure?
        min_len = min(ai_features.size(0), user_features.size(0))
        if min_len > 0:
            combined = torch.cat([
                ai_features[:min_len],
                user_features[:min_len]
            ], dim=-1)  # [min_len, D*2]
            alignment_scores = self.alignment_scorer(combined).squeeze(-1)  # [min_len]
        else:
            alignment_scores = torch.zeros(0, device=ai_features.device)

        return ai_contextualized, torch.sigmoid(alignment_scores)


class BeliefDecayDetector(nn.Module):
    """
    Main model for detecting belief decay (multi-turn sycophancy).

    This architecture is specifically designed for the TRUTH DECAY benchmark
    and similar multi-turn sycophancy evaluation datasets.
    """

    def __init__(self, metadata, hidden_dim: int = 384, num_hgt_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.metadata = metadata

        # HGT backbone for local message passing
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_hgt_layers):
            self.hgt_layers.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

        # Layer norms for node types
        self.norms = nn.ModuleDict()
        for node_type in metadata[0]:
            self.norms[node_type] = nn.LayerNorm(hidden_dim)

        # Stance trajectory encoder (where SGFormer is actually useful!)
        self.trajectory_encoder = StanceTrajectoryEncoder(hidden_dim, num_heads)

        # Pressure-response module
        self.pressure_response = PressureResponseModule(hidden_dim, num_heads)

        # Belief grounding (how turns relate to central belief)
        self.belief_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Initial vs Final stance comparison
        self.stance_comparison = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Final classifier
        # Input: trajectory + pressure_response + belief_context + stance_comparison
        classifier_input_dim = hidden_dim * 3 + hidden_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Auxiliary: per-turn sycophancy predictor
        self.turn_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData, return_analysis: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for belief decay detection.

        Args:
            data: HeteroData multi-turn conversation graph
            return_analysis: Whether to return detailed analysis

        Returns:
            Dict with:
                - logits: [1] classification logit
                - decay_score: [1] predicted decay score
                - per_turn_scores: [num_ai_turns] per-turn sycophancy scores
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. HGT message passing
        h_dict = x_dict
        for hgt in self.hgt_layers:
            h_dict = hgt(h_dict, edge_index_dict)
            h_dict = {
                k: self.norms[k](F.gelu(v)) if k in self.norms else F.gelu(v)
                for k, v in h_dict.items()
            }

        # Extract node representations
        user_h = h_dict.get('user_turn', torch.zeros(1, self.hidden_dim, device=data.y.device if hasattr(data, 'y') else 'cpu'))
        ai_h = h_dict.get('ai_turn', torch.zeros(1, self.hidden_dim, device=user_h.device))
        stance_h = h_dict.get('stance', torch.zeros(1, self.hidden_dim, device=user_h.device))
        pressure_h = h_dict.get('pressure', torch.zeros(1, self.hidden_dim, device=user_h.device))
        belief_h = h_dict.get('belief', torch.zeros(1, self.hidden_dim, device=user_h.device))

        # 2. Stance trajectory encoding (THE KEY COMPONENT)
        stance_shift_edges = None
        shift_attrs = None
        if ('stance', 'shifts_to', 'stance') in edge_index_dict:
            stance_shift_edges = edge_index_dict[('stance', 'shifts_to', 'stance')]
            if hasattr(data['stance', 'shifts_to', 'stance'], 'edge_attr'):
                shift_attrs = data['stance', 'shifts_to', 'stance'].edge_attr

        trajectory_embedding = self.trajectory_encoder(
            stance_h, stance_shift_edges, shift_attrs
        )  # [1, D]

        # 3. Pressure-response modeling
        ai_contextualized, pressure_scores = self.pressure_response(
            user_h, ai_h, pressure_h
        )

        # Pool AI representations
        ai_pooled = ai_contextualized.mean(dim=0, keepdim=True)  # [1, D]

        # 4. Belief grounding
        # How do AI responses relate to the central belief?
        if belief_h.size(0) > 0 and ai_h.size(0) > 0:
            ai_seq = ai_h.unsqueeze(0)  # [1, N, D]
            belief_seq = belief_h.unsqueeze(0)  # [1, 1, D]
            belief_context, _ = self.belief_attention(ai_seq, belief_seq, belief_seq)
            belief_context = belief_context.mean(dim=1)  # [1, D]
        else:
            belief_context = torch.zeros(1, self.hidden_dim, device=user_h.device)

        # 5. Initial vs Final stance comparison
        if stance_h.size(0) >= 2:
            initial_stance = stance_h[0:1]  # [1, D]
            final_stance = stance_h[-1:]  # [1, D]
            stance_delta = self.stance_comparison(
                torch.cat([initial_stance, final_stance], dim=-1)
            )  # [1, D//2]
        else:
            stance_delta = torch.zeros(1, self.hidden_dim // 2, device=user_h.device)

        # 6. Combine all features
        combined = torch.cat([
            trajectory_embedding,  # [1, D] - stance trajectory
            ai_pooled,            # [1, D] - AI response summary
            belief_context,       # [1, D] - belief grounding
            stance_delta          # [1, D//2] - initial vs final stance
        ], dim=-1)  # [1, D*3 + D//2]

        # 7. Classification
        logits = self.classifier(combined).squeeze()  # []
        decay_score = torch.sigmoid(logits)

        # 8. Per-turn predictions
        per_turn_scores = torch.sigmoid(self.turn_predictor(ai_contextualized).squeeze(-1))

        output = {
            'logits': logits,
            'decay_score': decay_score,
            'per_turn_scores': per_turn_scores,
            'pressure_response_scores': pressure_scores
        }

        if return_analysis:
            output['trajectory_embedding'] = trajectory_embedding
            output['stance_representations'] = stance_h
            output['ai_contextualized'] = ai_contextualized

        return output

    def predict(self, data: HeteroData, threshold: float = 0.5) -> Dict:
        """Make interpretable prediction."""
        self.eval()
        with torch.no_grad():
            output = self.forward(data, return_analysis=True)

            # Get stance sequence if available
            if hasattr(data, 'stance_sequence'):
                stance_seq = data.stance_sequence.tolist()
            else:
                stance_seq = []

            prediction = {
                'is_sycophantic': output['decay_score'].item() > threshold,
                'decay_score': output['decay_score'].item(),
                'confidence': abs(output['decay_score'].item() - 0.5) * 2,  # 0-1
                'stance_sequence': stance_seq,
                'per_turn_sycophancy': output['per_turn_scores'].tolist(),
                'pressure_response_alignment': output['pressure_response_scores'].tolist() if output['pressure_response_scores'].numel() > 0 else [],
            }

            # Analyze decay pattern
            if len(stance_seq) >= 2:
                initial = stance_seq[0]
                final = stance_seq[-1]
                if initial <= 1 and final >= 3:
                    prediction['decay_pattern'] = "STRONG_DECAY: Started disagreeing, ended agreeing"
                elif final > initial:
                    prediction['decay_pattern'] = f"GRADUAL_DECAY: Stance shifted +{final - initial} toward agreement"
                elif final < initial:
                    prediction['decay_pattern'] = "STRENGTHENING: AI became more firm in disagreement"
                else:
                    prediction['decay_pattern'] = "STABLE: AI maintained consistent stance"
            else:
                prediction['decay_pattern'] = "INSUFFICIENT_DATA: Need more turns to analyze"

        return prediction


def create_belief_decay_model(sample_graph: HeteroData, hidden_dim: int = 384) -> BeliefDecayDetector:
    """Factory function to create model from sample graph."""
    metadata = sample_graph.metadata()
    return BeliefDecayDetector(metadata, hidden_dim)


if __name__ == "__main__":
    print("Testing BeliefDecayDetector")
    print("=" * 60)

    # Create dummy multi-turn graph
    data = HeteroData()
    hidden_dim = 384
    num_turns = 5

    # Node features
    data['user_turn'].x = torch.randn(num_turns, hidden_dim)
    data['ai_turn'].x = torch.randn(num_turns, hidden_dim)
    data['stance'].x = torch.randn(num_turns, hidden_dim)
    data['pressure'].x = torch.randn(num_turns, hidden_dim)
    data['belief'].x = torch.randn(1, hidden_dim)

    # Edges
    # Temporal
    data['user_turn', 'follows', 'user_turn'].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )
    data['ai_turn', 'follows', 'ai_turn'].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )

    # Response
    data['ai_turn', 'responds_to', 'user_turn'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    )

    # Stance
    data['ai_turn', 'takes_stance', 'stance'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    )

    # Stance shifts (KEY!)
    data['stance', 'shifts_to', 'stance'].edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]]
    )
    # Simulating gradual decay: shifts toward agreement
    data['stance', 'shifts_to', 'stance'].edge_attr = torch.tensor([
        [1, 1, 1],  # shift +1
        [1, 1, 1],  # shift +1
        [0, 0, 0],  # no shift
        [1, 1, 1],  # shift +1
    ], dtype=torch.float)

    # Belief edges
    data['user_turn', 'debates', 'belief'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]
    )
    data['ai_turn', 'debates', 'belief'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 0, 0, 0, 0]]
    )

    # Pressure
    data['user_turn', 'pressures', 'ai_turn'].edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    )

    # Labels
    data.y = torch.tensor([1])  # Sycophantic
    data.stance_sequence = torch.tensor([1, 2, 2, 3, 4])  # Decay pattern

    # Create model
    model = create_belief_decay_model(data, hidden_dim=hidden_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    output = model(data)
    print(f"\nOutput decay score: {output['decay_score'].item():.4f}")
    print(f"Per-turn scores: {output['per_turn_scores'].tolist()}")

    # Prediction
    pred = model.predict(data)
    print(f"\nPrediction:")
    print(f"  Is sycophantic: {pred['is_sycophantic']}")
    print(f"  Decay score: {pred['decay_score']:.4f}")
    print(f"  Stance sequence: {pred['stance_sequence']}")
    print(f"  Decay pattern: {pred['decay_pattern']}")

    print("\nTest passed!")
