"""
ConvoGraph: Sycophancy Detection Model using Hybrid Local-Global Graph Transformers

This module implements the core GNN model for detecting sycophancy in conversations.
It extends the HGL (Heterogeneous Global-Local) architecture from the original
multi-hop reasoning work to detect manipulation patterns in AI conversations.

Key Components:
1. HGT (Heterogeneous Graph Transformer): Captures local manipulation tactics
2. SGFormer: Captures long-range strategic patterns across conversation turns
3. Conversation-aware attention: Focuses on agreement/disagreement patterns
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, global_mean_pool, global_max_pool
from torch_geometric.nn.models import SGFormer
from torch_geometric.utils import scatter
from typing import List, Dict, Optional, Tuple


class ConversationQueryInteraction(nn.Module):
    """
    Conversation-aware attention module for sycophancy detection.

    Instead of query-passage attention (from RAG), this computes
    attention between user turns and AI responses to detect alignment patterns.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project user and AI embeddings into shared space
        self.user_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.ai_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Cross-attention: AI attends to User
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Alignment scoring head
        self.alignment_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_reps: torch.Tensor, ai_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment scores between AI and User turns.

        Args:
            user_reps: [num_user_turns, hidden_dim] - User turn representations
            ai_reps: [num_ai_turns, hidden_dim] - AI turn representations

        Returns:
            alignment_scores: [num_ai_turns] - How much each AI turn aligns with user
            ai_contextualized: [num_ai_turns, hidden_dim] - AI reps with user context
        """
        # Project into shared space
        user_proj = self.user_projector(user_reps)  # [num_user, dim]
        ai_proj = self.ai_projector(ai_reps)  # [num_ai, dim]

        # Add batch dimension for attention
        # AI queries attend to all user turns as keys/values
        ai_query = ai_proj.unsqueeze(0)  # [1, num_ai, dim]
        user_kv = user_proj.unsqueeze(0)  # [1, num_user, dim]

        # Cross attention
        ai_contextualized, attn_weights = self.cross_attention(ai_query, user_kv, user_kv)
        ai_contextualized = ai_contextualized.squeeze(0)  # [num_ai, dim]

        # Compute alignment scores
        # Concatenate original AI rep with user-contextualized rep
        combined = torch.cat([ai_proj, ai_contextualized], dim=-1)  # [num_ai, dim*2]
        alignment_scores = self.alignment_head(combined).squeeze(-1)  # [num_ai]

        return torch.sigmoid(alignment_scores), ai_contextualized


class HeterogeneousConvoGNN(nn.Module):
    """
    Heterogeneous GNN backbone for conversation graphs.

    Uses HGTConv for local message passing across different node and edge types.
    """

    def __init__(self, metadata, hidden_dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # HGT layers for heterogeneous message passing
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))

        # Layer normalization for each node type
        self.norms = nn.ModuleDict()
        for node_type in metadata[0]:
            self.norms[node_type] = nn.LayerNorm(hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HGT layers.

        Args:
            x_dict: Dict mapping node types to feature tensors
            edge_index_dict: Dict mapping edge types to edge indices

        Returns:
            h_dict: Dict mapping node types to updated representations
        """
        h_dict = x_dict

        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            # Apply activation and normalization
            h_dict = {
                k: self.norms[k](F.gelu(v)) if k in self.norms else F.gelu(v)
                for k, v in h_dict.items()
            }

        return h_dict


class GlobalConvoReasoner(nn.Module):
    """
    Global reasoning module using SGFormer for capturing long-range patterns.

    This allows the model to detect manipulation strategies that span
    multiple conversation turns.
    """

    def __init__(self, hidden_dim: int, num_layers: int = 1, num_heads: int = 4):
        super().__init__()

        self.global_transformer = SGFormer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            trans_num_layers=num_layers,
            trans_num_heads=num_heads,
            gnn_num_layers=0,  # Only use global transformer
            graph_weight=0.0   # Only use global transformer
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.3))  # Learnable mixing weight

    def forward(self, local_reps: torch.Tensor) -> torch.Tensor:
        """
        Apply global attention to capture long-range dependencies.

        Args:
            local_reps: [num_nodes, hidden_dim] - Local representations from HGT

        Returns:
            global_reps: [num_nodes, hidden_dim] - Globally contextualized representations
        """
        # Create dummy batch tensor (all nodes in same graph)
        batch = local_reps.new_zeros(local_reps.size(0), dtype=torch.long)

        # Apply global transformer
        global_reps = self.global_transformer.trans_conv(local_reps, batch)

        # Mix local and global representations
        alpha = torch.sigmoid(self.alpha)
        mixed = self.norm((1 - alpha) * local_reps + alpha * global_reps)

        return mixed


class SycophancyDetector(nn.Module):
    """
    Main model for sycophancy detection in conversations.

    Architecture:
    1. HGT backbone for local pattern detection (immediate sycophantic responses)
    2. SGFormer for global pattern detection (strategic manipulation over turns)
    3. Conversation interaction module for alignment detection
    4. Classification head for sycophancy prediction
    """

    def __init__(self, metadata, hidden_dim: int = 384, num_hgt_layers: int = 2,
                 num_global_layers: int = 1, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.metadata = metadata

        # Local backbone (HGT)
        self.local_backbone = HeterogeneousConvoGNN(
            metadata, hidden_dim, num_layers=num_hgt_layers, num_heads=num_heads
        )

        # Global reasoner (SGFormer) - applied to AI turns
        self.global_reasoner = GlobalConvoReasoner(
            hidden_dim, num_layers=num_global_layers, num_heads=num_heads
        )

        # Conversation interaction
        self.interaction = ConversationQueryInteraction(hidden_dim, num_heads)

        # Pooling layer norms
        self.user_pool_norm = nn.LayerNorm(hidden_dim)
        self.ai_pool_norm = nn.LayerNorm(hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # user_pool + ai_pool + ai_global + alignment_features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification
        )

        # Auxiliary heads for interpretability
        self.alignment_predictor = nn.Linear(hidden_dim, 1)
        self.turn_sycophancy_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, data: HeteroData, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sycophancy detection.

        Args:
            data: HeteroData graph representing a conversation
            return_attention: Whether to return attention weights for interpretability

        Returns:
            Dict containing:
                - logits: [1] classification logits
                - probs: [1] classification probability
                - alignment_scores: [num_ai_turns] per-turn alignment scores
                - turn_scores: [num_ai_turns] per-turn sycophancy scores
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. Local message passing (HGT)
        h_dict = self.local_backbone(x_dict, edge_index_dict)

        # 2. Extract user and AI representations
        user_reps = h_dict.get('user_turn', None)
        ai_reps = h_dict.get('ai_turn', None)

        if user_reps is None or ai_reps is None:
            raise ValueError("Graph must have 'user_turn' and 'ai_turn' nodes")

        # 3. Apply global reasoning to AI turns
        ai_global = self.global_reasoner(ai_reps)

        # 4. Conversation interaction (alignment detection)
        alignment_scores, ai_contextualized = self.interaction(user_reps, ai_global)

        # 5. Pool representations for graph-level prediction
        # Mean pooling
        user_pooled = self.user_pool_norm(user_reps.mean(dim=0, keepdim=True))
        ai_pooled = self.ai_pool_norm(ai_global.mean(dim=0, keepdim=True))

        # Alignment-weighted pooling of AI turns
        alignment_weights = F.softmax(alignment_scores, dim=0)
        ai_alignment_pooled = (ai_contextualized * alignment_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        # Alignment feature (mean alignment score as feature)
        alignment_feature = alignment_scores.mean().unsqueeze(0).unsqueeze(0).expand(-1, self.hidden_dim)

        # 6. Concatenate features for classification
        combined = torch.cat([
            user_pooled,
            ai_pooled,
            ai_alignment_pooled,
            alignment_feature
        ], dim=-1)  # [1, hidden_dim * 4]

        # 7. Classification
        logits = self.classifier(combined).squeeze()  # []
        probs = torch.sigmoid(logits)

        # 8. Per-turn predictions (auxiliary)
        turn_scores = torch.sigmoid(self.turn_sycophancy_predictor(ai_contextualized).squeeze(-1))

        output = {
            'logits': logits,
            'probs': probs,
            'alignment_scores': alignment_scores,
            'turn_scores': turn_scores
        }

        if return_attention:
            output['user_reps'] = user_reps
            output['ai_reps'] = ai_global

        return output

    def predict(self, data: HeteroData, threshold: float = 0.5) -> Dict[str, any]:
        """
        Make predictions with interpretable outputs.

        Args:
            data: HeteroData graph
            threshold: Classification threshold

        Returns:
            Dict with prediction and explanations
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(data, return_attention=True)

            prediction = {
                'is_sycophantic': output['probs'].item() > threshold,
                'confidence': output['probs'].item(),
                'per_turn_alignment': output['alignment_scores'].tolist(),
                'per_turn_sycophancy': output['turn_scores'].tolist(),
                'most_sycophantic_turn': output['turn_scores'].argmax().item()
            }

        return prediction


class SycophancyDetectorLite(nn.Module):
    """
    Lightweight version of SycophancyDetector for faster inference.

    Uses a simpler architecture without SGFormer global reasoning.
    Suitable for real-time monitoring applications.
    """

    def __init__(self, metadata, hidden_dim: int = 384, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Single HGT layer
        self.conv = HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
        self.norm = nn.LayerNorm(hidden_dim)

        # Simple attention between user and AI
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Simple forward pass returning classification logits."""
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Message passing
        h_dict = self.conv(x_dict, edge_index_dict)
        h_dict = {k: self.norm(F.gelu(v)) for k, v in h_dict.items()}

        # Pool
        user_pool = h_dict['user_turn'].mean(dim=0)
        ai_pool = h_dict['ai_turn'].mean(dim=0)

        # Classify
        combined = torch.cat([user_pool, ai_pool], dim=-1).unsqueeze(0)
        logits = self.classifier(combined).squeeze()

        return logits


def create_model(sample_graph: HeteroData, hidden_dim: int = 384,
                 lite: bool = False) -> nn.Module:
    """
    Factory function to create a SycophancyDetector model.

    Args:
        sample_graph: Example graph to extract metadata from
        hidden_dim: Hidden dimension size
        lite: Whether to use lightweight model

    Returns:
        Initialized model
    """
    metadata = sample_graph.metadata()

    if lite:
        return SycophancyDetectorLite(metadata, hidden_dim)
    else:
        return SycophancyDetector(metadata, hidden_dim)


if __name__ == "__main__":
    # Test the model with a dummy graph
    print("Testing SycophancyDetector...")

    # Create dummy graph
    data = HeteroData()

    hidden_dim = 384
    num_user = 3
    num_ai = 3
    num_topics = 2

    # Node features
    data['user_turn'].x = torch.randn(num_user, hidden_dim)
    data['ai_turn'].x = torch.randn(num_ai, hidden_dim)
    data['topic'].x = torch.randn(num_topics, hidden_dim)
    data['sentiment'].x = torch.randn(3, hidden_dim)  # pos, neg, neutral

    # Edge indices
    data['ai_turn', 'responds_to', 'user_turn'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data['user_turn', 'receives', 'ai_turn'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    data['user_turn', 'follows', 'user_turn'].edge_index = torch.tensor([[0, 1], [1, 2]])
    data['ai_turn', 'follows', 'ai_turn'].edge_index = torch.tensor([[0, 1], [1, 2]])
    data['ai_turn', 'agrees_with', 'user_turn'].edge_index = torch.tensor([[0, 1], [0, 1]])

    # Label
    data.y = torch.tensor([1])  # Sycophantic

    # Create model
    model = create_model(data, hidden_dim=hidden_dim)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    output = model(data)
    print(f"Output logits: {output['logits'].item():.4f}")
    print(f"Output probs: {output['probs'].item():.4f}")
    print(f"Alignment scores: {output['alignment_scores'].tolist()}")
    print(f"Turn scores: {output['turn_scores'].tolist()}")

    # Test prediction
    pred = model.predict(data)
    print(f"\nPrediction: {pred}")

    print("\nTest passed!")
