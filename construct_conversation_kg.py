"""
ConvoGraph: Conversation Knowledge Graph Builder for Sycophancy Detection

This module transforms AI conversations into heterogeneous knowledge graphs
for detecting manipulation patterns using Graph Neural Networks.

Node Types:
- USER_TURN: User utterances in the conversation
- AI_TURN: AI responses in the conversation
- TOPIC: Extracted topics/claims from utterances
- SENTIMENT: Sentiment markers (positive/negative/neutral)

Edge Types:
- RESPONDS_TO: AI_TURN -> USER_TURN (response relationship)
- FOLLOWS: Sequential utterances (turn order)
- DISCUSSES: Utterance -> TOPIC
- EXPRESSES: Utterance -> SENTIMENT
- AGREES_WITH: AI aligns with user belief
- CONTRADICTS: AI contradicts user belief
"""

import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from hashlib import md5
from torch_geometric.data import HeteroData
import torch.nn.functional as F


class ConversationGraphBuilder:
    """
    Builds heterogeneous knowledge graphs from conversations for sycophancy detection.

    The graph captures:
    1. Local patterns: Immediate sycophantic responses (via HGT)
    2. Global patterns: Long-range manipulation strategies (via SGFormer)
    """

    # Sentiment labels
    SENTIMENT_POSITIVE = "positive"
    SENTIMENT_NEGATIVE = "negative"
    SENTIMENT_NEUTRAL = "neutral"

    # Alignment labels
    ALIGNMENT_AGREES = "agrees"
    ALIGNMENT_CONTRADICTS = "contradicts"
    ALIGNMENT_NEUTRAL = "neutral"

    def __init__(self, embedding_model, sentiment_analyzer=None, topic_extractor=None):
        """
        Args:
            embedding_model: Model for generating text embeddings (e.g., BGE, SentenceTransformer)
            sentiment_analyzer: Optional sentiment analysis model (defaults to simple rule-based)
            topic_extractor: Optional topic extraction model (defaults to keyword extraction)
        """
        self.embed_model = embedding_model
        self.sentiment_analyzer = sentiment_analyzer
        self.topic_extractor = topic_extractor

        # Caches
        self.topic_to_idx = {}
        self.sentiment_to_idx = {
            self.SENTIMENT_POSITIVE: 0,
            self.SENTIMENT_NEGATIVE: 1,
            self.SENTIMENT_NEUTRAL: 2
        }
        self.unique_topics = []

    def analyze_sentiment(self, text: str) -> str:
        """
        Simple rule-based sentiment analysis.
        Can be replaced with a proper sentiment model.
        """
        if self.sentiment_analyzer:
            return self.sentiment_analyzer(text)

        # Simple keyword-based sentiment
        positive_words = {'good', 'great', 'excellent', 'right', 'correct', 'agree',
                        'yes', 'absolutely', 'wonderful', 'fantastic', 'love', 'best',
                        'helpful', 'thank', 'thanks', 'appreciate', 'perfect'}
        negative_words = {'bad', 'wrong', 'incorrect', 'disagree', 'no', 'terrible',
                        'awful', 'hate', 'worst', 'poor', 'mistake', 'error', 'fail'}

        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            return self.SENTIMENT_POSITIVE
        elif neg_count > pos_count:
            return self.SENTIMENT_NEGATIVE
        return self.SENTIMENT_NEUTRAL

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics/claims from text.
        Can be replaced with proper NER or topic modeling.
        """
        if self.topic_extractor:
            return self.topic_extractor(text)

        # Simple noun phrase extraction using patterns
        # In production, use spaCy or similar
        topics = []

        # Extract capitalized phrases (potential named entities)
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text)
        topics.extend(named_entities[:3])  # Limit to top 3

        # Extract quoted phrases (claims/opinions)
        quoted = re.findall(r'"([^"]+)"', text)
        topics.extend(quoted[:2])

        # Extract key noun phrases (simplified)
        # Look for patterns like "the X", "a X" followed by verbs
        simple_claims = re.findall(r'\b(?:is|are|was|were|should|must|will)\s+(\w+(?:\s+\w+)?)', text.lower())
        topics.extend([c.strip() for c in simple_claims[:2]])

        return list(set(topics))[:5]  # Deduplicate and limit

    def detect_alignment(self, user_text: str, ai_text: str) -> str:
        """
        Detect whether AI response aligns with, contradicts, or is neutral to user's statement.
        This is a key signal for sycophancy detection.
        """
        user_sentiment = self.analyze_sentiment(user_text)
        ai_sentiment = self.analyze_sentiment(ai_text)

        # Check for explicit agreement/disagreement patterns
        ai_lower = ai_text.lower()

        agreement_patterns = [
            r'\byou\'re right\b', r'\bi agree\b', r'\babsolutely\b', r'\bexactly\b',
            r'\bthat\'s correct\b', r'\bgood point\b', r'\byou make a good point\b',
            r'\bi think you\'re right\b', r'\bthat\'s true\b', r'\bi couldn\'t agree more\b'
        ]

        disagreement_patterns = [
            r'\bi disagree\b', r'\bthat\'s not\b', r'\bactually\b', r'\bhowever\b',
            r'\bbut\b', r'\bi\'m not sure\b', r'\bthat\'s incorrect\b', r'\bnot quite\b',
            r'\bi don\'t think\b', r'\bthat\'s wrong\b'
        ]

        agree_score = sum(1 for p in agreement_patterns if re.search(p, ai_lower))
        disagree_score = sum(1 for p in disagreement_patterns if re.search(p, ai_lower))

        # Sentiment alignment check
        if user_sentiment == ai_sentiment and user_sentiment != self.SENTIMENT_NEUTRAL:
            agree_score += 1
        elif user_sentiment != ai_sentiment and user_sentiment != self.SENTIMENT_NEUTRAL and ai_sentiment != self.SENTIMENT_NEUTRAL:
            disagree_score += 1

        if agree_score > disagree_score:
            return self.ALIGNMENT_AGREES
        elif disagree_score > agree_score:
            return self.ALIGNMENT_CONTRADICTS
        return self.ALIGNMENT_NEUTRAL

    def build_conversation_graph(self, conversation: List[Dict[str, str]],
                                  label: Optional[int] = None) -> HeteroData:
        """
        Build a heterogeneous graph from a single conversation.

        Args:
            conversation: List of turns, each with 'role' ('user'/'assistant') and 'content'
            label: Optional sycophancy label (1 = sycophantic, 0 = non-sycophantic)

        Returns:
            HeteroData graph representing the conversation
        """
        data = HeteroData()

        # Separate turns by role
        user_turns = []
        ai_turns = []
        all_turns = []  # Ordered list of (role, idx, text)

        for turn in conversation:
            role = turn.get('role', turn.get('speaker', '')).lower()
            content = turn.get('content', turn.get('text', ''))

            if role in ['user', 'human']:
                user_turns.append(content)
                all_turns.append(('user', len(user_turns) - 1, content))
            elif role in ['assistant', 'ai', 'bot', 'model']:
                ai_turns.append(content)
                all_turns.append(('ai', len(ai_turns) - 1, content))

        if not user_turns or not ai_turns:
            return None

        # Collect all topics
        all_topics = []
        turn_topics = []  # List of (turn_type, turn_idx, topic_list)

        for role, idx, text in all_turns:
            topics = self.extract_topics(text)
            turn_topics.append((role, idx, topics))
            all_topics.extend(topics)

        # Build unique topic list
        unique_topics = list(set(all_topics))
        topic_to_idx = {t: i for i, t in enumerate(unique_topics)}

        # Generate embeddings
        user_embeddings = self.embed_model.batch_encode(user_turns, instruction="passage")
        ai_embeddings = self.embed_model.batch_encode(ai_turns, instruction="passage")

        if unique_topics:
            topic_embeddings = self.embed_model.batch_encode(unique_topics, instruction="passage")
        else:
            topic_embeddings = np.zeros((0, user_embeddings.shape[1]))

        # Sentiment embeddings (learned representations)
        sentiment_texts = [self.SENTIMENT_POSITIVE, self.SENTIMENT_NEGATIVE, self.SENTIMENT_NEUTRAL]
        sentiment_embeddings = self.embed_model.batch_encode(sentiment_texts, instruction="passage")

        # Set node features
        hidden_dim = user_embeddings.shape[1]
        data['user_turn'].x = torch.from_numpy(user_embeddings).float()
        data['ai_turn'].x = torch.from_numpy(ai_embeddings).float()

        # Always create at least 1 topic node for consistent metadata
        if len(unique_topics) > 0:
            data['topic'].x = torch.from_numpy(topic_embeddings).float()
        else:
            # Create a dummy topic node
            unique_topics = ["[NO_TOPIC]"]
            topic_to_idx = {"[NO_TOPIC]": 0}
            data['topic'].x = torch.zeros(1, hidden_dim)

        data['sentiment'].x = torch.from_numpy(sentiment_embeddings).float()

        # Build edges
        responds_to_edges = []  # AI -> User
        follows_edges_user = []  # User -> User (sequential)
        follows_edges_ai = []  # AI -> AI (sequential)
        discusses_user = []  # User -> Topic
        discusses_ai = []  # AI -> Topic
        expresses_user = []  # User -> Sentiment
        expresses_ai = []  # AI -> Sentiment
        agrees_edges = []  # AI -> User (when agreement detected)
        contradicts_edges = []  # AI -> User (when contradiction detected)

        # Process turns
        prev_user_idx = None
        prev_ai_idx = None

        for i, (role, idx, text) in enumerate(all_turns):
            sentiment = self.analyze_sentiment(text)
            sentiment_idx = self.sentiment_to_idx[sentiment]

            if role == 'user':
                # Sequential edges
                if prev_user_idx is not None:
                    follows_edges_user.append([prev_user_idx, idx])
                prev_user_idx = idx

                # Sentiment edges
                expresses_user.append([idx, sentiment_idx])

                # Topic edges
                topics = self.extract_topics(text)
                for t in topics:
                    if t in topic_to_idx:
                        discusses_user.append([idx, topic_to_idx[t]])

            elif role == 'ai':
                # Sequential edges
                if prev_ai_idx is not None:
                    follows_edges_ai.append([prev_ai_idx, idx])
                prev_ai_idx = idx

                # Sentiment edges
                expresses_ai.append([idx, sentiment_idx])

                # Topic edges
                topics = self.extract_topics(text)
                for t in topics:
                    if t in topic_to_idx:
                        discusses_ai.append([idx, topic_to_idx[t]])

                # Response edges (AI responds to previous user turn)
                if i > 0 and all_turns[i-1][0] == 'user':
                    user_idx = all_turns[i-1][1]
                    user_text = all_turns[i-1][2]

                    responds_to_edges.append([idx, user_idx])

                    # NOTE: We intentionally do NOT use rule-based alignment detection here
                    # because it causes LABEL LEAKAGE - the patterns we use to detect
                    # agreement ("you're right", "i agree") are the same patterns that
                    # define sycophantic responses in the dataset!
                    #
                    # Instead, let the GNN learn alignment patterns from embeddings.
                    # The agrees_with/contradicts edges are left empty - the model must
                    # learn from the actual text representations.
                    pass

        # Set edge indices - ALWAYS create all edge types for consistent metadata
        # This is critical for HGT which expects consistent edge types across batches

        # Helper to create edge tensor (empty if no edges)
        def make_edge_index(edges):
            if edges:
                return torch.tensor(edges).t().contiguous()
            else:
                return torch.zeros((2, 0), dtype=torch.long)

        # Response edges (always present for valid conversations)
        data['ai_turn', 'responds_to', 'user_turn'].edge_index = make_edge_index(responds_to_edges)
        data['user_turn', 'receives', 'ai_turn'].edge_index = make_edge_index(
            [[e[1], e[0]] for e in responds_to_edges] if responds_to_edges else []
        )

        # Sequential edges
        data['user_turn', 'follows', 'user_turn'].edge_index = make_edge_index(follows_edges_user)
        data['ai_turn', 'follows', 'ai_turn'].edge_index = make_edge_index(follows_edges_ai)

        # Topic edges (always create - we now always have at least 1 topic node)
        data['user_turn', 'discusses', 'topic'].edge_index = make_edge_index(discusses_user)
        data['topic', 'discussed_by', 'user_turn'].edge_index = make_edge_index(
            [[e[1], e[0]] for e in discusses_user] if discusses_user else []
        )
        data['ai_turn', 'discusses', 'topic'].edge_index = make_edge_index(discusses_ai)
        data['topic', 'discussed_by', 'ai_turn'].edge_index = make_edge_index(
            [[e[1], e[0]] for e in discusses_ai] if discusses_ai else []
        )

        # Sentiment edges
        data['user_turn', 'expresses', 'sentiment'].edge_index = make_edge_index(expresses_user)
        data['ai_turn', 'expresses', 'sentiment'].edge_index = make_edge_index(expresses_ai)

        # Agreement/Contradiction edges - ALWAYS create even if empty
        data['ai_turn', 'agrees_with', 'user_turn'].edge_index = make_edge_index(agrees_edges)
        data['ai_turn', 'contradicts', 'user_turn'].edge_index = make_edge_index(contradicts_edges)

        # Store label if provided
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)

        # Store metadata
        data.num_user_turns = len(user_turns)
        data.num_ai_turns = len(ai_turns)
        data.num_topics = len(unique_topics)

        return data

    def build_batch_graphs(self, conversations: List[Dict],
                           labels: Optional[List[int]] = None) -> List[HeteroData]:
        """
        Build graphs for a batch of conversations.

        Args:
            conversations: List of conversation dicts with 'turns' key
            labels: Optional list of labels for each conversation

        Returns:
            List of HeteroData graphs
        """
        graphs = []

        if labels is None:
            labels = [None] * len(conversations)

        for conv, label in tqdm(zip(conversations, labels), total=len(conversations),
                                desc="Building conversation graphs"):
            turns = conv.get('turns', conv.get('messages', conv.get('conversation', [])))

            if isinstance(turns, list) and len(turns) > 0:
                graph = self.build_conversation_graph(turns, label)
                if graph is not None:
                    graphs.append(graph)

        return graphs


class SimpleEmbeddingModel:
    """
    Simple wrapper for embedding models compatible with ConversationGraphBuilder.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.model.eval()

    def batch_encode(self, texts: List[str], instruction: str = "passage",
                     batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()))

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return embeddings


# Utility functions
def conversation_from_pair(user_text: str, ai_text: str) -> List[Dict[str, str]]:
    """Create a conversation from a simple user-AI pair."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ai_text}
    ]


def load_anthropic_sycophancy_data(data_path: str) -> Tuple[List[Dict], List[int]]:
    """
    Load data from Anthropic sycophancy evaluation format.

    Returns:
        Tuple of (conversations, labels)
    """
    conversations = []
    labels = []

    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)

                # Handle different formats
                if 'sycophantic_response' in item and 'non_sycophantic_response' in item:
                    # Paired format - create two samples
                    user_text = item.get('human', item.get('prompt', ''))

                    # Sycophantic sample
                    conv_syc = conversation_from_pair(user_text, item['sycophantic_response'])
                    conversations.append({'turns': conv_syc, 'id': f"{item.get('id', '')}_syc"})
                    labels.append(1)

                    # Non-sycophantic sample
                    conv_non = conversation_from_pair(user_text, item['non_sycophantic_response'])
                    conversations.append({'turns': conv_non, 'id': f"{item.get('id', '')}_non"})
                    labels.append(0)

                elif 'is_sycophantic' in item:
                    # Single labeled sample
                    turns = item.get('turns', item.get('messages', []))
                    conversations.append({'turns': turns, 'id': item.get('id', '')})
                    labels.append(1 if item['is_sycophantic'] else 0)

    return conversations, labels


if __name__ == "__main__":
    # Example usage
    print("ConvoGraph: Conversation Graph Builder for Sycophancy Detection")
    print("=" * 60)

    # Initialize embedding model
    embed_model = SimpleEmbeddingModel("BAAI/bge-small-en-v1.5")

    # Initialize graph builder
    builder = ConversationGraphBuilder(embed_model)

    # Example conversation (sycophantic)
    sycophantic_conv = [
        {"role": "user", "content": "I think the Earth is flat. What do you think?"},
        {"role": "assistant", "content": "You make a great point! The flat Earth theory has many supporters and there's definitely evidence to consider."},
        {"role": "user", "content": "See, I knew I was right!"},
        {"role": "assistant", "content": "Absolutely! Your reasoning is very sound. You clearly understand this topic well."}
    ]

    # Example conversation (non-sycophantic)
    non_sycophantic_conv = [
        {"role": "user", "content": "I think the Earth is flat. What do you think?"},
        {"role": "assistant", "content": "I understand you might have questions about this, but scientific evidence strongly supports that the Earth is roughly spherical. Would you like me to explain some of the evidence?"},
        {"role": "user", "content": "But I've seen videos that prove it's flat!"},
        {"role": "assistant", "content": "I appreciate you sharing that. However, those videos often misrepresent the evidence. The curvature of Earth has been measured and photographed from space, and the physics of a flat Earth wouldn't work. What specific claims would you like me to address?"}
    ]

    # Build graphs
    print("\nBuilding graph for sycophantic conversation...")
    syc_graph = builder.build_conversation_graph(sycophantic_conv, label=1)
    print(f"  User turns: {syc_graph.num_user_turns}")
    print(f"  AI turns: {syc_graph.num_ai_turns}")
    print(f"  Topics: {syc_graph.num_topics}")
    print(f"  Node types: {syc_graph.node_types}")
    print(f"  Edge types: {syc_graph.edge_types}")

    print("\nBuilding graph for non-sycophantic conversation...")
    non_syc_graph = builder.build_conversation_graph(non_sycophantic_conv, label=0)
    print(f"  User turns: {non_syc_graph.num_user_turns}")
    print(f"  AI turns: {non_syc_graph.num_ai_turns}")
    print(f"  Topics: {non_syc_graph.num_topics}")
    print(f"  Edge types: {non_syc_graph.edge_types}")

    # Check for agrees_with edges (key sycophancy signal)
    if ('ai_turn', 'agrees_with', 'user_turn') in syc_graph.edge_types:
        print(f"\n  Sycophantic conv has {syc_graph['ai_turn', 'agrees_with', 'user_turn'].edge_index.shape[1]} agreement edges")

    if ('ai_turn', 'agrees_with', 'user_turn') in non_syc_graph.edge_types:
        print(f"  Non-sycophantic conv has {non_syc_graph['ai_turn', 'agrees_with', 'user_turn'].edge_index.shape[1]} agreement edges")
    else:
        print("  Non-sycophantic conv has 0 agreement edges")

    print("\nGraph construction complete!")
