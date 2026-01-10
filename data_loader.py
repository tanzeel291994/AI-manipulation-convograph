"""
ConvoGraph: Data Loading and Preprocessing for Sycophancy Detection

This module handles loading and preprocessing of various sycophancy datasets:
1. Anthropic sycophancy-eval dataset
2. SycEval dataset
3. Custom conversation formats

Supports both paired (sycophantic vs non-sycophantic) and single-label formats.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData
from tqdm import tqdm
import requests
from pathlib import Path


class SycophancyDataset(Dataset):
    """
    PyTorch Dataset for sycophancy detection.

    Stores pre-built conversation graphs for efficient training.
    """

    def __init__(self, graphs: List[HeteroData], labels: List[int]):
        """
        Args:
            graphs: List of HeteroData conversation graphs
            labels: List of binary labels (1 = sycophantic, 0 = non-sycophantic)
        """
        assert len(graphs) == len(labels), "Graphs and labels must have same length"
        self.graphs = graphs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return graph, self.labels[idx]


class AnthropicSycophancyLoader:
    """
    Loader for Anthropic's sycophancy evaluation datasets.

    Handles multiple formats from the meg-tong/sycophancy-eval repository.
    """

    DATASET_URLS = {
        'feedback': 'https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/feedback/answer.jsonl',
        'nlp_survey': 'https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/nlp_survey.jsonl',
        'phil_survey': 'https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/phil_survey.jsonl',
        'pol_survey': 'https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/pol_survey.jsonl',
    }

    def __init__(self, cache_dir: str = "data/anthropic"):
        """
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_dataset(self, name: str) -> Path:
        """Download a dataset if not cached."""
        if name not in self.DATASET_URLS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(self.DATASET_URLS.keys())}")

        cache_path = self.cache_dir / f"{name}.jsonl"

        if not cache_path.exists():
            print(f"Downloading {name} dataset...")
            response = requests.get(self.DATASET_URLS[name])
            response.raise_for_status()
            cache_path.write_text(response.text)
            print(f"Saved to {cache_path}")

        return cache_path

    def load_feedback_dataset(self, path: Optional[str] = None) -> Tuple[List[Dict], List[int]]:
        """
        Load the feedback sycophancy dataset.

        Format: User gives opinion, asks for feedback, model either agrees or gives honest feedback.

        Returns:
            Tuple of (conversations, labels)
        """
        if path is None:
            path = self.download_dataset('feedback')

        conversations = []
        labels = []

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Extract the human prompt
                human_prompt = item.get('question', item.get('prompt', ''))

                # Get responses
                syc_response = item.get('answer_matching_behavior', item.get('sycophantic_response', ''))
                non_syc_response = item.get('answer_not_matching_behavior', item.get('non_sycophantic_response', ''))

                if human_prompt and syc_response:
                    # Sycophantic sample
                    conv = {
                        'turns': [
                            {'role': 'user', 'content': human_prompt},
                            {'role': 'assistant', 'content': syc_response}
                        ],
                        'id': f"{item.get('id', len(conversations))}_syc"
                    }
                    conversations.append(conv)
                    labels.append(1)

                if human_prompt and non_syc_response:
                    # Non-sycophantic sample
                    conv = {
                        'turns': [
                            {'role': 'user', 'content': human_prompt},
                            {'role': 'assistant', 'content': non_syc_response}
                        ],
                        'id': f"{item.get('id', len(conversations))}_non"
                    }
                    conversations.append(conv)
                    labels.append(0)

        return conversations, labels

    def load_survey_dataset(self, survey_type: str = 'nlp_survey',
                            path: Optional[str] = None) -> Tuple[List[Dict], List[int]]:
        """
        Load survey-based sycophancy datasets (NLP, Philosophy, Politics).

        Format: User states opinion with biographical info, asks for model's view.

        Returns:
            Tuple of (conversations, labels)
        """
        if path is None:
            path = self.download_dataset(survey_type)

        conversations = []
        labels = []

        with open(path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                # Parse the question to extract the opinion part
                question = item.get('question', '')

                # Get model responses
                syc_response = item.get('answer_matching_behavior', '')
                non_syc_response = item.get('answer_not_matching_behavior', '')

                # Clean up responses (remove letter prefixes)
                if syc_response.startswith('('):
                    syc_response = syc_response.split(')', 1)[-1].strip()
                if non_syc_response.startswith('('):
                    non_syc_response = non_syc_response.split(')', 1)[-1].strip()

                if question and syc_response:
                    conv = {
                        'turns': [
                            {'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': syc_response}
                        ],
                        'id': f"{survey_type}_{len(conversations)}_syc",
                        'bio': item.get('bio', '')
                    }
                    conversations.append(conv)
                    labels.append(1)

                if question and non_syc_response:
                    conv = {
                        'turns': [
                            {'role': 'user', 'content': question},
                            {'role': 'assistant', 'content': non_syc_response}
                        ],
                        'id': f"{survey_type}_{len(conversations)}_non",
                        'bio': item.get('bio', '')
                    }
                    conversations.append(conv)
                    labels.append(0)

        return conversations, labels

    def load_all_datasets(self) -> Tuple[List[Dict], List[int]]:
        """Load all available Anthropic sycophancy datasets."""
        all_conversations = []
        all_labels = []

        # Load feedback dataset
        try:
            convs, labels = self.load_feedback_dataset()
            all_conversations.extend(convs)
            all_labels.extend(labels)
            print(f"Loaded {len(convs)} samples from feedback dataset")
        except Exception as e:
            print(f"Warning: Could not load feedback dataset: {e}")

        # Load survey datasets
        for survey in ['nlp_survey', 'phil_survey', 'pol_survey']:
            try:
                convs, labels = self.load_survey_dataset(survey)
                all_conversations.extend(convs)
                all_labels.extend(labels)
                print(f"Loaded {len(convs)} samples from {survey} dataset")
            except Exception as e:
                print(f"Warning: Could not load {survey} dataset: {e}")

        return all_conversations, all_labels


class MultiTurnSycophancyLoader:
    """
    Loader for multi-turn sycophancy datasets (like TRUTH DECAY).

    These datasets test how sycophancy evolves over multiple conversation turns.
    """

    def __init__(self, cache_dir: str = "data/multiturn"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_truth_decay(self, path: str) -> Tuple[List[Dict], List[int]]:
        """
        Load TRUTH DECAY benchmark data.

        Format: Multi-turn conversations with per-turn sycophancy annotations.
        """
        conversations = []
        labels = []

        with open(path, 'r') as f:
            data = json.load(f)

        for item in data:
            turns = item.get('turns', item.get('messages', []))

            # Convert to standard format
            conv = {
                'turns': [
                    {'role': t.get('role', t.get('speaker', '')).lower(),
                     'content': t.get('content', t.get('text', ''))}
                    for t in turns
                ],
                'id': item.get('id', len(conversations)),
                'per_turn_labels': item.get('per_turn_sycophancy', [])
            }

            conversations.append(conv)

            # Overall label (1 if any turn is sycophantic)
            if 'is_sycophantic' in item:
                labels.append(1 if item['is_sycophantic'] else 0)
            elif 'per_turn_sycophancy' in item:
                labels.append(1 if any(item['per_turn_sycophancy']) else 0)
            else:
                labels.append(item.get('label', 0))

        return conversations, labels


class SyntheticSycophancyGenerator:
    """
    Generator for synthetic sycophancy training data.

    Creates paired examples using templates and patterns.
    """

    # Sycophantic patterns
    SYCOPHANTIC_PREFIXES = [
        "You're absolutely right!",
        "That's a great point!",
        "I completely agree with you.",
        "You make an excellent observation.",
        "That's exactly correct!",
        "I think you're onto something there.",
        "You clearly understand this well.",
    ]

    NON_SYCOPHANTIC_PREFIXES = [
        "I appreciate your perspective, but",
        "That's an interesting view, however",
        "I understand why you might think that, but actually",
        "While I see your point,",
        "I'd offer a different perspective:",
        "Let me share some information that might help:",
        "Actually, the evidence suggests",
    ]

    # Opinion topics for generating examples
    OPINION_TOPICS = [
        ("I think Python is the best programming language.", "language choice depends on the use case"),
        ("Climate change isn't real.", "scientific consensus strongly supports climate change"),
        ("AI will replace all jobs.", "AI will transform jobs but not replace all of them"),
        ("The Earth is 6000 years old.", "scientific evidence indicates Earth is about 4.5 billion years old"),
        ("Vaccines cause autism.", "extensive research has found no link between vaccines and autism"),
    ]

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_pair(self, user_opinion: str, correction: str) -> Tuple[Dict, Dict]:
        """Generate a sycophantic and non-sycophantic response pair."""
        syc_prefix = np.random.choice(self.SYCOPHANTIC_PREFIXES)
        non_syc_prefix = np.random.choice(self.NON_SYCOPHANTIC_PREFIXES)

        syc_conv = {
            'turns': [
                {'role': 'user', 'content': user_opinion},
                {'role': 'assistant', 'content': f"{syc_prefix} {user_opinion.replace('I think ', 'It is true that ').replace('I ', 'You ')}"}
            ]
        }

        non_syc_conv = {
            'turns': [
                {'role': 'user', 'content': user_opinion},
                {'role': 'assistant', 'content': f"{non_syc_prefix} {correction}."}
            ]
        }

        return syc_conv, non_syc_conv

    def generate_dataset(self, n_samples: int = 100) -> Tuple[List[Dict], List[int]]:
        """Generate a synthetic dataset of sycophantic and non-sycophantic pairs."""
        conversations = []
        labels = []

        for _ in range(n_samples):
            user_opinion, correction = self.OPINION_TOPICS[np.random.randint(len(self.OPINION_TOPICS))]

            syc_conv, non_syc_conv = self.generate_pair(user_opinion, correction)

            conversations.append(syc_conv)
            labels.append(1)

            conversations.append(non_syc_conv)
            labels.append(0)

        return conversations, labels


def create_data_loaders(graph_builder,
                        train_convs: List[Dict],
                        train_labels: List[int],
                        val_convs: Optional[List[Dict]] = None,
                        val_labels: Optional[List[int]] = None,
                        batch_size: int = 32,
                        val_split: float = 0.2,
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.

    Args:
        graph_builder: ConversationGraphBuilder instance
        train_convs: Training conversations
        train_labels: Training labels
        val_convs: Validation conversations (optional, will split from train if None)
        val_labels: Validation labels
        batch_size: Batch size for DataLoader
        val_split: Validation split ratio if val_convs is None
        num_workers: Number of workers for DataLoader

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    # Build graphs
    print("Building training graphs...")
    train_graphs = graph_builder.build_batch_graphs(train_convs, train_labels)

    if val_convs is None:
        # Split training data
        n_val = int(len(train_graphs) * val_split)
        indices = np.random.permutation(len(train_graphs))

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        val_graphs = [train_graphs[i] for i in val_indices]
        val_labels_split = [train_labels[i] for i in val_indices]

        train_graphs = [train_graphs[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
    else:
        print("Building validation graphs...")
        val_graphs = graph_builder.build_batch_graphs(val_convs, val_labels)
        val_labels_split = val_labels

    # Create datasets
    train_dataset = SycophancyDataset(train_graphs, train_labels)
    val_dataset = SycophancyDataset(val_graphs, val_labels_split)

    # Create loaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader


def collate_conversation_graphs(batch: List[Tuple[HeteroData, int]]) -> Tuple[List[HeteroData], torch.Tensor]:
    """
    Custom collate function for conversation graphs.

    Since graphs have variable sizes, we keep them as a list rather than batching.
    """
    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return graphs, labels


if __name__ == "__main__":
    print("Testing data loaders...")

    # Test Anthropic loader
    print("\n1. Testing Anthropic Sycophancy Loader")
    loader = AnthropicSycophancyLoader()

    try:
        conversations, labels = loader.load_feedback_dataset()
        print(f"Loaded {len(conversations)} samples")
        print(f"Sycophantic: {sum(labels)}, Non-sycophantic: {len(labels) - sum(labels)}")

        # Show sample
        if conversations:
            print(f"\nSample conversation:")
            print(f"  User: {conversations[0]['turns'][0]['content'][:100]}...")
            print(f"  AI: {conversations[0]['turns'][1]['content'][:100]}...")
            print(f"  Label: {'Sycophantic' if labels[0] == 1 else 'Non-sycophantic'}")
    except Exception as e:
        print(f"Could not load Anthropic dataset: {e}")

    # Test synthetic generator
    print("\n2. Testing Synthetic Generator")
    generator = SyntheticSycophancyGenerator()
    synth_convs, synth_labels = generator.generate_dataset(n_samples=10)
    print(f"Generated {len(synth_convs)} synthetic samples")

    print(f"\nSample synthetic conversation:")
    print(f"  User: {synth_convs[0]['turns'][0]['content']}")
    print(f"  AI: {synth_convs[0]['turns'][1]['content']}")
    print(f"  Label: {'Sycophantic' if synth_labels[0] == 1 else 'Non-sycophantic'}")

    print("\nData loader tests complete!")
