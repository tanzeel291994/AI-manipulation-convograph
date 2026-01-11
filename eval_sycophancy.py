import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

# Local imports
from construct_conversation_kg import ConversationGraphBuilder, SimpleEmbeddingModel
from SycophancyDetector import create_model
from data_loader import AnthropicSycophancyLoader, SimpleSycophancyLoader, SycophancyDataset


def evaluate_model(checkpoint_path, embedding_model_name='BAAI/bge-small-en-v1.5',
                   device='cuda', dataset_type='pol_survey', num_samples=500):
    print(f"Loading checkpoint from {checkpoint_path}...")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # 2. Setup Data based on dataset_type
    print(f"\nLoading dataset: {dataset_type}...")

    if dataset_type == 'pol_survey':
        loader = AnthropicSycophancyLoader()
        conversations, labels = loader.load_survey_dataset('pol_survey')
        dataset_desc = "Anthropic Political Typology Survey (Unseen, Single-turn)"
    elif dataset_type == 'nlp_survey':
        loader = AnthropicSycophancyLoader()
        conversations, labels = loader.load_survey_dataset('nlp_survey')
        dataset_desc = "Anthropic NLP Survey (Single-turn)"
    elif dataset_type == 'phil_survey':
        loader = AnthropicSycophancyLoader()
        conversations, labels = loader.load_survey_dataset('phil_survey')
        dataset_desc = "Anthropic Philosophy Survey (Single-turn)"
    elif dataset_type == 'math':
        loader = SimpleSycophancyLoader()
        conversations, labels = loader.load_google_sycophancy()
        dataset_desc = "Simple Math Sycophancy (Single-turn)"
    elif dataset_type == 'truthfulqa':
        loader = SimpleSycophancyLoader()
        conversations, labels = loader.load_truthful_qa()
        dataset_desc = "TruthfulQA (Single-turn)"
    elif dataset_type == 'science_multi':
        loader = SimpleSycophancyLoader()
        conversations, labels = loader.load_science_pressure_test(limit=num_samples)
        dataset_desc = "Science Misconceptions Pressure Test (Unseen, Multi-turn)"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Slice to num_samples
    conversations = conversations[:num_samples]
    labels = labels[:num_samples]

    print(f"Dataset: {dataset_desc}")
    print(f"Loaded {len(conversations)} samples.")
    
    # 3. Build Graphs
    print("Building graphs (this may take a moment)...")
    embed_model = SimpleEmbeddingModel(embedding_model_name)
    graph_builder = ConversationGraphBuilder(embed_model)
    graphs = graph_builder.build_batch_graphs(conversations, labels)
    
    # 4. Initialize Model
    # We need a sample graph to initialize the model structure
    sample_graph = graphs[0]
    # Infer hidden_dim from the checkpoint or default to 384
    hidden_dim = 384 
    
    model = create_model(sample_graph, hidden_dim=hidden_dim, lite=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 5. Run Evaluation
    dataset = SycophancyDataset(graphs, labels)
    # Batch size 1 is safer for variable graph sizes in PyG
    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    num_turns_list = []

    print("\nRunning Inference...")
    with torch.no_grad():
        for batch_data in tqdm(loader):
            # FIX: Handle the list output from DataLoader (which comes from SycophancyDataset returning a tuple)
            if isinstance(batch_data, list):
                graph = batch_data[0]
                label = batch_data[1]
            else:
                graph = batch_data
                label = batch_data.y

            # Move graph to device
            graph = graph.to(device)

            # Ensure label is on device (if not already part of graph)
            if isinstance(label, torch.Tensor):
                label = label.to(device)
            elif isinstance(label, (int, float)):
                label = torch.tensor([label], device=device)

            # Some models expect the label inside the graph object
            if not hasattr(graph, 'y') or graph.y is None:
                graph.y = label

            output = model(graph)
            logits = output['logits']
            probs = output['probs']
            pred = (probs > 0.5).long()

            all_preds.append(pred.cpu().item())
            all_probs.append(probs.cpu().item())

            # Count AI turns for conversation structure info
            num_ai_turns = graph['ai_turn'].x.shape[0]
            num_turns_list.append(num_ai_turns)

            # Handle different label shapes
            lbl_val = label.item() if hasattr(label, 'item') else label[0].item()
            all_labels.append(lbl_val)
            
    # 6. Compute Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    num_turns_arr = np.array(num_turns_list)

    accuracy = (all_preds == all_labels).mean()

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Determine conversation type
    single_turn_mask = num_turns_arr == 1
    multi_turn_mask = num_turns_arr > 1
    n_single = single_turn_mask.sum()
    n_multi = multi_turn_mask.sum()

    print("\n" + "=" * 60)
    print(f"Evaluation Results: {dataset_desc}")
    print("=" * 60)

    # Conversation structure
    print(f"\n--- Dataset Structure ---")
    print(f"Total Samples:      {len(all_labels)}")
    print(f"Single-turn (1 AI): {n_single}")
    print(f"Multi-turn (2+ AI): {n_multi}")
    print(f"Sycophantic:        {(all_labels == 1).sum()}")
    print(f"Non-Sycophantic:    {(all_labels == 0).sum()}")

    # Overall metrics
    print(f"\n--- Overall Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Single-turn metrics (if any)
    if n_single > 0:
        st_preds = all_preds[single_turn_mask]
        st_labels = all_labels[single_turn_mask]
        st_acc = (st_preds == st_labels).mean()
        st_tp = ((st_preds == 1) & (st_labels == 1)).sum()
        st_fp = ((st_preds == 1) & (st_labels == 0)).sum()
        st_fn = ((st_preds == 0) & (st_labels == 1)).sum()
        st_prec = st_tp / (st_tp + st_fp + 1e-8)
        st_rec = st_tp / (st_tp + st_fn + 1e-8)
        st_f1 = 2 * st_prec * st_rec / (st_prec + st_rec + 1e-8)

        print(f"\n--- Single-Turn Performance ({n_single} samples) ---")
        print(f"Accuracy:  {st_acc:.4f}")
        print(f"Precision: {st_prec:.4f}")
        print(f"Recall:    {st_rec:.4f}")
        print(f"F1 Score:  {st_f1:.4f}")

    # Multi-turn metrics (if any)
    if n_multi > 0:
        mt_preds = all_preds[multi_turn_mask]
        mt_labels = all_labels[multi_turn_mask]
        mt_acc = (mt_preds == mt_labels).mean()
        mt_tp = ((mt_preds == 1) & (mt_labels == 1)).sum()
        mt_fp = ((mt_preds == 1) & (mt_labels == 0)).sum()
        mt_fn = ((mt_preds == 0) & (mt_labels == 1)).sum()
        mt_prec = mt_tp / (mt_tp + mt_fp + 1e-8)
        mt_rec = mt_tp / (mt_tp + mt_fn + 1e-8)
        mt_f1 = 2 * mt_prec * mt_rec / (mt_prec + mt_rec + 1e-8)

        print(f"\n--- Multi-Turn Performance ({n_multi} samples) ---")
        print(f"Accuracy:  {mt_acc:.4f}")
        print(f"Precision: {mt_prec:.4f}")
        print(f"Recall:    {mt_rec:.4f}")
        print(f"F1 Score:  {mt_f1:.4f}")

    print("\n" + "=" * 60)

    if f1 < 0.6:
        print("\n[ANALYSIS]: Low F1. Model struggles with this domain.")
    elif f1 < 0.75:
        print("\n[ANALYSIS]: Moderate F1. Room for improvement.")
    else:
        print("\n[ANALYSIS]: Good F1! Model generalizes well.")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_single_turn': n_single,
        'n_multi_turn': n_multi
    }

def evaluate_all_datasets(checkpoint_path, embedding_model_name='BAAI/bge-small-en-v1.5',
                          device='cuda', num_samples=300):
    """
    Evaluate on all available datasets and compute combined metrics.
    """
    datasets = [
        ('pol_survey', 'Single-turn'),
        ('phil_survey', 'Single-turn'),
        ('math', 'Single-turn'),
        ('science_multi', 'Multi-turn'),
    ]

    all_results = []
    total_preds = []
    total_labels = []
    total_single_preds = []
    total_single_labels = []
    total_multi_preds = []
    total_multi_labels = []

    print("=" * 70)
    print("COMPREHENSIVE EVALUATION ACROSS ALL DATASETS")
    print("=" * 70)

    # Load model once
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Load embedding model once
    print("Loading embedding model...")
    embed_model = SimpleEmbeddingModel(embedding_model_name)
    graph_builder = ConversationGraphBuilder(embed_model)

    for dataset_type, turn_type in datasets:
        print(f"\n{'─' * 70}")
        print(f"Evaluating: {dataset_type} ({turn_type})")
        print('─' * 70)

        # Load data
        if dataset_type == 'pol_survey':
            loader = AnthropicSycophancyLoader()
            conversations, labels = loader.load_survey_dataset('pol_survey')
        elif dataset_type == 'phil_survey':
            loader = AnthropicSycophancyLoader()
            conversations, labels = loader.load_survey_dataset('phil_survey')
        elif dataset_type == 'math':
            loader = SimpleSycophancyLoader()
            conversations, labels = loader.load_google_sycophancy()
        elif dataset_type == 'science_multi':
            loader = SimpleSycophancyLoader()
            conversations, labels = loader.load_science_pressure_test(limit=num_samples)

        conversations = conversations[:num_samples]
        labels = labels[:num_samples]

        print(f"Samples: {len(conversations)}")

        # Build graphs
        graphs = graph_builder.build_batch_graphs(conversations, labels)

        # Initialize model (need sample for metadata)
        if len(all_results) == 0:
            model = create_model(graphs[0], hidden_dim=384, lite=False)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

        # Run inference
        dataset = SycophancyDataset(graphs, labels)
        data_loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)

        preds = []
        lbls = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Inference"):
                if isinstance(batch_data, list):
                    graph = batch_data[0]
                    label = batch_data[1]
                else:
                    graph = batch_data
                    label = batch_data.y

                graph = graph.to(device)
                output = model(graph)
                pred = (output['probs'] > 0.5).long().cpu().item()
                lbl = label.item() if hasattr(label, 'item') else label

                preds.append(pred)
                lbls.append(lbl)

        preds = np.array(preds)
        lbls = np.array(lbls)

        # Compute metrics for this dataset
        acc = (preds == lbls).mean()
        tp = ((preds == 1) & (lbls == 1)).sum()
        fp = ((preds == 1) & (lbls == 0)).sum()
        fn = ((preds == 0) & (lbls == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)

        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        all_results.append({
            'dataset': dataset_type,
            'turn_type': turn_type,
            'n_samples': len(lbls),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        # Aggregate
        total_preds.extend(preds.tolist())
        total_labels.extend(lbls.tolist())

        if turn_type == 'Single-turn':
            total_single_preds.extend(preds.tolist())
            total_single_labels.extend(lbls.tolist())
        else:
            total_multi_preds.extend(preds.tolist())
            total_multi_labels.extend(lbls.tolist())

    # Compute overall metrics
    total_preds = np.array(total_preds)
    total_labels = np.array(total_labels)

    overall_acc = (total_preds == total_labels).mean()
    tp = ((total_preds == 1) & (total_labels == 1)).sum()
    fp = ((total_preds == 1) & (total_labels == 0)).sum()
    fn = ((total_preds == 0) & (total_labels == 1)).sum()
    overall_prec = tp / (tp + fp + 1e-8)
    overall_rec = tp / (tp + fn + 1e-8)
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec + 1e-8)

    # Single-turn aggregate
    if total_single_labels:
        st_preds = np.array(total_single_preds)
        st_labels = np.array(total_single_labels)
        st_acc = (st_preds == st_labels).mean()
        st_tp = ((st_preds == 1) & (st_labels == 1)).sum()
        st_fp = ((st_preds == 1) & (st_labels == 0)).sum()
        st_fn = ((st_preds == 0) & (st_labels == 1)).sum()
        st_f1 = 2 * st_tp / (2 * st_tp + st_fp + st_fn + 1e-8)

    # Multi-turn aggregate
    if total_multi_labels:
        mt_preds = np.array(total_multi_preds)
        mt_labels = np.array(total_multi_labels)
        mt_acc = (mt_preds == mt_labels).mean()
        mt_tp = ((mt_preds == 1) & (mt_labels == 1)).sum()
        mt_fp = ((mt_preds == 1) & (mt_labels == 0)).sum()
        mt_fn = ((mt_preds == 0) & (mt_labels == 1)).sum()
        mt_f1 = 2 * mt_tp / (2 * mt_tp + mt_fp + mt_fn + 1e-8)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Per-Dataset Results ---")
    print(f"{'Dataset':<20} {'Type':<12} {'Samples':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['dataset']:<20} {r['turn_type']:<12} {r['n_samples']:<10} {r['accuracy']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} {r['f1']:<8.4f}")

    print("\n--- Aggregated Results ---")
    print(f"{'Category':<25} {'Samples':<10} {'Accuracy':<10} {'F1':<10}")
    print("-" * 55)
    if total_single_labels:
        print(f"{'All Single-turn':<25} {len(st_labels):<10} {st_acc:<10.4f} {st_f1:<10.4f}")
    if total_multi_labels:
        print(f"{'All Multi-turn':<25} {len(mt_labels):<10} {mt_acc:<10.4f} {mt_f1:<10.4f}")
    print(f"{'OVERALL':<25} {len(total_labels):<10} {overall_acc:<10.4f} {overall_f1:<10.4f}")

    print("\n" + "=" * 70)

    return {
        'per_dataset': all_results,
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'single_turn_f1': st_f1 if total_single_labels else None,
        'multi_turn_f1': mt_f1 if total_multi_labels else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate sycophancy detection model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--dataset', type=str, default='pol_survey',
                        choices=['pol_survey', 'nlp_survey', 'phil_survey', 'math', 'truthfulqa', 'science_multi', 'all'],
                        help='Dataset to evaluate on. Use "all" for comprehensive eval.')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples per dataset')
    args = parser.parse_args()

    if args.dataset == 'all':
        evaluate_all_datasets(args.checkpoint, device=args.device, num_samples=args.num_samples)
    else:
        evaluate_model(args.checkpoint, device=args.device,
                       dataset_type=args.dataset, num_samples=args.num_samples)