import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from AuditableHybridGNN_POC import AuditableHybridGNN, POCGraphBuilder, NVEmbedV2EmbeddingModel
from tqdm import tqdm
import numpy as np

# --- 1. DATA LOADING FOR EVALUATION ---
def load_musique_eval_samples(path: str, limit: int = 200):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

# --- 2. EVALUATION SCRIPT ---
def evaluate(model_path, dataset_path, hidden_dim, device='cuda', k_list=[1, 3, 5, 10]):
    print("Initializing Model and loading fine-tuned weights...")
    
    # Load metadata for initialization
    temp_graph = torch.load("kg_storage/global_graph_bge_small.pt", map_location='cpu', weights_only=False)
    metadata = temp_graph.metadata()
    del temp_graph
    
    model = AuditableHybridGNN(metadata, hidden_dim).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    else:
        print(f"Warning: {model_path} not found. Running zero-shot evaluation.")
    
    model.eval()
    
    # Setup Tools
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    builder = POCGraphBuilder(embed_model, {"api_key": "", "endpoint": "", "api_version": "", "deployment_name": ""}, cache_path="musique_ie_cache.json")
    
    samples = load_musique_eval_samples(dataset_path)
    print(f"Evaluating on {len(samples)} samples from {dataset_path}...")
    
    recalls = {k: [] for k in k_list}
    mrr = []
    all_hops_found = {k: [] for k in k_list}
    
    with torch.no_grad():
        pbar = tqdm(samples, desc="Evaluating")
        for sample in pbar:
            question = sample['question']
            paragraphs = sample['paragraphs']
            
            target_indices = [idx for idx, p in enumerate(paragraphs) if p['is_supporting']]
            if not target_indices: continue
            
            # 1. Build Subgraph from the sample
            doc_texts = [p['paragraph_text'] for p in paragraphs]
            graph, _ = builder.build_graph(doc_texts)
            graph = graph.to(device)
            
            # 2. Get Query Embedding
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)
            
            # 3. Forward Pass
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)
            if scores.dim() == 0: scores = scores.unsqueeze(0)
            
            # 4. Rank indices
            # Applying sigmoid because we used BCEWithLogitsLoss
            probs = torch.sigmoid(scores)
            sorted_indices = torch.argsort(probs, descending=True).tolist()
            
            # 5. Calculate Metrics
            # MRR Calculation (for the first supporting paragraph found)
            rank = 999
            for i, idx in enumerate(sorted_indices):
                if idx in target_indices:
                    rank = i + 1
                    break
            mrr.append(1.0 / rank if rank != 999 else 0.0)
            
            for k in k_list:
                top_k_indices = sorted_indices[:k]
                
                # Recall@K: Did we find ANY of the supporting paragraphs?
                hits = [1 for idx in target_indices if idx in top_k_indices]
                recalls[k].append(len(hits) / len(target_indices))
                
                # All-Hops-Found@K: Did we find ALL supporting paragraphs in Top-K?
                all_hops_found[k].append(1.0 if len(hits) == len(target_indices) else 0.0)
            
            pbar.set_postfix({'MRR': f"{np.mean(mrr):.4f}", 'R@5': f"{np.mean(recalls[5]):.4f}"})
            
    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Dataset: {dataset_path}")
    print(f"MRR: {np.mean(mrr):.4f}")
    for k in k_list:
        print(f"Recall@{k}: {np.mean(recalls[k]):.2%}")
        print(f"Complete-Retrieval@{k}: {np.mean(all_hops_found[k]):.2%}")
    print("---------------------------------")

if __name__ == "__main__":
    MODEL_PATH = "finetuned_gnn_epoch_5.pth" # Update this to your best epoch
    DATASET_PATH = "dataset/musique_eval.json"
    HIDDEN_DIM = 384
    
    evaluate(MODEL_PATH, DATASET_PATH, HIDDEN_DIM, device='cuda')

