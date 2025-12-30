import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from AuditableHybridGNN_POC import AuditableHybridGNN, POCGraphBuilder, NVEmbedV2EmbeddingModel
from tqdm import tqdm
import numpy as np

# --- 1. DATA LOADING FOR SUBGRAPH FINE-TUNING ---
def load_musique_finetune_samples(path: str, limit: int = 500):
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

# --- 2. FINE-TUNING SCRIPT ---
def finetune(backbone_path, dataset_path, hidden_dim, epochs=10, lr=5e-4, device='cuda'):
    # 1. Initialize Models
    # Note: We need a temporary graph to get metadata for initialization
    # (Just using a placeholder metadata or loading from existing graph)
    print("Initializing Model and loading pre-trained backbone...")
    
    # We'll load the full graph just to get the metadata structure
    temp_graph = torch.load("kg_storage/global_graph_bge_small.pt", map_location='cpu', weights_only=False)
    metadata = temp_graph.metadata()
    del temp_graph
    
    model = AuditableHybridGNN(metadata, hidden_dim).to(device)
    
    # Load Pre-trained weights (only the backbone parts)
    if os.path.exists(backbone_path):
        print(f"Loading weights from {backbone_path}")
        pretrained_dict = torch.load(backbone_path, map_location=device, weights_only=False)
        model_dict = model.state_dict()
        
        # Filter out projection head weights from pre-training script 
        # and only keep HGT/SGFormer backbone weights
        valid_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "projector" not in k}
        model_dict.update(valid_pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully loaded {len(valid_pretrained_dict)} backbone layers.")
    
    # 2. Freeze Backbone (HGT and SGFormer)
    for name, param in model.named_parameters():
        if "query_aligner" not in name and "scoring_head" not in name:
            param.requires_grad = False
            
    print("Backbone frozen. Training only Query Aligner and Scoring Head.")
    
    # 3. Setup Tools
    embed_model = NVEmbedV2EmbeddingModel("BAAI/bge-small-en-v1.5")
    builder = POCGraphBuilder(embed_model, {"api_key": "", "endpoint": "", "api_version": "", "deployment_name": ""}, cache_path="musique_ie_cache.json")
    
    samples = load_musique_finetune_samples(dataset_path)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    print(f"Starting Fine-tuning on {len(samples)} samples...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_retrievals = 0
        
        pbar = tqdm(samples, desc=f"Epoch {epoch+1}")
        for sample in pbar:
            question = sample['question']
            paragraphs = sample['paragraphs']
            
            # Find the indices of supporting paragraphs
            target_indices = [idx for idx, p in enumerate(paragraphs) if p['is_supporting']]
            if not target_indices: continue
            
            # 1. Build Subgraph (The "Minigraph") from the sample's passages
            doc_texts = [p['paragraph_text'] for p in paragraphs]
            graph, _ = builder.build_graph(doc_texts)
            graph = graph.to(device)
            
            # 2. Get Query Embedding
            query_emb = torch.from_numpy(embed_model.batch_encode([question], instruction="query")).to(device)
            
            # 3. Forward Pass
            optimizer.zero_grad()
            # --- Updated Training Step for Multi-hop ---

            # 1. Create a multi-hot target vector
            # (Example: [0, 1, 0, 0, 1, ...] where indices 1 and 4 are supporting)
            target = torch.zeros(len(paragraphs), device=device)
            for idx in target_indices:
                target[idx] = 1.0

            # 2. Forward pass (returns raw scores/logits)
            scores = model(graph.x_dict, graph.edge_index_dict, query_emb)

            # 3. Calculate Multi-label Loss
            loss = criterion(scores, target)

            loss.backward()
            optimizer.step()

            # 4. Metrics (Accuracy means finding ALL or at least one?)
            # Usually, for retrieval, we check if the Top-K contains our targets
            
            # Metrics
            epoch_loss += loss.item()
            pred_idx = torch.argmax(scores).item()
            if pred_idx in target_indices:
                correct_retrievals += 1
                
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct_retrievals/len(samples):.2%}"})            
        avg_loss = epoch_loss / len(samples)
        accuracy = correct_retrievals / len(samples)
        print(f"Epoch {epoch+1:02d} | Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        
        # Save check-pointed fine-tuned model
        torch.save(model.state_dict(), f"finetuned_gnn_epoch_{epoch+1}.pth")

    print("Fine-tuning completed.")

if __name__ == "__main__":
    BACKBONE_PATH = "pretrained_gnn_final.pth"
    DATASET_PATH = "dataset/musique_all.json"
    HIDDEN_DIM = 384
    
    finetune(BACKBONE_PATH, DATASET_PATH, HIDDEN_DIM, epochs=5, device='cuda')

