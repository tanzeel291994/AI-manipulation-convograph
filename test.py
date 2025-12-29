# import json

# cache_path = "musique_ie_cache.json"

# with open(cache_path, 'r') as f:
#     cache = json.load(f)

# # Count before
# total_before = len(cache)

# # Remove entries where both entities and triples are empty
# cleaned_cache = {
#     k: v for k, v in cache.items() 
#     if len(v.get("entities", [])) > 0 or len(v.get("triples", [])) > 0
# }

# # Count after
# total_after = len(cleaned_cache)

# with open(cache_path, 'w') as f:
#     json.dump(cleaned_cache, f, indent=2)

# print(f"Removed {total_before - total_after} empty entries. Cache now has {total_after} valid items.")
import torch
import transformers
from AuditableHybridGNN_POC import NVEmbedV2EmbeddingModel

def test_embedding_model():
    print(f"Transformers version: {transformers.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Target model
    model_name = "nvidia/NV-Embed-v2"
    
    try:
        print(f"\nInitializing {model_name}...")
        # Note: This will download several GBs if not already cached
        embed_model = NVEmbedV2EmbeddingModel(model_name)
        
        test_texts = [
            "The capital of France is Paris.",
            "Deep learning is a subset of machine learning.",
            "A graph neural network operates on graph-structured data."
        ]
        
        print(f"Encoding {len(test_texts)} samples...")
        embeddings = embed_model.batch_encode(test_texts, instruction="passage")
        
        print("\nSuccess!")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"First 5 values of first embedding: {embeddings[0][:5]}")
        
    except AttributeError as e:
        print(f"\nFAILED with AttributeError: {e}")
        if "get_usable_length" in str(e):
            print(">>> CONFIRMED: This is the DynamicCache incompatibility.")
            print(">>> FIX: Run 'poetry add transformers==4.43.3'")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    test_embedding_model()