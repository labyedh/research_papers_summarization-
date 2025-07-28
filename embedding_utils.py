from transformers import AutoTokenizer, AutoModel
import torch
from config import EMBEDDING_MODEL

def generate_embedding(text, model_name=EMBEDDING_MODEL):
    """Generate sentence embeddings using BioBERT."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # Average all token embeddings for sentence representation
    embedding = outputs.last_hidden_state.mean(dim=1)  # Shape: (1, 768)
    
    return embedding.squeeze().numpy().tolist()

