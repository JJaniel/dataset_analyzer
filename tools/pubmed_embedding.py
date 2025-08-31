from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class PubMedEmbedding:
    def __init__(self):
        model_path = Path(os.getenv("EMBEDDING_MODEL_PATH"))
        
        # Add local_files_only=True to all from_pretrained calls
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, config=config, local_files_only=True)

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.tolist()
