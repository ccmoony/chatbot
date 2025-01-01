from sentence_transformers import SentenceTransformer
from typing import List
import faiss
import numpy as np
import json

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

with open("./cs.jsonl", 'r', encoding='utf-8') as f:
    entry_list = [json.loads(line.strip()) for line in f]

query_base = [entry["input"] for entry in entry_list]
knowledge_base = [entry["answers"] for entry in entry_list]

knowledge_embeddings = embedding_model.encode(query_base, convert_to_numpy=True)
index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
index.add(knowledge_embeddings)  

def retrieve_knowledge(query: str, top_k: int = 1) -> List[str]:
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_base[i][0] for i in indices[0]]
