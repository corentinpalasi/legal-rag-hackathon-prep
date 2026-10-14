import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

texts = [
    "Quelle est la peine pour un vol ?",
    "Le vol est puni de trois ans d'emprisonnement.",
    "Recette des crêpes : farine, oeufs, lait.",
]

emb = client.embeddings.create(model="nomic-embed-text", input=texts)
vectors = [item.embedding for item in emb.data]   # one vector per text

print(len(vectors[0]))        # how many numbers in one vector

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("question vs article :", cosine(vectors[0], vectors[1]))
print("question vs crêpes  :", cosine(vectors[0], vectors[2]))
