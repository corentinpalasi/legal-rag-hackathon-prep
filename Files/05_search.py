import json
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# 1. Load the index built in 03 (instant, no recompute)
vectors = np.load("data/vectors.npy")
with open("data/chunks.json") as f:
    chunks = json.load(f)

#for i, c in enumerate(chunks):
    #if "Article 311-3" in c:
        #print(i, c[:200])


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(question, k=3):
    # 2. Turn the question into a vector (same model as the chunks!)
    emb = client.embeddings.create(model="bge-m3", input=[question])
    q = emb.data[0].embedding

    # 3. Score every chunk against the question
    scores = []
    for v in vectors:
        scores.append(cosine(q, v))

    # 4. Keep the positions of the k highest scores
    best = np.argsort(scores)[::-1][:k]
    return [(scores[i], chunks[i]) for i in best]


def rank_of(question, target):
    emb = client.embeddings.create(model="bge-m3", input=[question])
    q = emb.data[0].embedding
    scores = [cosine(q, v) for v in vectors]
    order = list(np.argsort(scores)[::-1])
    return order.index(target) + 1, round(scores[target], 3)

# 3. Use them
print(rank_of("Quelle est la peine encourue pour un vol ?", 731))


for score, text in search("Quelle est la peine encourue pour vol?"):
    print(round(score, 3), "|", text[:300])
    print("-----")
