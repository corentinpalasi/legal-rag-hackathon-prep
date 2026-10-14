import json
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

reader = PdfReader("data/code_penal.pdf")
#print(len(reader.pages))                      # how many pages

#print(reader.pages[0].extract_text()[:500])  # first 500 characters of page 1



full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text

#print(len(full_text))

chunks = []                                    # empty list of pieces
size = 800                                     # characters per piece
overlap = 150                                  # characters shared between 2 pieces

for start in range(0, len(full_text), size - overlap):
    piece = full_text[start : start + size]    # slice: from start, 800 characters
    chunks.append(piece)                       # add the piece to the list

print(len(chunks))
#print(chunks[500])


client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

vectors = []
for i in range(0, len(chunks), 50):                 # 50 chunks at a time
    batch = chunks[i : i + 50]
    #emb = client.embeddings.create(model="nomic-embed-text", input=batch) => worst version
    emb = client.embeddings.create(model="bge-m3", input=batch)
    vectors = vectors + [item.embedding for item in emb.data]
    print(i, "/", len(chunks))                      # progress

np.save("data/vectors.npy", np.array(vectors))      # the coordinates
with open("data/chunks.json", "w") as f:
    json.dump(chunks, f, ensure_ascii=False)
