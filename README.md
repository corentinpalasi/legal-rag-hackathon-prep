# Legal RAG on the French Code pénal (hackathon prep)

A small Retrieval-Augmented Generation (RAG) prototype built to prepare for the **Microsoft AI x Justice hackathon** (Paris, 15 October 2026).
Goal: answer questions about French criminal law **from the actual text of the Code pénal**, with the source article, instead of relying on the model's memory.

Everything runs **locally and for free** with [Ollama](https://ollama.com) and the `openai` Python library (the same library used for Azure OpenAI).

---

## Why RAG? What the first experiments showed

| Step | What I tried | What happened | Lesson |
|---|---|---|---|
| 1 | Asked `llama3.2` "Quel article du Code civil définit le contrat ?" | Answered *article 1098* (correct: **1101**) | LLMs hallucinate with confidence |
| 2 | Same question, temperature 1 vs 0 | T=1: 1128, 8, 1107. T=0: 1118 three times | Low temperature = consistent, **not** correct |
| 3 | Strict system prompt ("never invent, say *je ne sais pas*") | "Je ne sais pas" | A prompt makes the model honest, not knowledgeable |
| 4 | Pasted articles 311-1 and 311-3 into the prompt | Correct answer citing 311-3; "je ne sais pas" for an offence not in the text | **Giving the model the right text fixes hallucination → RAG** |

---

## How it works

```
INDEXING (once)                              QUERYING (each question)
Code pénal PDF (408 pages)                   Question
   → text (952,617 characters)                  → vector (same embedding model)
   → 1,466 chunks (800 chars, 150 overlap)      → cosine similarity with all chunk vectors
   → 1 vector per chunk (embedding model)       → top-3 chunks
   → saved: data/chunks.json, data/vectors.npy  → (next step) prompt + LLM → answer with source
```

## Project structure

| File | Role |
|---|---|
| `Files/01_first_call.py` | First call to a local LLM (`llama3.2`) |
| `Files/02_prompts.py` | `ask(question, system_prompt)` function; generic vs strict prompt; RAG by hand |
| `Files/03_ingest.py` | Builds the index: PDF → text → chunks → embeddings → saved to `data/` |
| `Files/04_embeddings.py` | Demo of embeddings and cosine similarity on 3 sentences |
| `Files/05_search.py` | Semantic search: question → top-3 chunks; `rank_of()` evaluation helper |

`data/` (PDF, chunks, vectors) is not versioned. Download the Code pénal PDF from [Légifrance](https://www.legifrance.gouv.fr) and save it as `data/code_penal.pdf`.

## How to run

```bash
# one-time setup
python -m venv .venv
source .venv/bin/activate
pip install openai pypdf numpy python-dotenv
ollama pull llama3.2
ollama pull bge-m3

# every new terminal
cd legal-rag-hackathon-prep
source .venv/bin/activate

# build the index (slow, once), then search (fast)
python Files/03_ingest.py
python Files/05_search.py
```

---

## Evaluating retrieval

The first search **failed**: for "Quelle est la peine encourue pour un vol ?", the chunk containing article 311-3 (chunk 731) was not in the top 3.

I checked that the article was really in the index, then measured **its rank among the 1,466 chunks** while changing one thing at a time:

| Experiment | Rank of the right chunk (lower is better) | Cosine score |
|---|---|---|
| Baseline: `nomic-embed-text` | 188 | 0.551 |
| A: `nomic-embed-text` + `search_query:` / `search_document:` prefixes | 70 | 0.642 |
| B: `bge-m3` (multilingual model) | **11** | 0.660 |

**Takeaway:** the embedding model matters a lot for French legal text. A multilingual model moved the right article from rank 188 to 11. It is still not in the top 3, so retrieval is the bottleneck to fix next.

Limits: this is measured on **one question**. A real evaluation needs a test set of ~10–20 questions with known articles (recall@3).

## Next steps

- **Chunk by article** (`"Article ..."`) instead of fixed 800-character windows, and remove PDF page footers.
- **Add metadata** (article number, chapter) to each chunk, to cite sources precisely.
- **Hybrid search** (vectors + keywords) and **reranking**: available natively in **Azure AI Search**.
- **Generation step:** pass the top-3 chunks to the LLM with a strict "answer only from the sources and cite them" prompt.
- **Test set** of questions with expected articles; track recall@3 for every change.
- Move to **Azure OpenAI** (only the client, key and model names change).

## Responsible use

Built for learning. Not legal advice. In justice, an AI tool should **assist** a professional (human in the loop), always show its sources, and say when it doesn't know.
