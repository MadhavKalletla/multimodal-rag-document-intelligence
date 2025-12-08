# day2/src/indexing.py

import os
import json
import pickle
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ======== CONFIGURATION ========
BATCH_SIZE = 200  # adjust depending on system speed
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")

# ======== LOAD ARTIFACTS ========
with open(os.path.join(DATA_DIR, "embeddings_meta.json"), "r") as f:
    meta = json.load(f)

n = meta.get("n_chunks", None)
dim = meta.get("embedding_dim", 384)
dtype = meta.get("dtype", "float32")

E = np.memmap(os.path.join(DATA_DIR, "embeddings.npy"),
              dtype=dtype, mode="r", shape=(n, dim))

with open(os.path.join(DATA_DIR, "embeddings_chunks.pkl"), "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks with dimension {dim}")

# ======== SETUP CHROMA ========
os.makedirs(DB_DIR, exist_ok=True)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="docs_day4")

# ======== BUILD METADATA ========
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [{"chunk_id": i} for i in range(len(chunks))]

# ======== BATCHED UPSERT ========
print("⚙️  Indexing embeddings into ChromaDB...")
for i in range(0, len(chunks), BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_embs = [E[j].tolist() for j in range(i, min(i+BATCH_SIZE, len(chunks)))]
    batch_docs = chunks[i:i+BATCH_SIZE]
    batch_meta = metadatas[i:i+BATCH_SIZE]

    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embs,
        documents=batch_docs,
        metadatas=batch_meta
    )

    print(f"✅ Indexed {i + len(batch_ids)}/{len(chunks)} chunks")

print("\n🎯 All embeddings indexed successfully into 'docs_day4'!")

# ======== QUERY TEST (OPTIONAL) ========
print("\n🔍 Running a quick similarity query test...")
model_name = meta.get("model_name", "all-MiniLM-L6-v2")
encoder = SentenceTransformer(model_name)

query = "Summarize what this document discusses."
q_emb = encoder.encode([query])

res = collection.query(query_embeddings=q_emb, n_results=3)

if res and "documents" in res and res["documents"][0]:
    print("\nTop Results:\n")
    for idx, doc in enumerate(res["documents"][0]):
        print(f"Result {idx+1}: {doc[:200]}...\n---")
else:
    print("⚠️ No results found. Check if your embeddings were created correctly.")
