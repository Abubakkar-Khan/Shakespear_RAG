import psycopg2
import requests
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

NEON_DB_URL = os.getenv("NEON_DB_URL")
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

def embed(text):
    """Generate embedding"""
    try:
        response = requests.post(
            OLLAMA_URL + "/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None

# Test embedding
print("Testing embedding for 'Who is Caius Marcius?'...")
question_emb = embed("Who is Caius Marcius?")
if question_emb:
    print(f"Embedding generated successfully, dimension: {len(question_emb)}")
else:
    print("Failed to generate embedding!")
    exit(1)

# Connect to database and test retrieval
conn = psycopg2.connect(NEON_DB_URL)
cur = conn.cursor()

# Check chunk count
cur.execute("SELECT COUNT(*) FROM rag_chunks")
count = cur.fetchone()[0]
print(f"\nDatabase has {count} chunks")

# Test retrieval
question_vector = str(question_emb)
cur.execute(
    """
    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
    FROM rag_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (question_vector, question_vector),
)

results = cur.fetchall()
print(f"\nTop 5 retrieval results:")
for i, (content, sim) in enumerate(results):
    print(f"\n{i+1}. Similarity: {sim:.4f}")
    print(f"   Content: {content[:200]}...")

cur.close()
conn.close()
