"""
Load all remaining chunks into Neon Postgres with pgvector
"""
import requests
import time
from pathlib import Path
from psycopg2.extras import execute_values
from app import get_db_connection, chunk_text, INPUT_FILE, OLLAMA_URL, EMBED_MODEL

def main():
    print("Reading input text file...")
    raw = Path(INPUT_FILE).read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(raw)
    total_chunks = len(chunks)
    print(f"Total chunks in source text: {total_chunks}")

    # Check which chunks are already in DB
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_index FROM rag_chunks")
    existing_indices = set(r[0] for r in cur.fetchall())
    cur.close()
    conn.close()

    print(f"Currently in DB: {len(existing_indices)} chunks")

    # Determine missing chunks
    missing_items = [(idx, chunks[idx]) for idx in range(total_chunks) if idx not in existing_indices]
    print(f"Chunks to embed and insert: {len(missing_items)}")

    if not missing_items:
        print("All chunks are already loaded in database!")
        return

    batch_size = 25
    total_stored = 0
    t_start = time.time()

    for i in range(0, len(missing_items), batch_size):
        batch = missing_items[i : i + batch_size]
        indices = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        # Call Ollama batch embed endpoint
        retries = 3
        vectors = None
        for attempt in range(retries):
            try:
                res = requests.post(
                    f"{OLLAMA_URL}/api/embed",
                    json={"model": EMBED_MODEL, "input": texts},
                    timeout=120
                )
                if res.status_code == 200:
                    data = res.json()
                    vectors = data.get("embeddings", [])
                    if len(vectors) == len(texts):
                        break
                print(f"Embed attempt {attempt + 1} status {res.status_code}, retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"Embed attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2)

        if not vectors:
            print(f"Failed to embed batch at index {indices[0]}; skipping.")
            continue

        # Insert batch into PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()
        rows = [
            (idx, "shakespeare", ch, str(vec))
            for idx, ch, vec in zip(indices, texts, vectors)
        ]
        execute_values(
            cur,
            "INSERT INTO rag_chunks (chunk_index, source, content, embedding) VALUES %s",
            rows,
            template="(%s, %s, %s, %s::vector)"
        )
        cur.close()
        conn.close()

        total_stored += len(batch)
        elapsed = time.time() - t_start
        rate = total_stored / elapsed if elapsed > 0 else 0
        remaining = (len(missing_items) - total_stored) / rate if rate > 0 else 0
        print(f"Stored {total_stored}/{len(missing_items)} missing chunks ({len(existing_indices) + total_stored}/{total_chunks} total) - Est. remaining: {remaining:.0f}s")

    # Final count verification
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rag_chunks")
    final_count = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"\nAll chunks loaded successfully!")
    print(f"Total chunks in database: {final_count}/{total_chunks}")

if __name__ == "__main__":
    main()
