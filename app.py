"""
Shakespeare RAG Web App
A simple web application for querying Shakespeare's works using local AI with RAG (Retrieval Augmented Generation).
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import psycopg2
from psycopg2.extras import execute_values
import requests
from pathlib import Path
import numpy as np
import hashlib
import re
import os
import sys
import subprocess
import time
from dotenv import load_dotenv

# Ensure console supports UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Load environment variables
load_dotenv()

# ---- CONFIGURATION ----
NEON_DB_URL = os.getenv("NEON_DB_URL")
INPUT_FILE = os.getenv("INPUT_FILE", "./input.txt")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # 'groq' or 'ollama'

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("CHAT_MODEL", "qwen2:1.5b")

# Embedding Configuration (always uses Ollama)
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))  # Smaller chunks for better precision
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # Better overlap
TOP_K = int(os.getenv("TOP_K", "6"))  # More context for better answers
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "500")) if os.getenv("MAX_CHUNKS") else None  # More chunks

app = Flask(__name__)
CORS(app)

# ---- DATABASE SETUP ----
def get_db_connection():
    """Create and return a database connection with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                NEON_DB_URL,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
                connect_timeout=10
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Database connection attempt {attempt + 1} failed, retrying...")
                import time
                time.sleep(2)
            else:
                raise e

def setup_database():
    """Initialize database tables and extensions"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create chunks table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id          BIGSERIAL PRIMARY KEY,
            chunk_index INTEGER,
            source      TEXT,
            content     TEXT,
            embedding   vector({EMBED_DIM})
        )
    """)
    
    cur.close()
    conn.close()
    print("Database initialized")

# ---- TEXT PROCESSING ----
def extract_speaker_and_scene(text):
    """Extract speaker names and scene context from dialogue"""
    lines = text.split('\n')
    speakers = []
    scene_info = []
    
    for line in lines[:5]:  # Check first few lines
        line = line.strip()
        # Character names are usually in caps or followed by colon
        if line and (line.isupper() or ':' in line):
            speaker = line.split(':')[0].strip()
            if speaker and len(speaker) < 50:
                speakers.append(speaker)
        # Scene descriptions often contain brackets or stage directions
        if any(word in line.lower() for word in ['scene', 'act', 'enter', 'exit']):
            scene_info.append(line)
    
    return speakers[:3], scene_info[:2]  # Return top speakers and scene info

def enhance_chunk_with_context(chunk, chunk_index):
    """Add contextual summary to chunk for better retrieval"""
    speakers, scenes = extract_speaker_and_scene(chunk)
    
    # Build context prefix
    context_parts = []
    if scenes:
        context_parts.append(f"[Scene: {' | '.join(scenes)}]")
    if speakers:
        context_parts.append(f"[Characters: {', '.join(speakers)}]")
    
    # Add context prefix if we found any
    if context_parts:
        enhanced = '\n'.join(context_parts) + '\n\n' + chunk
        return enhanced
    
    return chunk

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks with better boundary detection"""
    chunks = []
    
    # Split by double newline first (scene/paragraph breaks)
    sections = text.split('\n\n')
    
    current_chunk = ""
    chunk_index = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # If adding this section exceeds chunk size, save current and start new
        if len(current_chunk) + len(section) > size and current_chunk:
            # Enhance chunk with context before adding
            enhanced = enhance_chunk_with_context(current_chunk.strip(), chunk_index)
            chunks.append(enhanced)
            chunk_index += 1
            
            # Keep overlap from end of previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + section
        else:
            current_chunk += ("\n\n" if current_chunk else "") + section
    
    # Add final chunk
    if current_chunk.strip():
        enhanced = enhance_chunk_with_context(current_chunk.strip(), chunk_index)
        chunks.append(enhanced)
    
    return chunks

# ---- EMBEDDINGS ----
def fallback_embed(text, dim=EMBED_DIM):
    """Offline fallback for embeddings"""
    vec = np.zeros(dim, dtype=np.float32)
    for word in re.findall(r"[a-z]+", text.lower()):
        bucket = int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % dim
        vec[bucket] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def embed(text, use_ollama=True):
    """Generate embedding vector for text"""
    if not use_ollama:
        return fallback_embed(text)
    
    try:
        response = requests.post(
            OLLAMA_URL + "/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"Ollama embedding failed, using fallback: {e}")
        return fallback_embed(text)

# ---- RAG FUNCTIONS ----
def load_and_store_chunks():
    """Load text file and store chunks in database"""
    # Check if already loaded
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rag_chunks")
    existing_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    if existing_count > 0:
        print(f"Database already contains {existing_count} chunks. Skipping load.")
        print("To reload data, use the /api/reload endpoint")
        return existing_count
    
    print("Loading text file...")
    raw = Path(INPUT_FILE).read_text(encoding="utf-8", errors="ignore")
    
    print("Creating enhanced chunks with context...")
    chunks = chunk_text(raw)
    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(embed(chunk))
        if (i + 1) % 25 == 0:
            print(f"Embedded {i + 1}/{len(chunks)} chunks")
    
    print("Storing in database...")
    
    # Store in smaller batches to avoid connection timeout
    batch_size = 50
    total_stored = 0
    
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_vectors = vectors[batch_start:batch_end]
        
        # Reconnect for each batch to avoid timeout
        conn = get_db_connection()
        cur = conn.cursor()
        
        rows = [
            (i + batch_start, "shakespeare", chunk, str(vector))
            for i, (chunk, vector) in enumerate(zip(batch_chunks, batch_vectors))
        ]
        
        execute_values(
            cur,
            "INSERT INTO rag_chunks (chunk_index, source, content, embedding) VALUES %s",
            rows,
            template="(%s, %s, %s, %s::vector)",
        )
        
        cur.close()
        conn.close()
        
        total_stored += len(batch_chunks)
        print(f"Stored {total_stored}/{len(chunks)} chunks")
    
    # Final count
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM rag_chunks")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    print(f"Completed! Total chunks in database: {count}")
    return count

def retrieve(question, k=TOP_K):
    """Retrieve k most similar chunks to the question with reranking"""
    question_vector = str(embed(question))
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Retrieve more candidates for reranking
    cur.execute(
        """
        SELECT content, 1 - (embedding <=> %s::vector) AS similarity
        FROM rag_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (question_vector, question_vector, k * 2),  # Get 2x candidates
    )
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Simple keyword-based reranking
    question_words = set(question.lower().split())
    
    def score_result(result):
        content, similarity = result
        content_words = set(content.lower().split())
        keyword_overlap = len(question_words & content_words) / len(question_words) if question_words else 0
        # Combine semantic similarity with keyword matching
        return similarity * 0.7 + keyword_overlap * 0.3
    
    # Rerank and return top k
    reranked = sorted(results, key=score_result, reverse=True)
    return reranked[:k]

def ask_question(question, k=TOP_K, provider=None, model=None):
    """Ask a question using RAG with creative, optimistic responses"""
    # Retrieve relevant chunks
    hits = retrieve(question, k)
    
    # Filter out low-similarity results (lowered threshold for better recall)
    hits = [(content, sim) for content, sim in hits if sim > 0.1]
    
    if not hits:
        return "I couldn't find relevant information in Shakespeare's works to answer this question.", []
    
    context = "\n\n---\n\n".join(hit[0] for hit in hits)
    
    # Direct, structured, concise prompt with database scope awareness
    prompt = f"""Answer the question directly, clearly, and factually based on Shakespeare's works.

Guidelines:
- Keep the response short, clear, and structured (use bullet points or concise sentences).
- Be direct, factual, and concise. Do NOT be poetic, flowery, or overly dramatic.
- Use plain, simple language.
- Database Scope: Due to chunk size limits, the database currently only includes excerpts from three plays: *Romeo and Juliet*, *Coriolanus*, and *Richard III*.
- If the question asks about a character, event, or play NOT present in the provided excerpts (e.g., Hamlet, Macbeth, Othello, King Lear, etc.):
  1. Clearly state: "Note: This information is not present in the database. Due to chunk size limits, the database currently only includes excerpts from *Romeo and Juliet*, *Coriolanus*, and *Richard III*."
  2. Then, provide a brief 1-2 sentence direct factual answer from general Shakespeare knowledge.
- If the question IS covered by the excerpts, answer directly and concisely based on the excerpts.

Context from Shakespeare excerpts:
{context}

Question: {question}

Answer directly and concisely:"""
    
    # Use provided provider/model or fall back to defaults
    use_provider = provider or LLM_PROVIDER
    
    # Get answer based on provider
    try:
        if use_provider == "groq":
            use_model = model or GROQ_MODEL
            answer = get_groq_answer(prompt, use_model)
        else:
            use_model = model or OLLAMA_CHAT_MODEL
            answer = get_ollama_answer(prompt, use_model)
    except Exception as e:
        answer = f"Error getting answer: {str(e)}"
    
    return answer, hits

def get_groq_answer(prompt, model=None):
    """Get answer from Groq API with automatic fallback for decommissioned models"""
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not set in environment variables")
    
    primary_model = model or GROQ_MODEL
    # Groq decommissioned llama-3.1-8b-instant in favor of openai/gpt-oss-20b
    # Prepare fallback candidates recommended by Groq
    candidate_models = [primary_model]
    for fallback in ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)
    
    last_error = None
    for candidate in candidate_models:
        try:
            # Try using Groq SDK if available
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=candidate,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.95
            )
            msg = chat_completion.choices[0].message
            content = msg.content or getattr(msg, 'reasoning', '') or ''
            if content.strip():
                return content
        except ImportError:
            # Fallback to requests if SDK not available
            import requests
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": candidate,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.95
                },
                timeout=60
            )
            if response.status_code == 200:
                res_data = response.json()
                choice_msg = res_data["choices"][0]["message"]
                content = choice_msg.get("content") or choice_msg.get("reasoning") or ""
                if content.strip():
                    return content
            
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"Groq candidate '{candidate}' failed ({error_msg}). Trying fallback...")
            last_error = Exception(error_msg)
        except Exception as e:
            print(f"Groq candidate '{candidate}' failed ({e}). Trying fallback...")
            last_error = e

    # If all Groq attempts fail, try Ollama as secondary safety net
    try:
        print("Groq API failed; attempting local Ollama fallback...")
        return get_ollama_answer(prompt)
    except Exception as ollama_err:
        print(f"Ollama fallback also failed: {ollama_err}")

    raise last_error or Exception("Failed to get answer from Groq")

def get_ollama_answer(prompt, model=None):
    """Get answer from Ollama"""
    use_model = model or OLLAMA_CHAT_MODEL
    
    response = requests.post(
        OLLAMA_URL + "/api/chat",
        json={
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,  # More creative
                "top_p": 0.95,
                "top_k": 40
            }
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

# ---- ROUTES ----
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/architecture')
def architecture():
    """Serve the architecture page"""
    return render_template('architecture.html')

@app.route('/api/reload', methods=['POST'])
def reload_data():
    """Force reload all data (clears and reloads chunks)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Clear existing data
        cur.execute("TRUNCATE rag_chunks")
        cur.close()
        conn.close()
        
        # Reload
        count = load_and_store_chunks()
        
        return jsonify({
            "success": True,
            "message": f"Successfully reloaded {count} chunks"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/load', methods=['POST'])
def load_data():
    """Load and store chunks from input file"""
    try:
        count = load_and_store_chunks()
        return jsonify({
            "success": True,
            "message": f"Successfully loaded {count} chunks into database"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/ask', methods=['POST'])
def ask():
    """Handle question asking"""
    data = request.json
    question = data.get('question', '')
    provider = data.get('provider', None)  # Optional: 'groq' or 'ollama'
    model = data.get('model', None)  # Optional: specific model name
    
    if not question:
        return jsonify({
            "success": False,
            "error": "Question is required"
        }), 400
    
    try:
        answer, hits = ask_question(question, provider=provider, model=model)
        
        # Format sources
        sources = [
            {
                "content": hit[0][:200] + "...",
                "similarity": round(hit[1], 3)
            }
            for hit in hits
        ]
        
        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/status')
def status():
    """Check if database is ready"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_chunks")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "chunk_count": count,
            "database": "connected",
            "llm_provider": LLM_PROVIDER,
            "chat_model": GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_CHAT_MODEL,
            "ollama_url": OLLAMA_URL,
            "enhanced_chunks": True
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/models')
def models():
    """Get available models"""
    available_models = []
    
    # Groq models (only 8B Llama model)
    if GROQ_API_KEY:
        available_models.append({
            "provider": "groq",
            "name": "llama-3.1-8b-instant",
            "display": "Llama 3.1 8B (Groq)"
        })
    
    # Ollama models (check what's available)
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            ollama_models = response.json().get("models", [])
            for model in ollama_models:
                model_name = model.get("name", "")
                # Filter out embedding models from chat selection
                if model_name and "embed" not in model_name.lower():
                    available_models.append({
                        "provider": "ollama",
                        "name": model_name,
                        "display": f"{model_name} (Ollama)"
                    })
    except:
        # Ollama not available or not responding
        pass
    
    return jsonify({
        "success": True,
        "models": available_models,
        "default_provider": LLM_PROVIDER,
        "default_model": GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_CHAT_MODEL
    })

def check_and_start_ollama():
    """Check if Ollama is running, start it if not"""
    import subprocess
    import time
    
    try:
        # Check if Ollama is already running
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            print("[OK] Ollama is already running")
            return True
    except:
        print("[!] Ollama is not running, attempting to start (needed for nomic-embed-text embeddings)...")
        
        try:
            # Try to start Ollama in the background
            if os.name == 'nt':  # Windows
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            else:  # Unix-like
                subprocess.Popen(['ollama', 'serve'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            
            # Wait for Ollama to start
            print("[...] Waiting for Ollama to start...")
            for i in range(30):
                time.sleep(1)
                try:
                    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("[OK] Ollama started successfully")
                        return True
                except:
                    continue
            
            print("[X] Failed to start Ollama automatically")
            print("  Please start Ollama manually: ollama serve")
            return False
            
        except FileNotFoundError:
            print("[X] Ollama is not installed or not in PATH")
            print("  Download from: https://ollama.ai")
            return False
        except Exception as e:
            print(f"[X] Error starting Ollama: {e}")
            return False

if __name__ == '__main__':
    print("Setting up database...")
    setup_database()
    
    print("\n" + "="*60)
    print("Shakespeare RAG Web App")
    print("="*60)
    print(f"\nDatabase: Neon Postgres with pgvector")
    print(f"LLM Provider: {LLM_PROVIDER.upper()}")
    
    if LLM_PROVIDER == "groq":
        print(f"Chat Model: {GROQ_MODEL} (Groq)")
        if not GROQ_API_KEY:
            print("[!] WARNING: GROQ_API_KEY not set!")
    else:
        print(f"Chat Model: {OLLAMA_CHAT_MODEL} (Ollama)")
        print(f"Ollama URL: {OLLAMA_URL}")
    
    print(f"Embed Model: {EMBED_MODEL} (Ollama - required)")
    print("\nChecking services...")
    
    check_and_start_ollama()
    
    if LLM_PROVIDER == "ollama":
        print("\nMake sure you have pulled the chat model:")
        print(f"  ollama pull {OLLAMA_CHAT_MODEL}")
    
    print("\nMake sure you have pulled the embedding model:")
    print(f"  ollama pull {EMBED_MODEL}")
    
    # Auto-load data if database is empty
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM rag_chunks")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if count == 0:
            print("\n[!] Database is empty. Auto-loading data...")
            print("This may take a few minutes...")
            load_and_store_chunks()
    except Exception as e:
        print(f"\n[!] Warning: Could not check database: {e}")
    
    print("\nStarting web server on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
