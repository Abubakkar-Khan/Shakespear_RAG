# Shakespeare RAG Web App

A web application for querying Shakespeare's works using RAG (Retrieval Augmented Generation) powered by Groq API (Llama 3.1 8B) or local Ollama, Neon Postgres with pgvector, and Ollama embeddings.

## Features

- **LLM Support**: Groq API with Llama 3.1 8B (fast inference with fallback resilience) and local Ollama support
- **Vector Search**: Neon Postgres with `pgvector` for semantic search
- **Modern Web Interface**: Responsive UI to query indexed Shakespeare plays
- **Grounded Answers**: Direct, structured answers derived from indexed text chunks

## Prerequisites

1. **Groq API Key** (optional, recommended for cloud inference):
   - Get your key from https://console.groq.com
   
2. **Ollama** (for embeddings and optional local chat):
   - Download from: https://ollama.ai
   - Pull the required embedding model:
     ```bash
     ollama pull nomic-embed-text
     ```
   - (Optional) If using Ollama for chat:
     ```bash
     ollama pull qwen2:1.5b
     ```

3. **Neon Postgres database** with `pgvector` extension:
   - Create a free instance at https://neon.tech

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Update database URL** (if needed):
   - Edit `app.py` and update `NEON_DB_URL` with your Neon connection string

## Running the App

1. **Make sure Ollama is running**:
   ```bash
   ollama serve
   ```

2. **Start the web app**:
   ```bash
   python app.py
   ```

3. **Open in browser**:
   - Navigate to: http://localhost:5000

4. **First time setup**:
   - Click "Load Shakespeare Texts" to process and store the text chunks
   - This takes about 1-2 minutes

5. **Ask questions**:
   - Type any question about Shakespeare's works
   - Or click example questions to get started

## How It Works

1. **Text Chunking**: Shakespeare's texts are split into overlapping ~1000 character chunks
2. **Embedding**: Each chunk is converted to a 768-dimensional vector using nomic-embed-text
3. **Storage**: Chunks and vectors are stored in Neon Postgres with pgvector
4. **Retrieval**: When you ask a question, it finds the 4 most similar chunks using cosine distance
5. **Generation**: The local Ollama model (qwen2:1.5b) answers using only the retrieved context

## Configuration

Key parameters in `app.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 600 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100 characters)
- `TOP_K`: Number of chunks to retrieve (default: 6)
- `MAX_CHUNKS`: Maximum chunks loaded (default: 2500)
- `LLM_PROVIDER`: LLM provider (`groq` or `ollama`)
- `GROQ_MODEL`: Groq model (default: `llama-3.1-8b-instant`)
- `EMBED_MODEL`: Ollama model for embeddings (default: `nomic-embed-text`)

## Example Questions

- "What does Romeo want?" (*Romeo and Juliet*)
- "Why does Tybalt challenge Romeo?" (*Romeo and Juliet*)
- "Who is Caius Marcius and what do the citizens want?" (*Coriolanus*)
- "What are Gloucester's motives in Richard III?" (*Richard III*)

## Troubleshooting

### "Error getting answer from Ollama"
- Make sure Ollama is running: `ollama serve`
- Verify models are pulled: `ollama list`

### "Database connection error"
- Check your `NEON_DB_URL` is correct
- Verify the database is accessible from your network

### "ModuleNotFoundError"
- Install requirements: `pip install -r requirements.txt`

## Architecture

```
┌─────────────┐
│   Browser   │
│  (HTML/JS)  │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   Flask     │
│   Server    │
└──────┬──────┘
       │
       ├─────► Neon Postgres (pgvector)
       │       └─ Stores chunks + embeddings
       │
       └─────► Ollama (localhost:11434)
               ├─ nomic-embed-text (embeddings)
               └─ qwen2:1.5b (answers)
```

## Credits

- Shakespeare texts from the public domain
- Powered by Ollama and local AI models
- Vector search with Neon Postgres and pgvector
