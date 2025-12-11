# 🔮 Egregore

**Archetype Builder & Semantic Distillation Engine**

Transform images, PDFs, and text into evocative archetypal concepts using AI-powered semantic analysis. Egregore discovers hidden patterns and symbolic relationships in your corpus, distilling them into rich, poetic archetypes.

<p align="center">
  <img src="static/favicon.png" alt="Egregore" width="120">
</p>

---

## ✨ Features

### 🖼️ Multi-Modal Input
- **Images**: Upload images individually or in batches. Describe them using:
  - **Semantic Matching** (Fast): CLIP-based word matching with 4000+ word vocabulary across textures, emotions, colors, animals, body parts, sensations, sounds, and more
  - **Vision LLM** (Detailed): Cloudflare LLaVA for rich natural language descriptions
- **PDFs**: Extract and chunk text with smart strategies (paragraph, sentence, sliding window)
- **Text**: Direct text input with automatic chunking

### 🧠 Intelligent Processing
- **Embedding-based clustering**: Graph-based semantic clustering using BGE embeddings
- **LLM Refinement**: Transform raw clusters into named archetypes with evocative descriptors
- **Directional Mining**: Guide archetype generation with filters (e.g., cardinal directions, seasons, elements)

### 🎛️ Real-Time Interface
- **WebSocket-powered UI**: Live progress updates during mining
- **3D Visualization**: Interactive embedding space visualization with Three.js
- **Export/Import**: Save and load your archetypal discoveries

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) account (free tier available)

### Installation

```bash
# Clone the repository
git clone https://github.com/KairosHive/egregore.git
cd egregore

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For CLIP-based semantic matching (optional but recommended)
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### Configuration

Create a `.streamlit/secrets.toml` file (or set environment variables):

```toml
CLOUDFLARE_ACCOUNT_ID = "your-account-id"
CLOUDFLARE_API_TOKEN = "your-api-token"
```

Or use environment variables:
```bash
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_API_TOKEN="your-api-token"
```

### Run

```bash
# Start the server
uvicorn miner_server:app --reload --port 8765

# Open in browser
# http://localhost:8765
```

---

## 📖 Usage Guide

### 1. Add Data Sources

**Images**
- Click "Select Images" to upload one or more images
- Enable "Describe Images" to generate semantic descriptions
- Choose method:
  - **Semantic (CLIP + LLM)**: Fast (~0.5s/image), uses 4000+ word vocabulary
  - **Vision LLM**: Slower (~3-5s/image), more detailed descriptions

**Text**
- Paste text directly into the text input area
- Click "Add Text" to add to corpus

**PDFs**
- Upload PDF files for automatic text extraction

### 2. Configure Mining

**Mining Mode**
- **Open Discovery**: Let the algorithm find natural clusters
- **Directional**: Guide with filters (e.g., "North, South, East, West")

**Parameters**
- **K-Neighbors**: Graph connectivity (higher = more connections)
- **Min Cluster Size**: Minimum concepts per archetype
- **Resolution**: Clustering granularity

### 3. Run Mining

Click **"Begin Mining"** to start the process:
1. Embedding generation for all corpus items
2. Similarity graph construction
3. Community detection / clustering
4. LLM refinement into named archetypes

### 4. Explore Results

- View discovered archetypes with their descriptors
- Interact with the 3D embedding visualization
- Export results as JSON for use in other applications

---

## 🏗️ Architecture

```
egregore/
├── miner_server.py      # FastAPI server + WebSocket handlers
├── enhanced_miner.py    # Core mining logic + LLM integration
├── text_embedder.py     # Multi-backend text embedding
├── miner_ui.html        # Single-page web interface
├── requirements.txt     # Python dependencies
└── static/              # Static assets
```

### Key Components

| Component | Description |
|-----------|-------------|
| `EnhancedArchetypeMiner` | Main orchestrator: ingestion, embedding, clustering, refinement |
| `SemanticWordMatcher` | CLIP + FAISS image→words matching with 4000+ vocabulary |
| `VisionDescriber` | Cloudflare LLaVA integration for image descriptions |
| `LLMArchetypeRefiner` | LLM-powered cluster→archetype transformation |
| `TextEmbedder` | Multi-backend embeddings (Cloudflare BGE, Sentence Transformers) |

### Semantic Matching Performance

The `SemanticWordMatcher` uses **FAISS** (Facebook AI Similarity Search) for lightning-fast vector similarity:

- **4000+ word vocabulary** embedded with CLIP
- **FAISS IndexFlatIP** for exact cosine similarity search
- **~1ms search time** per image (vs ~10ms with numpy)
- Automatic fallback to numpy if FAISS not installed

### Vocabulary Categories

The semantic matcher includes rich vocabularies for diverse matching:

| Category | Examples | Count |
|----------|----------|-------|
| Evocative | ethereal, liminal, crystalline, volcanic | ~250 |
| Archetypal | transcendence, metamorphosis, prophecy | ~800 |
| Animals | wolf, octopus, salamander, raven | ~250 |
| Body | spine, iris, sinew, ventricle | ~200 |
| Textures | gossamer, corroded, fibrous, velvety | ~400 |
| Sensations | euphoric, smoldering, tingling | ~250 |
| Flavors | umami, astringent, saccharine | ~180 |
| Colors | vermillion, cerulean, obsidian | ~250 |
| Nature | fjord, aurora, tundra, monsoon | ~300 |
| Emotions | melancholy, euphoria, serenity | ~250 |
| Sounds | reverberation, cacophony, whisper | ~150 |
| Actions | coalesce, crystallize, meander | ~600 |

---

## 🌐 Deployment

### Railway / Render / Fly.io

The project includes deployment configs:

```bash
# Procfile
web: uvicorn miner_server:app --host 0.0.0.0 --port ${PORT:-8765}
```

```toml
# nixpacks.toml
[phases.setup]
nixPkgs = ["python311"]

[start]
cmd = "uvicorn miner_server:app --host 0.0.0.0 --port ${PORT:-8765}"
```

Set environment variables in your deployment platform:
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

---

## 🧪 Development

```bash
# Run with auto-reload
uvicorn miner_server:app --reload --port 8765

# Run with specific host (for network access)
uvicorn miner_server:app --host 0.0.0.0 --port 8765
```

### Caching

- CLIP word embeddings are cached at `.cache/word_embeddings.npy`
- First run computes embeddings (~4000 words), subsequent runs load from cache

---

## 📚 Concepts

### What is an Archetype?

In Egregore, an archetype is a distilled semantic pattern discovered from your corpus. Each archetype has:
- **Name**: An evocative 1-2 word identifier (e.g., "DARK ZENITH", "SILENT VECTOR")
- **Descriptors**: 10-12 rare, poetic words capturing the archetype's essence
- **Essence**: A philosophical sentence describing the archetype's meaning

### The Mining Process

1. **Ingestion**: Convert images/text into embeddable chunks
2. **Embedding**: Transform chunks into high-dimensional vectors
3. **Graph Construction**: Build similarity network between concepts
4. **Clustering**: Detect communities using Louvain algorithm
5. **Refinement**: LLM transforms clusters into named archetypes

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) for embedding and LLM APIs
- [OpenAI CLIP](https://github.com/openai/CLIP) for image-text semantic matching
- [Three.js](https://threejs.org/) for 3D visualization

---

<p align="center">
  <i>Egregore: Where meaning crystallizes from chaos.</i>
</p>
