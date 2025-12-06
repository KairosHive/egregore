"""
Real-time Archetype Miner Server
================================
FastAPI + WebSocket server for reactive, real-time monitoring of mining computations.

Run with: uvicorn miner_server:app --reload --port 8765
"""

import asyncio
import json
import os
import sys
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
import traceback

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================
# Load Cloudflare credentials from Streamlit secrets if not in env
# ============================================================
def load_secrets():
    """Load secrets from .streamlit/secrets.toml if env vars not set."""
    if os.environ.get("CLOUDFLARE_ACCOUNT_ID") and os.environ.get("CLOUDFLARE_API_TOKEN"):
        return  # Already set
    
    secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Manual parsing for simple TOML
                with open(secrets_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("CLOUDFLARE_ACCOUNT_ID"):
                            val = line.split("=", 1)[1].strip().strip('"\'')
                            os.environ["CLOUDFLARE_ACCOUNT_ID"] = val
                        elif line.startswith("CLOUDFLARE_API_TOKEN"):
                            val = line.split("=", 1)[1].strip().strip('"\'')
                            os.environ["CLOUDFLARE_API_TOKEN"] = val
                return
        
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
            if "CLOUDFLARE_ACCOUNT_ID" in secrets:
                os.environ["CLOUDFLARE_ACCOUNT_ID"] = secrets["CLOUDFLARE_ACCOUNT_ID"]
            if "CLOUDFLARE_API_TOKEN" in secrets:
                os.environ["CLOUDFLARE_API_TOKEN"] = secrets["CLOUDFLARE_API_TOKEN"]

load_secrets()

app = FastAPI(title="Archetype Miner", version="2.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Data Classes
# ============================================================

@dataclass
class PDFItem:
    id: str
    path: str
    name: str
    pages: int = 0
    chunks: int = 0
    raw_pages: List[Dict] = field(default_factory=list)  # Store raw text per page for rechunking

@dataclass
class ImageItem:
    id: str
    path: str
    name: str
    folder: str = ""
    description: Optional[str] = None

@dataclass
class TextItem:
    id: str
    text: str
    source: str
    chunks: int = 1

@dataclass
class MiningJob:
    id: str
    status: str  # pending, running, completed, failed
    created_at: str
    config: Dict[str, Any]
    progress: float = 0.0
    stage: str = ""
    logs: List[str] = field(default_factory=list)
    result: Optional[Dict] = None
    error: Optional[str] = None


class MinerState:
    """Per-session state for a single user."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.jobs: Dict[str, MiningJob] = {}
        self.current_job: Optional[str] = None
        self.pdfs: List[PDFItem] = []
        self.images: List[ImageItem] = []
        self.texts: List[TextItem] = []
        self.text_chunks: List[Dict] = []  # Actual chunks for mining
        self.websockets: List[WebSocket] = []
        self.embedder = None
        self.image_encoder = None
        self._current_embedder = None  # Track which embedder is loaded
    
    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
        
    @property
    def corpus_summary(self):
        return {
            "pdfs": len(self.pdfs),
            "images": len(self.images),
            "texts": len(self.texts),
            "chunks": len(self.text_chunks),
            "pdf_items": [{"name": p.name, "pages": p.pages, "chunks": p.chunks} for p in self.pdfs],
            "image_items": [{"name": i.name, "folder": i.folder} for i in self.images[:50]],
            "text_items": [{"source": t.source, "chunks": t.chunks} for t in self.texts],
        }
        
    async def broadcast(self, message: Dict):
        """Send message to all connected WebSocket clients for this session."""
        dead = []
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception as e:
                print(f"[Broadcast] Error sending to websocket: {e}")
                dead.append(ws)
        for ws in dead:
            self.websockets.remove(ws)
    
    def log(self, job_id: str, message: str):
        """Add log entry to job."""
        if job_id in self.jobs:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            entry = f"[{timestamp}] {message}"
            self.jobs[job_id].logs.append(entry)
            return entry
        return None
    
    def clear(self):
        """Clear all corpus data."""
        self.pdfs = []
        self.images = []
        self.texts = []
        self.text_chunks = []


class SessionManager:
    """Manages per-user sessions to isolate state between users."""
    
    # Session timeout in seconds (1 hour)
    SESSION_TIMEOUT = 3600
    # Cleanup interval in seconds (5 minutes)
    CLEANUP_INTERVAL = 300
    
    def __init__(self):
        self.sessions: Dict[str, MinerState] = {}
        self._cleanup_task = None
    
    def get_or_create(self, session_id: str) -> MinerState:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            print(f"[SessionManager] Creating new session: {session_id[:8]}...")
            self.sessions[session_id] = MinerState(session_id)
        else:
            self.sessions[session_id].touch()
        return self.sessions[session_id]
    
    def get(self, session_id: str) -> Optional[MinerState]:
        """Get session if it exists."""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session
    
    def remove(self, session_id: str):
        """Remove a session."""
        if session_id in self.sessions:
            print(f"[SessionManager] Removing session: {session_id[:8]}...")
            del self.sessions[session_id]
    
    async def cleanup_expired(self):
        """Remove expired sessions (no activity for SESSION_TIMEOUT)."""
        now = time.time()
        expired = []
        for sid, session in self.sessions.items():
            # Only expire if no websockets connected and timed out
            if not session.websockets and (now - session.last_activity) > self.SESSION_TIMEOUT:
                expired.append(sid)
        
        for sid in expired:
            print(f"[SessionManager] Expiring inactive session: {sid[:8]}...")
            del self.sessions[sid]
        
        if expired:
            print(f"[SessionManager] Cleaned up {len(expired)} expired sessions. Active: {len(self.sessions)}")
    
    async def start_cleanup_task(self):
        """Start background task to clean up expired sessions."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.CLEANUP_INTERVAL)
                await self.cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    @property
    def active_count(self) -> int:
        """Number of active sessions."""
        return len(self.sessions)


# Global session manager (not per-user state)
session_manager = SessionManager()

# For backwards compatibility during transition - will be removed
# This is now a function that raises an error if used directly
class _DeprecatedState:
    def __getattr__(self, name):
        raise RuntimeError("Global 'state' is deprecated. Use session_manager.get_or_create(session_id) instead.")

state = _DeprecatedState()

# ============================================================
# WebSocket Handler
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup."""
    await session_manager.start_cleanup_task()
    print(f"[Server] Session cleanup task started (timeout: {SessionManager.SESSION_TIMEOUT}s)")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Wait for the first message which should contain the session ID
    try:
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except asyncio.TimeoutError:
        await websocket.close(code=4000, reason="Session ID not received")
        return
    
    # Get or create session
    session_id = init_data.get("session_id")
    if not session_id:
        # Generate a new session ID if client didn't provide one
        session_id = str(uuid.uuid4())
    
    session = session_manager.get_or_create(session_id)
    session.websockets.append(websocket)
    
    print(f"[WebSocket] Client connected to session {session_id[:8]}... (active sessions: {session_manager.active_count})")
    
    # Send current state on connect
    await websocket.send_json({
        "type": "init",
        "session_id": session_id,
        "corpus": session.corpus_summary,
        "jobs": {jid: asdict(job) for jid, job in session.jobs.items()},
        "current_job": session.current_job
    })
    
    try:
        while True:
            data = await websocket.receive_json()
            await handle_ws_message(websocket, session, data)
    except WebSocketDisconnect:
        if websocket in session.websockets:
            session.websockets.remove(websocket)
        print(f"[WebSocket] Client disconnected from session {session_id[:8]}... (remaining: {len(session.websockets)})")


async def handle_ws_message(ws: WebSocket, session: MinerState, data: Dict):
    """Handle incoming WebSocket messages."""
    msg_type = data.get("type")
    session.touch()  # Update last activity
    
    if msg_type == "ping":
        await ws.send_json({"type": "pong"})
    
    elif msg_type == "add_text":
        await handle_add_text(session, data)
    
    elif msg_type == "scan_pdfs":
        await handle_scan_pdfs(session, data)
    
    elif msg_type == "scan_images":
        await handle_scan_images(session, data)
    
    elif msg_type == "clear_corpus":
        session.clear()
        await session.broadcast({"type": "corpus_cleared"})
    
    elif msg_type == "remove_item":
        await handle_remove_item(session, data)
    
    elif msg_type == "start_mining":
        config = data.get("config", {})
        await start_mining_job(session, config)
    
    elif msg_type == "cancel_job":
        job_id = data.get("job_id")
        if job_id and job_id in session.jobs:
            session.jobs[job_id].status = "cancelled"
            await session.broadcast({"type": "job_cancelled", "job_id": job_id})
    
    elif msg_type == "rechunk_pdfs":
        await handle_rechunk_pdfs(session, data)


# ============================================================
# Data Ingestion Handlers
# ============================================================

async def handle_add_text(session: MinerState, data: Dict):
    """Add text to corpus with chunking."""
    text = data.get("text", "")
    source = data.get("source", "manual")
    chunking = data.get("chunking", "paragraph")
    params = data.get("chunk_params", {})
    
    if not text.strip():
        return
    
    # Import chunker
    try:
        from enhanced_miner import TextChunker
        has_chunker = True
    except ImportError:
        has_chunker = False
    
    # Apply chunking with parameters
    if chunking == "paragraph":
        min_len = params.get("min_length", 50)
        max_len = params.get("max_length", 1000)
        if has_chunker:
            chunks = TextChunker.chunk_paragraphs(text, min_length=min_len, max_length=max_len)
        else:
            chunks = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) >= min_len]
    elif chunking == "sentence":
        sent_per = params.get("sentences_per_chunk", 5)
        overlap = params.get("overlap_sentences", 1)
        if has_chunker:
            chunks = TextChunker.chunk_sentences(text, sentences_per_chunk=sent_per, overlap_sentences=overlap)
        else:
            import re
            chunks = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip() and len(s.strip()) > 10]
    elif chunking == "sliding":
        window = params.get("window_size", 512)
        stride = params.get("stride", 256)
        if has_chunker:
            chunks = TextChunker.chunk_sliding_window(text, window_size=window, stride=stride)
        else:
            chunks = [text[i:i+window] for i in range(0, len(text), stride) if len(text[i:i+window].strip()) > 50]
    else:
        # none / page
        chunks = [text] if text.strip() else []
    
    # Add to session
    text_item = TextItem(
        id=f"text_{len(session.texts)}",
        text=text[:200] + "..." if len(text) > 200 else text,
        source=source,
        chunks=len(chunks)
    )
    session.texts.append(text_item)
    
    # Add chunks
    base_idx = len(session.text_chunks)
    for i, chunk in enumerate(chunks):
        session.text_chunks.append({
            "id": f"{source}_{base_idx + i}",
            "text": chunk,
            "source": source
        })
    
    await session.broadcast({
        "type": "text_added",
        "texts": [{"source": source, "chunks": len(chunks)}],
        "chunks": len(chunks),
        "total_chunks": len(session.text_chunks)
    })
    
    await session.broadcast({
        "type": "corpus_updated",
        **session.corpus_summary
    })


async def handle_scan_pdfs(session: MinerState, data: Dict):
    """Scan folder for PDFs and extract text."""
    folder_path = data.get("path", "")
    recursive = data.get("recursive", True)
    chunking = data.get("chunking", "paragraph")
    params = data.get("chunk_params", {})
    
    if not folder_path:
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        await session.broadcast({
            "type": "scan_progress",
            "message": f"Folder not found: {folder_path}"
        })
        return
    
    # Find PDFs
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = list(folder.glob(pattern))
    
    if not pdf_files:
        await session.broadcast({
            "type": "scan_progress",
            "message": "No PDF files found"
        })
        return
    
    await session.broadcast({
        "type": "scan_progress",
        "message": f"Found {len(pdf_files)} PDFs, extracting text..."
    })
    
    # Import PDF extractor
    try:
        from enhanced_miner import PDFExtractor, TextChunker
        has_extractor = True
    except ImportError:
        has_extractor = False
        await session.broadcast({
            "type": "scan_progress",
            "message": "PDF extraction not available (install pymupdf or pdfplumber)"
        })
        return
    
    # Process PDFs
    total_chunks = 0
    new_pdfs = []
    
    for pdf_path in pdf_files:
        try:
            pages = PDFExtractor.extract_text(str(pdf_path))
            
            pdf_chunks = 0
            for page_data in pages:
                text = page_data.get("text", "")
                if not text.strip():
                    continue
                
                # Apply chunking with parameters
                if chunking == "paragraph":
                    min_len = params.get("min_length", 50)
                    max_len = params.get("max_length", 1000)
                    chunks = TextChunker.chunk_paragraphs(text, min_length=min_len, max_length=max_len)
                elif chunking == "sentence":
                    sent_per = params.get("sentences_per_chunk", 5)
                    overlap = params.get("overlap_sentences", 1)
                    chunks = TextChunker.chunk_sentences(text, sentences_per_chunk=sent_per, overlap_sentences=overlap)
                elif chunking == "sliding":
                    window = params.get("window_size", 512)
                    stride = params.get("stride", 256)
                    chunks = TextChunker.chunk_sliding_window(text, window_size=window, stride=stride)
                elif chunking == "page":
                    chunks = [text]
                else:
                    chunks = TextChunker.chunk_paragraphs(text)
                
                base_idx = len(session.text_chunks)
                for i, chunk in enumerate(chunks):
                    session.text_chunks.append({
                        "id": f"{pdf_path.name}_p{page_data['page']}_{base_idx + i}",
                        "text": chunk,
                        "source": pdf_path.name,
                        "page": page_data["page"]
                    })
                    pdf_chunks += 1
            
            pdf_item = PDFItem(
                id=f"pdf_{len(session.pdfs)}",
                path=str(pdf_path),
                name=pdf_path.name,
                pages=len(pages),
                chunks=pdf_chunks
            )
            session.pdfs.append(pdf_item)
            new_pdfs.append({"name": pdf_path.name, "pages": len(pages), "chunks": pdf_chunks})
            total_chunks += pdf_chunks
            
        except Exception as e:
            await session.broadcast({
                "type": "scan_progress",
                "message": f"Error reading {pdf_path.name}: {e}"
            })
    
    await session.broadcast({
        "type": "pdf_scanned",
        "pdfs": new_pdfs,
        "chunks": total_chunks
    })
    
    await session.broadcast({
        "type": "corpus_updated",
        **session.corpus_summary
    })


async def handle_scan_images(session: MinerState, data: Dict):
    """Scan folder for images."""
    folder_path = data.get("path", "")
    recursive = data.get("recursive", True)
    max_files = data.get("max_files", 500)
    describe = data.get("describe", False)
    
    if not folder_path:
        return
    
    folder = Path(folder_path)
    if not folder.exists():
        await session.broadcast({
            "type": "scan_progress",
            "message": f"Folder not found: {folder_path}"
        })
        return
    
    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    pattern = "**/*" if recursive else "*"
    
    image_files = []
    for f in folder.glob(pattern):
        if f.suffix.lower() in extensions:
            image_files.append(f)
            if len(image_files) >= max_files:
                break
    
    if not image_files:
        await session.broadcast({
            "type": "scan_progress",
            "message": "No image files found"
        })
        return
    
    await session.broadcast({
        "type": "scan_progress",
        "message": f"Found {len(image_files)} images"
    })
    
    # Add images
    new_images = []
    for img_path in image_files:
        img_item = ImageItem(
            id=f"img_{len(session.images)}",
            path=str(img_path),
            name=img_path.name,
            folder=str(img_path.parent.name)
        )
        session.images.append(img_item)
        new_images.append({"name": img_path.name, "folder": img_item.folder})
    
    # Vision description if requested
    if describe:
        await session.broadcast({
            "type": "scan_progress",
            "message": "Describing images with Vision LLM..."
        })
        
        try:
            from enhanced_miner import VisionDescriber
            describer = VisionDescriber()
            
            success_count = 0
            for i, img in enumerate(session.images[-len(new_images):]):
                if i % 5 == 0:
                    await session.broadcast({
                        "type": "scan_progress",
                        "message": f"Describing image {i+1}/{len(new_images)}..."
                    })
                
                try:
                    desc = describer.describe_image(img.path)
                    if desc:
                        img.description = desc
                        success_count += 1
                except Exception as e:
                    print(f"[Vision] Error describing {img.name}: {e}")
                
                await asyncio.sleep(0.01)  # Allow UI updates
            
            await session.broadcast({
                "type": "scan_progress",
                "message": f"Described {success_count}/{len(new_images)} images"
            })
                
        except ImportError as e:
            print(f"[Vision] Import error: {e}")
            await session.broadcast({
                "type": "scan_progress",
                "message": "Vision describer not available"
            })
    
    await session.broadcast({
        "type": "images_scanned",
        "images": new_images,
        "count": len(new_images)
    })
    
    await session.broadcast({
        "type": "corpus_updated",
        **session.corpus_summary
    })


async def handle_remove_item(session: MinerState, data: Dict):
    """Remove an item from corpus."""
    item_type = data.get("item_type")
    index = data.get("index", -1)
    
    if item_type == "pdf" and 0 <= index < len(session.pdfs):
        removed = session.pdfs.pop(index)
        # Also remove associated chunks
        session.text_chunks = [c for c in session.text_chunks if c.get("source") != removed.name]
    elif item_type == "image" and 0 <= index < len(session.images):
        session.images.pop(index)
    elif item_type == "text" and 0 <= index < len(session.texts):
        removed = session.texts.pop(index)
        session.text_chunks = [c for c in session.text_chunks if c.get("source") != removed.source]
    
    await session.broadcast({
        "type": "corpus_updated",
        **session.corpus_summary
    })


async def handle_rechunk_pdfs(session: MinerState, data: Dict):
    """Rechunk all PDFs with new chunking parameters."""
    chunking = data.get("chunking", "paragraph")
    params = data.get("chunk_params", {})
    
    if not session.pdfs:
        return
    
    # Import chunker
    try:
        from enhanced_miner import TextChunker
    except ImportError:
        await session.broadcast({
            "type": "scan_progress",
            "message": "TextChunker not available"
        })
        return
    
    await session.broadcast({
        "type": "scan_progress",
        "message": f"Rechunking {len(session.pdfs)} PDFs with {chunking} method..."
    })
    
    # Remove all PDF-sourced chunks
    pdf_names = {pdf.name for pdf in session.pdfs}
    session.text_chunks = [c for c in session.text_chunks if c.get("source") not in pdf_names]
    
    # Re-chunk each PDF from stored raw pages
    total_new_chunks = 0
    for pdf in session.pdfs:
        if not pdf.raw_pages:
            continue
        
        pdf_chunks = 0
        for page_data in pdf.raw_pages:
            text = page_data.get("text", "")
            if not text.strip():
                continue
            
            # Apply new chunking with parameters
            if chunking == "paragraph":
                min_len = params.get("min_length", 50)
                max_len = params.get("max_length", 1000)
                chunks = TextChunker.chunk_paragraphs(text, min_length=min_len, max_length=max_len)
            elif chunking == "sentence":
                sent_per = params.get("sentences_per_chunk", 5)
                overlap = params.get("overlap_sentences", 1)
                chunks = TextChunker.chunk_sentences(text, sentences_per_chunk=sent_per, overlap_sentences=overlap)
            elif chunking == "sliding":
                window = params.get("window_size", 512)
                stride = params.get("stride", 256)
                chunks = TextChunker.chunk_sliding_window(text, window_size=window, stride=stride)
            elif chunking == "page":
                chunks = [text]
            else:
                chunks = TextChunker.chunk_paragraphs(text)
            
            base_idx = len(session.text_chunks)
            for i, chunk in enumerate(chunks):
                session.text_chunks.append({
                    "id": f"{pdf.name}_p{page_data['page']}_{base_idx + i}",
                    "text": chunk,
                    "source": pdf.name,
                    "page": page_data["page"]
                })
                pdf_chunks += 1
        
        pdf.chunks = pdf_chunks
        total_new_chunks += pdf_chunks
    
    await session.broadcast({
        "type": "rechunk_complete",
        "chunks": total_new_chunks,
        "message": f"Rechunked to {total_new_chunks} chunks"
    })
    
    await session.broadcast({
        "type": "corpus_updated",
        **session.corpus_summary
    })


# ============================================================
# Ambiguity Metrics Computation
# Matches EXACTLY the formulas in delyrism.py SymbolSpace class
# Computes metrics on DESCRIPTOR EMBEDDINGS (not corpus chunks)
# ============================================================

def _softmax(x, tau=1.0):
    """Temperature-scaled softmax matching delyrism.py implementation."""
    z = (x - np.max(x)) / max(tau, 1e-6)
    e = np.exp(z)
    return e / np.sum(e)

def _entropy(p, eps=1e-12):
    """Entropy matching delyrism.py implementation."""
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def _l2_normalize(X, axis=1, eps=1e-9):
    """L2 normalize embeddings."""
    n = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / n

def compute_archetype_metrics(archetypes: Dict, embedder, clusters: Dict = None, ids: List = None, 
                              corpus_embeddings: np.ndarray = None, cluster_to_archetype: Dict = None) -> Dict:
    """
    Compute dispersion, leakage, and soft_entropy metrics for each archetype.
    
    CRITICAL: This embeds the ARCHETYPE DESCRIPTORS (like delyrism.py SymbolSpace)
    and computes metrics on those embeddings, NOT on corpus chunks.
    
    Args:
        archetypes: Dict mapping archetype name -> list of descriptor strings
        embedder: TextEmbedder instance to embed descriptors
        clusters, ids, corpus_embeddings, cluster_to_archetype: Optional, for corpus-based stats
    
    Returns per-archetype metrics, corpus-level aggregates, and inter-archetype similarity matrix.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    
    if not archetypes:
        return {"per_archetype": {}, "corpus": {}, "inter_similarity": {}, "archetype_order": []}
    
    # ====================================================================
    # STEP 1: Embed all descriptors for each archetype
    # This mirrors delyrism.py's self.D (descriptor embeddings)
    # ====================================================================
    archetype_order = list(archetypes.keys())
    
    # Collect all descriptors and their ownership
    all_descriptors = []
    descriptor_to_archetype = {}
    archetype_to_descriptor_indices = {name: [] for name in archetype_order}
    
    for name in archetype_order:
        descriptors = archetypes[name]
        if isinstance(descriptors, list):
            for desc in descriptors:
                if isinstance(desc, str) and desc.strip():
                    idx = len(all_descriptors)
                    all_descriptors.append(desc.strip())
                    descriptor_to_archetype[idx] = name
                    archetype_to_descriptor_indices[name].append(idx)
    
    # Brief summary
    print(f"[Metrics] Computing metrics for {len(all_descriptors)} descriptors across {len(archetype_order)} archetypes")
    
    if not all_descriptors:
        return {"per_archetype": {}, "corpus": {}, "inter_similarity": {}, "archetype_order": []}
    
    # Embed all descriptors at once (efficient batching)
    try:
        D = embedder.encode(all_descriptors)
        D = _l2_normalize(D)
        
    except Exception as e:
        print(f"[Metrics] Embedding failed: {e}")
        return {"per_archetype": {}, "corpus": {}, "inter_similarity": {}, "archetype_order": archetype_order}
    
    # ====================================================================
    # STEP 2: Compute centroids for each archetype (like delyrism.py symbol_centroids)
    # NOTE: delyrism.py does NOT re-normalize centroids after averaging
    # ====================================================================
    centroids = {}
    archetype_embeddings = {}  # name -> array of descriptor embeddings
    
    for name in archetype_order:
        indices = archetype_to_descriptor_indices[name]
        if indices:
            embs = D[indices]
            archetype_embeddings[name] = embs
            # Match delyrism.py: centroid is mean of L2-normalized vectors, NOT re-normalized
            centroids[name] = embs.mean(axis=0)
    
    # Centroid matrix for soft_entropy
    centroid_matrix = np.stack([centroids[name] for name in archetype_order if name in centroids])
    
    # Build k-NN for leakage computation (on all descriptors)
    k_leakage = min(10, len(D) - 1)
    if k_leakage > 0 and len(D) > 1:
        nbrs = NearestNeighbors(metric="cosine", n_neighbors=k_leakage + 1).fit(D)
    else:
        nbrs = None
    
    # Max entropy for normalization
    max_entropy = np.log(len(centroids)) if len(centroids) > 1 else 1.0
    
    # ====================================================================
    # STEP 3: Compute per-archetype metrics (matching delyrism.py exactly)
    # ====================================================================
    per_archetype = {}
    all_dispersion = []
    all_leakage = []
    all_entropy = []
    
    for name in archetype_order:
        if name not in archetype_embeddings:
            continue
            
        embs = archetype_embeddings[name]
        indices = archetype_to_descriptor_indices[name]
        n = len(embs)
        
        # ----------------------------------------------------------------
        # DISPERSION: mean pairwise dissimilarity within archetype
        # Formula: mean(1 - S[upper_triangle]) where S is cosine similarity
        # ----------------------------------------------------------------
        if n >= 2:
            S = cosine_similarity(embs)
            triu_idx = np.triu_indices(n, k=1)
            dispersion = float(np.mean(1 - S[triu_idx]))
        else:
            dispersion = 0.0
        
        # ----------------------------------------------------------------
        # LEAKAGE: fraction of k-NN from other archetypes
        # For each descriptor, find k neighbors, count those from other archetypes
        # ----------------------------------------------------------------
        if nbrs is not None and k_leakage > 0:
            k = min(k_leakage, len(D) - 1)
            _, nbr_indices = nbrs.kneighbors(embs, n_neighbors=k + 1)
            leakages = []
            for r, row in enumerate(nbr_indices):
                my_global_idx = indices[r]
                # Exclude self
                row = [j for j in row if j != my_global_idx][:k]
                # Count neighbors from other archetypes
                leak_count = sum(1 for j in row if descriptor_to_archetype.get(j) != name)
                leakages.append(leak_count / k if k > 0 else 0.0)
            leakage = float(np.mean(leakages)) if leakages else 0.0
        else:
            leakage = 0.0
        
        # ----------------------------------------------------------------
        # SOFT ENTROPY: how spread out soft assignments are across archetypes
        # For each descriptor: p = softmax(emb @ centroids.T, tau), then entropy(p)
        # 
        # CRITICAL: With L2-normalized embeddings, cosine sims cluster in narrow 
        # range (e.g., 0.59-0.65). Standard softmax can't distinguish these well.
        # Solution: Z-score normalize logits THEN apply softmax with moderate tau.
        # This makes the relative differences matter, not the absolute values.
        # ----------------------------------------------------------------
        if centroid_matrix is not None and len(centroid_matrix) > 1:
            entropies = []
            for i_emb, emb in enumerate(embs):
                logits = emb @ centroid_matrix.T
                
                # Z-score normalize logits to amplify relative differences
                logits_mean = logits.mean()
                logits_std = logits.std()
                if logits_std > 1e-9:
                    logits_z = (logits - logits_mean) / logits_std
                else:
                    logits_z = logits - logits_mean
                
                # Now apply softmax with tau=1.0 (z-scores are already scaled)
                probs = _softmax(logits_z, tau=1.0)
                ent = _entropy(probs)
                entropies.append(ent)
            
            # Normalize to [0,1] range
            soft_entropy = float(np.mean(entropies)) / max_entropy
        else:
            soft_entropy = 0.0
        
        per_archetype[name] = {
            "dispersion": round(dispersion, 4),
            "leakage": round(leakage, 4),
            "entropy": round(soft_entropy, 4),
            "size": n  # Number of descriptors
        }
        
        all_dispersion.append(dispersion)
        all_leakage.append(leakage)
        all_entropy.append(soft_entropy)
    
    # ====================================================================
    # STEP 4: Inter-archetype similarity matrix
    # Using sigmoid contrast enhancement for better visualization
    # Raw cosine sims cluster in narrow band - sigmoid emphasizes differences
    # ====================================================================
    inter_similarity = {}
    if len(centroids) > 1:
        centroid_list = [centroids[name] for name in archetype_order if name in centroids]
        names_with_centroids = [name for name in archetype_order if name in centroids]
        sim_matrix = cosine_similarity(np.array(centroid_list))
        
        # Get off-diagonal elements for centering
        n = len(sim_matrix)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = sim_matrix[off_diag_mask]
        
        if len(off_diag) > 0:
            sim_mean = float(np.mean(off_diag))
            sim_std = float(np.std(off_diag))
            print(f"[Metrics DEBUG] Inter-archetype raw sims: mean={sim_mean:.4f}, std={sim_std:.4f}")
        else:
            sim_mean, sim_std = 0.5, 0.1
        
        # Sigmoid contrast: center on mean, scale by std, apply sigmoid
        # k controls steepness (higher = more contrast)
        k = 4.0  # Moderate contrast
        
        for i, name_i in enumerate(names_with_centroids):
            inter_similarity[name_i] = {}
            for j, name_j in enumerate(names_with_centroids):
                raw_sim = float(sim_matrix[i, j])
                if i == j:
                    # Diagonal stays 1.0
                    scaled_sim = 1.0
                else:
                    # Z-score then sigmoid: maps to (0, 1) with soft transitions
                    z = (raw_sim - sim_mean) / max(sim_std, 0.01)
                    sigmoid_val = 1.0 / (1.0 + np.exp(-k * z))
                    # Blend with raw to keep some absolute meaning (70% sigmoid, 30% raw)
                    scaled_sim = float(0.7 * sigmoid_val + 0.3 * raw_sim)
                inter_similarity[name_i][name_j] = round(float(scaled_sim), 4)
    
    # ====================================================================
    # STEP 5: Compute 3D coordinates for visualization using PCA
    # ====================================================================
    embedding_coords = []
    if len(D) >= 3:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        coords_3d = pca.fit_transform(D)
        
        # Normalize to [-1, 1] range for Three.js
        for dim in range(3):
            c_min, c_max = coords_3d[:, dim].min(), coords_3d[:, dim].max()
            if c_max - c_min > 1e-9:
                coords_3d[:, dim] = 2 * (coords_3d[:, dim] - c_min) / (c_max - c_min) - 1
        
        # Build list of {archetype, descriptor, x, y, z}
        for i, desc in enumerate(all_descriptors):
            arch = descriptor_to_archetype.get(i, "unknown")
            embedding_coords.append({
                "archetype": arch,
                "descriptor": desc,
                "x": round(float(coords_3d[i, 0]), 4),
                "y": round(float(coords_3d[i, 1]), 4),
                "z": round(float(coords_3d[i, 2]), 4)
            })
    
    # ====================================================================
    # STEP 6: Corpus-level aggregates
    # ====================================================================
    if all_dispersion:
        mean_disp = float(np.mean(all_dispersion))
        mean_leak = float(np.mean(all_leakage))
        mean_ent = float(np.mean(all_entropy))
        std_disp = float(np.std(all_dispersion))
        std_leak = float(np.std(all_leakage))
        std_ent = float(np.std(all_entropy))
        
        # Derived quality metrics
        coherence = 1.0 - mean_disp
        separation = 1.0 - mean_leak
        focus = 1.0 - mean_ent
        balance = float(1.0 - np.mean([std_disp, std_leak, std_ent]))
        quality = float(3.0 / (1.0/(coherence+0.01) + 1.0/(separation+0.01) + 1.0/(focus+0.01)))
    else:
        mean_disp = mean_leak = mean_ent = 0
        std_disp = std_leak = std_ent = 0
        coherence = separation = focus = balance = quality = 0
    
    corpus_metrics = {
        "mean_dispersion": round(mean_disp, 4),
        "mean_leakage": round(mean_leak, 4),
        "mean_entropy": round(mean_ent, 4),
        "std_dispersion": round(std_disp, 4),
        "std_leakage": round(std_leak, 4),
        "std_entropy": round(std_ent, 4),
        "coherence": round(coherence, 4),
        "separation": round(separation, 4),
        "focus": round(focus, 4),
        "balance": round(balance, 4),
        "quality": round(quality, 4)
    }
    
    return {
        "per_archetype": per_archetype,
        "corpus": corpus_metrics,
        "inter_similarity": inter_similarity,
        "archetype_order": archetype_order,
        "embedding_coords": embedding_coords  # 3D coordinates for visualization
    }


# ============================================================
# Mining Pipeline
# ============================================================

async def start_mining_job(session: MinerState, config: Dict):
    """Start a new mining job with real-time progress updates."""
    
    job_id = str(uuid.uuid4())[:8]
    job = MiningJob(
        id=job_id,
        status="pending",
        created_at=datetime.now().isoformat(),
        config=config
    )
    session.jobs[job_id] = job
    session.current_job = job_id
    
    await session.broadcast({
        "type": "job_created",
        "job": asdict(job)
    })
    
    # Run mining in background
    asyncio.create_task(run_mining_pipeline(session, job_id, config))


async def run_mining_pipeline(session: MinerState, job_id: str, config: Dict):
    """Execute the full mining pipeline with real-time updates."""
    job = session.jobs[job_id]
    job.status = "running"
    
    async def update_progress(stage: str, progress: float, detail: str = ""):
        job.stage = stage
        job.progress = progress
        log_entry = session.log(job_id, f"{stage}: {detail}" if detail else stage)
        await session.broadcast({
            "type": "progress",
            "job_id": job_id,
            "stage": stage,
            "progress": progress,
            "detail": detail,
            "log": log_entry
        })
        await asyncio.sleep(0.01)
    
    try:
        await update_progress("Initializing", 0.0, "Loading modules...")
        
        # Lazy load
        from enhanced_miner import (
            EnhancedArchetypeMiner, LLMArchetypeRefiner,
            TextChunk, MinerCorpus
        )
        from text_embedder import TextEmbedder
        
        # Initialize embedder (handle cloudflare variants)
        embedder_type = config.get("embedder", "cloudflare")
        if session.embedder is None or session._current_embedder != embedder_type:
            await update_progress("Loading embedder", 0.05, embedder_type)
            
            # Map UI values to backend + model
            if embedder_type == "cloudflare":
                session.embedder = TextEmbedder(backend="cloudflare", model="@cf/baai/bge-base-en-v1.5")
            elif embedder_type == "cloudflare-large":
                session.embedder = TextEmbedder(backend="cloudflare", model="@cf/baai/bge-large-en-v1.5")
            elif embedder_type == "cloudflare-qwen3":
                session.embedder = TextEmbedder(backend="cloudflare", model="@cf/qwen/qwen3-embedding-0.6b")
            else:
                session.embedder = TextEmbedder(backend=embedder_type)
            
            session._current_embedder = embedder_type
        
        await update_progress("Creating miner", 0.1)
        
        # Build LLM refiner
        llm_refiner = None
        if config.get("use_llm", True):
            llm_refiner = LLMArchetypeRefiner(
                backend="cloudflare",
                model=config.get("llm_model", "@cf/meta/llama-3.1-8b-instruct")
            )
        
        # Create miner
        miner = EnhancedArchetypeMiner(
            text_encoder=session.embedder,
            image_encoder=session.image_encoder,
            llm_refiner=llm_refiner
        )
        
        # Add text chunks
        await update_progress("Loading corpus", 0.15, f"{len(session.text_chunks)} chunks")
        
        for i, chunk in enumerate(session.text_chunks):
            miner.add_text(chunk["text"], source=chunk.get("source", "unknown"), chunk=False)
            if i % 50 == 0 and i > 0:
                await update_progress("Loading corpus", 0.15 + 0.05 * (i / len(session.text_chunks)), 
                                     f"Chunk {i}/{len(session.text_chunks)}")
        
        # Add images
        if session.images:
            await update_progress("Loading images", 0.2, f"{len(session.images)} images")
            images_with_descriptions = 0
            for img in session.images:
                miner.corpus.images.append(type('ImageItem', (), {
                    'id': img.id,
                    'path': img.path,
                    'filename': img.name,
                    'source_folder': img.folder,
                    'description': img.description,
                    'meta': {}
                })())
                
                # If image has a vision LLM description, add it as a text chunk for embedding
                if img.description:
                    miner.add_text(
                        img.description, 
                        source=f"image:{img.name}", 
                        chunk=False
                    )
                    images_with_descriptions += 1
                else:
                    # Fallback: use filename (without extension) as minimal text
                    name_without_ext = img.name.rsplit('.', 1)[0] if '.' in img.name else img.name
                    # Convert underscores/hyphens to spaces for better embedding
                    fallback_text = name_without_ext.replace('_', ' ').replace('-', ' ')
                    if fallback_text.strip():
                        miner.add_text(
                            fallback_text,
                            source=f"image:{img.name}",
                            chunk=False
                        )
            
            if images_with_descriptions > 0:
                await update_progress("Loading images", 0.22, 
                    f"{len(session.images)} images ({images_with_descriptions} with descriptions)")
            else:
                await update_progress("Loading images", 0.22,
                    f"{len(session.images)} images (using filenames - enable Vision LLM for better results)")
        
        # Embedding
        await update_progress("Embedding", 0.25, "Computing embeddings...")
        
        loop = asyncio.get_event_loop()
        ids, embeddings, types = await loop.run_in_executor(None, miner._embed_corpus)
        
        await update_progress("Embedding complete", 0.5, f"{len(ids)} items")
        
        if len(ids) == 0:
            raise ValueError("No items to cluster")
        
        # Build graph
        await update_progress("Building graph", 0.55, f"k={config.get('k_neighbors', 15)}")
        
        G = await loop.run_in_executor(
            None, 
            lambda: miner._build_similarity_graph(ids, embeddings, k=config.get("k_neighbors", 15))
        )
        
        await update_progress("Graph built", 0.65, f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Get mode and filters
        mode = config.get("mode", "exploratory")  # "exploratory" or "directional"
        filters = config.get("filters", [])  # e.g. ["North", "South", "East", "West"]
        
        archetypes = {}
        cluster_to_archetype_name = {}
        
        # =========================================================
        # BRANCH 1: DIRECTIONAL / FILTER MODE
        # =========================================================
        if mode == "directional" and filters:
            await update_progress("Directional Mining", 0.7, f"Applying filters: {', '.join(filters)}")
            
            # 1. Cluster (Common Step)
            clusters = await loop.run_in_executor(
                None,
                lambda: miner._cluster_graph(
                    G, 
                    min_size=config.get("min_cluster_size", 3),
                    resolution=config.get("resolution", 1.0)
                )
            )
            
            # 2. Extract Concept Soup
            await update_progress("Extracting Concept Soup", 0.75)
            all_concepts = []
            for cid, members in clusters.items():
                c_concepts = miner._extract_cluster_concepts(members, top_k=30)
                all_concepts.extend(c_concepts)
            
            # 3. LLM Mapping
            if config.get("use_llm", True) and llm_refiner:
                await update_progress("LLM Mapping", 0.85, f"Synthesizing into {len(filters)} filters...")
                archetypes = await loop.run_in_executor(
                    None,
                    lambda: llm_refiner.refine_directional(all_concepts, filters)
                )
            else:
                raise ValueError("LLM required for Directional Mode")
                
            # For visualization, we map specific clusters to filters based on overlap
            # (This is an approximation for the UI graph coloring)
            for cid in clusters.keys():
                # Assign round-robin or random for viz purposes since 
                # directional archetypes don't map 1:1 to structural clusters
                import random
                cluster_to_archetype_name[cid] = random.choice(list(archetypes.keys()))

        # =========================================================
        # BRANCH 2: STANDARD EXPLORATORY MODE (Existing Logic)
        # =========================================================
        else:
            # Clustering
            await update_progress("Clustering", 0.7, f"resolution={config.get('resolution', 1.0)}")
            clusters = await loop.run_in_executor(
                None,
                lambda: miner._cluster_graph(
                    G, 
                    min_size=config.get("min_cluster_size", 3),
                    resolution=config.get("resolution", 1.0)
                )
            )
            
            if not clusters: raise ValueError("No clusters found.")

            # Extract concepts
            await update_progress("Extracting concepts", 0.8)
            cluster_concepts = {}
            for i, (cid, members) in enumerate(clusters.items()):
                concepts = miner._extract_cluster_concepts(members, top_k=30)
                cluster_concepts[cid] = concepts if concepts else [f"cluster_{cid}"]

            # LLM refinement
            if config.get("use_llm", True) and llm_refiner:
                await update_progress("LLM Pass", 0.85, f"Refining {len(cluster_concepts)} archetypes...")
                semantic_spread = config.get("semantic_spread", 0.5)
                archetypes = await loop.run_in_executor(
                    None,
                    lambda: llm_refiner.refine_clusters(cluster_concepts, semantic_spread=semantic_spread)
                )
                
                # Map names
                archetype_names = list(archetypes.keys())
                for i, cid in enumerate(clusters.keys()):
                    if i < len(archetype_names):
                        cluster_to_archetype_name[cid] = archetype_names[i]
                    else:
                        cluster_to_archetype_name[cid] = f"CLUSTER_{cid}"
            else:
                archetypes = {}
                for cid, concepts in cluster_concepts.items():
                    name = concepts[0].upper()
                    archetypes[name] = concepts[:15]
                    cluster_to_archetype_name[cid] = name
        
        # Build graph data for visualization
        # Create a simplified graph structure with cluster assignments
        graph_data = {"nodes": [], "edges": []}
        
        # Map node IDs to cluster/archetype names
        node_to_cluster = {}
        for cid, members in clusters.items():
            cluster_name = cluster_to_archetype_name.get(cid, f"CLUSTER_{cid}")
            for member in members:
                node_to_cluster[member] = cluster_name
        
        # Add nodes with cluster assignment
        for node_id in G.nodes():
            graph_data["nodes"].append({
                "id": node_id,
                "cluster": node_to_cluster.get(node_id, "unknown")
            })
        
        # Sample edges (limit to prevent huge payloads)
        edges_list = list(G.edges())
        max_edges = min(500, len(edges_list))
        import random
        sampled_edges = random.sample(edges_list, max_edges) if len(edges_list) > max_edges else edges_list
        
        for u, v in sampled_edges:
            graph_data["edges"].append({"source": u, "target": v})
        
        # Compute ambiguity metrics for each archetype (using descriptor embeddings)
        metrics = compute_archetype_metrics(archetypes, session.embedder)
        
        # Done
        job.status = "completed"
        job.progress = 1.0
        job.result = {
            "archetypes": archetypes,
            "mode": mode,  # Return mode so UI knows
            "stats": {
                "total_items": len(ids),
                "clusters": len(clusters),
                "archetypes": len(archetypes),
                "edges": G.number_of_edges() if G else 0
            },
            "graph": graph_data,
            "metrics": metrics
        }
        
        print(f"[Mining] Job completed. Archetypes: {list(archetypes.keys())}")
        print(f"[Mining] Broadcasting to {len(session.websockets)} websockets...")
        
        await session.broadcast({
            "type": "job_completed",
            "job_id": job_id,
            "result": job.result
        })
        
        print("[Mining] Broadcast complete.")
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        session.log(job_id, f"ERROR: {e}")
        session.log(job_id, traceback.format_exc())
        
        await session.broadcast({
            "type": "job_failed",
            "job_id": job_id,
            "error": str(e)
        })


# ============================================================
# REST Endpoints (Session-aware via X-Session-Id header)
# ============================================================

def get_session_from_request(request: Request) -> Optional[MinerState]:
    """Get session from request header or return None."""
    session_id = request.headers.get("X-Session-Id")
    if session_id:
        return session_manager.get(session_id)
    return None


@app.get("/api/corpus")
async def get_corpus(request: Request):
    session = get_session_from_request(request)
    if not session:
        return {"error": "No session. Connect via WebSocket first.", "pdfs": 0, "images": 0, "texts": 0, "chunks": 0}
    return session.corpus_summary

@app.post("/upload/pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...), chunking: str = Form("{}")):
    """Upload and process a single PDF file."""
    import tempfile
    import json
    
    session = get_session_from_request(request)
    if not session:
        return {"error": "No session. Connect via WebSocket first."}
    
    # Parse chunking config
    try:
        chunk_config = json.loads(chunking)
    except:
        chunk_config = {"method": "paragraph"}
    
    method = chunk_config.get("method", "paragraph")
    
    try:
        # Save to temp file
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract text
        try:
            from enhanced_miner import PDFExtractor, TextChunker
        except ImportError:
            os.unlink(tmp_path)
            return {"error": "PDF extraction not available"}
        
        pages = PDFExtractor.extract_text(tmp_path)
        os.unlink(tmp_path)  # Clean up
        
        pdf_chunks = 0
        for page_data in pages:
            text = page_data.get("text", "")
            if not text.strip():
                continue
            
            # Apply chunking with parameters
            if method == "paragraph":
                min_len = chunk_config.get("min_length", 50)
                max_len = chunk_config.get("max_length", 1000)
                chunks = TextChunker.chunk_paragraphs(text, min_length=min_len, max_length=max_len)
            elif method == "sentence":
                sent_per = chunk_config.get("sentences_per_chunk", 5)
                overlap = chunk_config.get("overlap_sentences", 1)
                chunks = TextChunker.chunk_sentences(text, sentences_per_chunk=sent_per, overlap_sentences=overlap)
            elif method == "sliding":
                window = chunk_config.get("window_size", 512)
                stride = chunk_config.get("stride", 256)
                chunks = TextChunker.chunk_sliding_window(text, window_size=window, stride=stride)
            elif method == "page":
                chunks = [text]
            else:
                chunks = TextChunker.chunk_paragraphs(text)
            
            base_idx = len(session.text_chunks)
            for i, chunk in enumerate(chunks):
                session.text_chunks.append({
                    "id": f"{file.filename}_p{page_data['page']}_{base_idx + i}",
                    "text": chunk,
                    "source": file.filename,
                    "page": page_data["page"]
                })
                pdf_chunks += 1
        
        pdf_item = PDFItem(
            id=f"pdf_{len(session.pdfs)}",
            path=file.filename,
            name=file.filename,
            pages=len(pages),
            chunks=pdf_chunks,
            raw_pages=pages  # Store raw pages for rechunking
        )
        session.pdfs.append(pdf_item)
        
        await session.broadcast({
            "type": "corpus_updated",
            **session.corpus_summary
        })
        
        return {"name": file.filename, "pages": len(pages), "chunks": pdf_chunks}
        
    except Exception as e:
        return {"error": str(e)}


@app.post("/upload/image")
async def upload_image(request: Request, file: UploadFile = File(...), describe: str = "false"):
    """Upload and process a single image file."""
    import tempfile
    import base64
    
    session = get_session_from_request(request)
    if not session:
        return {"error": "No session. Connect via WebSocket first."}
    
    # Debug: log raw form data
    form = await request.form()
    print(f"[Upload] RAW FORM KEYS: {list(form.keys())}", flush=True)
    
    # FormData sends booleans as strings, so we need to parse
    describe_val = form.get('describe', 'false')
    describe_method = form.get('describe_method', 'semantic')
    should_describe = str(describe_val).lower() in ("true", "1", "yes")
    print(f"[Upload] file={file.filename}, describe={should_describe}, method={describe_method}", flush=True)
    
    try:
        content = await file.read()
        
        # Save temporarily if we need to describe
        description = None
        if should_describe:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                if describe_method == "vision":
                    # Use Vision LLM (slow but detailed)
                    from enhanced_miner import VisionDescriber
                    print(f"[Upload] Using Vision LLM for {file.filename}...", flush=True)
                    describer = VisionDescriber()
                    description = describer.describe_image(tmp_path)
                else:
                    # Use Semantic matching (fast)
                    from enhanced_miner import SemanticWordMatcher
                    print(f"[Upload] Using Semantic CLIP matching for {file.filename}...", flush=True)
                    
                    # Use cached matcher if available, recreate if it seems broken
                    # Cache in static folder so it can be committed and deployed
                    cache_path = Path(__file__).parent / "static" / "word_embeddings.npy"
                    cache_path.parent.mkdir(exist_ok=True)
                    
                    if not hasattr(session, '_semantic_matcher') or session._semantic_matcher is None:
                        session._semantic_matcher = SemanticWordMatcher(cache_path=str(cache_path))
                    
                    try:
                        description = session._semantic_matcher.describe_image_with_llm(tmp_path)
                    except Exception as matcher_error:
                        print(f"[Upload] Matcher error, recreating: {matcher_error}", flush=True)
                        # Recreate matcher in case of stale state
                        session._semantic_matcher = SemanticWordMatcher(cache_path=str(cache_path))
                        description = session._semantic_matcher.describe_image_with_llm(tmp_path)
                
                print(f"[Upload] Got description: {description[:80] if description else 'None'}...", flush=True)
            except Exception as e:
                print(f"[Upload] Description error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                description = None
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        img_item = ImageItem(
            id=f"img_{len(session.images)}",
            path=file.filename,
            name=file.filename,
            folder="upload",
            description=description
        )
        session.images.append(img_item)
        
        # If described, add as text chunk too
        if description:
            session.text_chunks.append({
                "id": f"img_desc_{len(session.text_chunks)}",
                "text": description,
                "source": f"image:{file.filename}"
            })
        
        await session.broadcast({
            "type": "corpus_updated",
            **session.corpus_summary
        })
        
        return {"name": file.filename, "described": description is not None}
        
    except Exception as e:
        return {"error": str(e)}


@app.delete("/api/corpus")
async def clear_corpus(request: Request):
    session = get_session_from_request(request)
    if not session:
        return {"error": "No session"}
    session.clear()
    await session.broadcast({"type": "corpus_cleared"})
    return {"status": "cleared"}

@app.get("/api/jobs")
async def list_jobs(request: Request):
    session = get_session_from_request(request)
    if not session:
        return {}
    return {jid: asdict(job) for jid, job in session.jobs.items()}

@app.get("/api/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    session = get_session_from_request(request)
    if not session:
        raise HTTPException(status_code=404, detail="No session")
    if job_id not in session.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return asdict(session.jobs[job_id])

@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": session_manager.active_count}


@app.post("/api/export-to-delyrism")
async def export_to_delyrism(request: Request):
    """Export archetypes to a file that Delyrism can read."""
    try:
        data = await request.json()
        archetypes = data.get("archetypes", {})
        
        if not archetypes:
            return {"success": False, "error": "No archetypes provided"}
        
        # Save to .egregore_export.json in the delyrism folder
        export_path = Path(__file__).parent / ".egregore_export.json"
        export_path.write_text(json.dumps(archetypes, indent=2, ensure_ascii=False), encoding="utf-8")
        
        return {"success": True, "count": len(archetypes), "path": str(export_path)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/generate-vision")
async def generate_vision(request: Request):
    """Proxy to Google AI API for image generation with multiple model support."""
    try:
        import httpx
    except ImportError:
        return {"success": False, "error": "httpx not installed. Run: pip install httpx"}
    
    try:
        data = await request.json()
        api_key = data.get("apiKey")
        prompt = data.get("prompt")
        model = data.get("model", "gemini-2.5-flash-image")
        
        if not api_key:
            return {"success": False, "error": "API key required"}
        if not prompt:
            return {"success": False, "error": "Prompt required"}
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            
            # Imagen 4 uses predict method
            if model.startswith("imagen"):
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict?key={api_key}"
                payload = {
                    "instances": [{"prompt": prompt}],
                    "parameters": {
                        "sampleCount": 1,
                        "aspectRatio": "1:1",
                        "personGeneration": "DONT_ALLOW"
                    }
                }
                response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                    except:
                        error_msg = f"API error {response.status_code}: {response.text[:300]}"
                    return {"success": False, "error": error_msg}
                
                result = response.json()
                predictions = result.get("predictions", [])
                if not predictions:
                    return {"success": False, "error": "No image generated by Imagen"}
                
                image_base64 = predictions[0].get("bytesBase64Encoded")
                mime_type = predictions[0].get("mimeType", "image/png")
                
                if not image_base64:
                    return {"success": False, "error": "No image data in Imagen response"}
                
                return {"success": True, "imageData": f"data:{mime_type};base64,{image_base64}"}
            
            # Gemini and Nano Banana use generateContent method
            else:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {
                    "contents": [{
                        "parts": [{"text": f"Generate an image: {prompt}"}]
                    }],
                    "generationConfig": {
                        "responseModalities": ["TEXT", "IMAGE"]
                    }
                }
                response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                    except:
                        error_msg = f"API error {response.status_code}: {response.text[:300]}"
                    return {"success": False, "error": error_msg}
                
                result = response.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    return {"success": False, "error": "No response generated"}
                
                parts = candidates[0].get("content", {}).get("parts", [])
                
                # Find the image part
                image_part = None
                for part in parts:
                    if "inlineData" in part:
                        image_part = part["inlineData"]
                        break
                
                if not image_part:
                    text_response = " ".join([p.get("text", "") for p in parts if "text" in p])
                    return {"success": False, "error": f"No image generated. Response: {text_response[:200]}"}
                
                image_base64 = image_part.get("data")
                mime_type = image_part.get("mimeType", "image/png")
                
                if not image_base64:
                    return {"success": False, "error": "No image data in response"}
                
                return {"success": True, "imageData": f"data:{mime_type};base64,{image_base64}"}
            
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out (120s). Try again."}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


# ============================================================
# Frontend
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    from starlette.responses import Response
    html_path = Path(__file__).parent / "miner_ui.html"
    content = html_path.read_text(encoding='utf-8')
    return Response(
        content=content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    # Use PORT from environment (Railway) or default to 8765 (local)
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run(app, host="0.0.0.0", port=port)
