# enhanced_miner.py — Advanced Archetype Mining with PDF, Folders, and LLM Refinement
"""
Enhanced corpus mining with:
  • Folder scanning for images (recursive)
  • PDF text extraction with smart chunking strategies
  • LLM-powered archetype derivation (Cloudflare or local)
  • Embedding clustering visualization
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Literal
from pathlib import Path
import numpy as np
import re
import json
import hashlib
from collections import Counter, defaultdict

# Optional imports with graceful fallbacks
try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# ============================================================
# Data structures
# ============================================================

@dataclass
class TextChunk:
    """A chunk of text with provenance metadata."""
    id: str
    text: str
    source: str  # filename or URL
    page: Optional[int] = None
    chunk_idx: int = 0
    strategy: str = "paragraph"  # paragraph, sentence, sliding, semantic
    meta: Dict = field(default_factory=dict)


@dataclass
class ImageItem:
    """An image with metadata."""
    id: str
    path: str
    source_folder: str
    filename: str
    description: Optional[str] = None  # LLM-generated description
    meta: Dict = field(default_factory=dict)


@dataclass 
class MinerCorpus:
    """Collection of text chunks and images for mining."""
    text_chunks: List[TextChunk] = field(default_factory=list)
    images: List[ImageItem] = field(default_factory=list)
    
    def __len__(self):
        return len(self.text_chunks) + len(self.images)
    
    def summary(self) -> Dict:
        return {
            "text_chunks": len(self.text_chunks),
            "images": len(self.images),
            "sources": list(set(c.source for c in self.text_chunks)),
        }


# ============================================================
# PDF Extraction
# ============================================================

class PDFExtractor:
    """Extract text from PDFs with multiple backend support."""
    
    @staticmethod
    def extract_text(pdf_path: str | Path, backend: str = "auto") -> List[Dict]:
        """
        Extract text from PDF, returning list of {page, text} dicts.
        
        Args:
            pdf_path: Path to PDF file
            backend: "pymupdf", "pdfplumber", or "auto"
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Auto-select backend
        if backend == "auto":
            if fitz is not None:
                backend = "pymupdf"
            elif pdfplumber is not None:
                backend = "pdfplumber"
            else:
                raise ImportError(
                    "No PDF backend available. Install: pip install pymupdf or pip install pdfplumber"
                )
        
        pages = []
        
        if backend == "pymupdf" and fitz is not None:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
            doc.close()
            
        elif backend == "pdfplumber" and pdfplumber is not None:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append({"page": i + 1, "text": text})
        else:
            raise ValueError(f"Backend '{backend}' not available")
        
        return pages


# ============================================================
# Text Chunking Strategies
# ============================================================

class TextChunker:
    """Smart text chunking with multiple strategies."""
    
    @staticmethod
    def chunk_paragraphs(
        text: str,
        min_length: int = 50,
        max_length: int = 1000,
        overlap: int = 0
    ) -> List[str]:
        """Split by double newlines (paragraphs), merge small ones."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Merge small paragraphs
        merged = []
        buffer = ""
        for p in paragraphs:
            if len(buffer) + len(p) < max_length:
                buffer = (buffer + "\n\n" + p).strip() if buffer else p
            else:
                if buffer and len(buffer) >= min_length:
                    merged.append(buffer)
                buffer = p
        if buffer and len(buffer) >= min_length:
            merged.append(buffer)
        
        return merged
    
    @staticmethod
    def chunk_sentences(
        text: str,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
        min_length: int = 50
    ) -> List[str]:
        """Split by sentences, group into chunks with optional overlap."""
        # Simple sentence splitting (handles common cases)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text] if len(text) >= min_length else []
        
        chunks = []
        i = 0
        while i < len(sentences):
            chunk_sents = sentences[i:i + sentences_per_chunk]
            chunk = " ".join(chunk_sents)
            if len(chunk) >= min_length:
                chunks.append(chunk)
            i += sentences_per_chunk - overlap_sentences
            if i <= 0:
                i = sentences_per_chunk  # prevent infinite loop
        
        return chunks
    
    @staticmethod
    def chunk_sliding_window(
        text: str,
        window_size: int = 512,
        stride: int = 256,
        min_length: int = 50
    ) -> List[str]:
        """Fixed-size sliding window (character-based)."""
        text = text.strip()
        if len(text) < min_length:
            return []
        
        chunks = []
        for start in range(0, len(text), stride):
            chunk = text[start:start + window_size].strip()
            if len(chunk) >= min_length:
                chunks.append(chunk)
            if start + window_size >= len(text):
                break
        
        return chunks
    
    @staticmethod
    def chunk_semantic(
        text: str,
        embedder: Callable[[List[str]], np.ndarray],
        similarity_threshold: float = 0.7,
        min_length: int = 50,
        max_length: int = 1500
    ) -> List[str]:
        """
        Semantic chunking: merge consecutive sentences while similar,
        break when similarity drops below threshold.
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) >= 20]
        
        if len(sentences) < 2:
            return [text] if len(text) >= min_length else []
        
        # Embed all sentences
        try:
            embs = embedder(sentences)
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        except Exception:
            # Fallback to paragraph chunking if embedding fails
            return TextChunker.chunk_paragraphs(text, min_length=min_length, max_length=max_length)
        
        chunks = []
        buffer = [sentences[0]]
        buffer_emb = embs[0]
        
        for i in range(1, len(sentences)):
            sim = float(np.dot(buffer_emb, embs[i]))
            combined_len = sum(len(s) for s in buffer) + len(sentences[i])
            
            if sim >= similarity_threshold and combined_len < max_length:
                buffer.append(sentences[i])
                # Update buffer embedding (running average)
                buffer_emb = (buffer_emb * (len(buffer) - 1) + embs[i]) / len(buffer)
                buffer_emb = buffer_emb / (np.linalg.norm(buffer_emb) + 1e-8)
            else:
                chunk = " ".join(buffer)
                if len(chunk) >= min_length:
                    chunks.append(chunk)
                buffer = [sentences[i]]
                buffer_emb = embs[i]
        
        # Final buffer
        chunk = " ".join(buffer)
        if len(chunk) >= min_length:
            chunks.append(chunk)
        
        return chunks


# ============================================================
# Folder Scanning
# ============================================================

class FolderScanner:
    """Scan folders for images and text files."""
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
    TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".json"}
    
    @classmethod
    def scan_images(
        cls,
        folder: str | Path,
        recursive: bool = True,
        max_files: int = 1000
    ) -> List[ImageItem]:
        """Scan folder for image files."""
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        pattern = "**/*" if recursive else "*"
        images = []
        
        for path in folder.glob(pattern):
            if path.suffix.lower() in cls.IMAGE_EXTENSIONS and path.is_file():
                item = ImageItem(
                    id=hashlib.md5(str(path).encode()).hexdigest()[:12],
                    path=str(path),
                    source_folder=str(folder),
                    filename=path.name,
                    meta={"relative_path": str(path.relative_to(folder))}
                )
                images.append(item)
                
                if len(images) >= max_files:
                    break
        
        return images
    
    @classmethod
    def scan_pdfs(cls, folder: str | Path, recursive: bool = True) -> List[Path]:
        """Find all PDFs in a folder."""
        folder = Path(folder)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return list(folder.glob(pattern))


# ============================================================
# Vision LLM Image Describer
# ============================================================

IMAGE_DESCRIPTION_PROMPT = """Describe this image in rich detail for semantic analysis. Focus on:
1. Main subjects and objects (people, animals, things)
2. Actions, movements, or states
3. Setting, environment, atmosphere
4. Colors, textures, patterns
5. Emotional tone or mood
6. Symbolic or archetypal elements (if any)

Be specific and evocative. Use concrete nouns and vivid adjectives.
Write 2-4 sentences. Do not start with "This image shows" or similar."""


class VisionDescriber:
    """Use Cloudflare's vision LLM to generate rich descriptions of images."""
    
    # Cloudflare vision-capable models
    VISION_MODELS = {
        "llava": "@cf/llava-hf/llava-1.5-7b-hf",
        "llava-next": "@cf/llava-hf/llava-v1.6-mistral-7b-instruct",  # Better quality
    }
    
    def __init__(
        self,
        model: str = "llava",
        cloudflare_account_id: Optional[str] = None,
        cloudflare_api_token: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.model = self.VISION_MODELS.get(model, model)
        self.cf_account = cloudflare_account_id
        self.cf_token = cloudflare_api_token
        self.prompt = custom_prompt or IMAGE_DESCRIPTION_PROMPT
        self._cache: Dict[str, str] = {}  # path -> description cache
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string."""
        import base64
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read and optionally resize large images
        if Image is not None:
            img = Image.open(path)
            # Resize if too large (vision models have limits)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed (handle RGBA, P mode, etc.)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Save to bytes
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
        else:
            # Fallback: read raw bytes
            image_bytes = path.read_bytes()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def describe_image(self, image_path: str, use_cache: bool = True) -> str:
        """
        Generate a text description for an image using vision LLM.
        
        Args:
            image_path: Path to the image file
            use_cache: Whether to use cached descriptions
            
        Returns:
            Text description of the image
        """
        import requests
        import os
        
        # Check cache
        if use_cache and image_path in self._cache:
            return self._cache[image_path]
        
        account_id = self.cf_account or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        api_token = self.cf_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        
        if not account_id or not api_token:
            raise ValueError(
                "Cloudflare credentials not configured. "
                "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN environment variables."
            )
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image_path)
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model}"
        
        # Cloudflare vision API format
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt
                }
            ],
            "image": image_b64,
            "max_tokens": 256,
            "temperature": 0.7,
        }
        
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("success"):
                errors = data.get("errors", [])
                raise RuntimeError(f"Cloudflare API error: {errors}")
            
            description = data.get("result", {}).get("response", "").strip()
            
            # Cache the result
            if description:
                self._cache[image_path] = description
            
            return description
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Vision API timeout. Try again or use a smaller image.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Vision API request failed: {e}")
    
    def describe_batch(
        self,
        images: List,  # List of ImageItem or paths
        progress_callback: Optional[Callable[[int, int], None]] = None,
        skip_on_error: bool = True,
        max_workers: int = 5,  # Concurrent API calls
    ) -> Dict[str, str]:
        """
        Describe multiple images using concurrent API calls.
        
        Args:
            images: List of ImageItem objects or image file paths
            progress_callback: Optional callback(current, total) for progress updates
            skip_on_error: If True, continue on errors; if False, raise
            max_workers: Number of concurrent API requests (don't exceed rate limits)
            
        Returns:
            Dict mapping image_path -> description
        """
        import concurrent.futures
        import threading
        
        # Extract paths from ImageItem if needed
        image_paths = []
        for img in images:
            if hasattr(img, 'path'):
                image_paths.append(img.path)
            else:
                image_paths.append(str(img))
        
        results = {}
        completed = [0]  # Use list for mutable counter in closure
        lock = threading.Lock()
        
        def describe_one(path: str) -> tuple:
            """Describe a single image, return (path, description or None)."""
            try:
                desc = self.describe_image(path)
                return (path, desc)
            except Exception as e:
                if not skip_on_error:
                    raise
                return (path, None)
        
        # Use ThreadPoolExecutor for concurrent API calls
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(describe_one, path): path 
                for path in image_paths
            }
            
            # Process as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path, desc = future.result()
                
                with lock:
                    completed[0] += 1
                    if desc:
                        results[path] = desc
                    
                    if progress_callback:
                        progress_callback(completed[0], len(image_paths))
        
        return results
    
    def clear_cache(self):
        """Clear the description cache."""
        self._cache.clear()


# ============================================================
# LLM Archetype Refiner
# ============================================================

ARCHETYPE_SYSTEM_PROMPT = """You are a poet-scholar of symbolic systems, archetypes, and mythological patterns.

Your task: transform raw concept clusters into archetypes with DIVERSE, UNUSUAL, EVOCATIVE descriptors.

ABSOLUTE RULES:
1. NO PROPER NOUNS: Never use names of people, places, brands, religions, cultures
2. ZERO REDUNDANCY: A word may appear in ONLY ONE archetype across the entire output
3. NO SEMANTIC OVERLAP within an archetype: avoid synonyms or closely related words
   BAD: "letters, alphabet, vowels" (all about writing symbols)
   GOOD: "inscription, echo, chronicle" (diverse aspects of written tradition)
4. PREFER UNUSUAL VOCABULARY: avoid common words like "experience", "place", "things", "nature"
   Instead use: "phenomenology", "locus", "artifacts", "wilderness"
5. SEMANTIC BREADTH: each archetype's descriptors should span different domains
   Include: objects, actions, qualities, metaphors, textures, sounds, movements

For each cluster, generate:
- NAME: evocative, 1-3 words
- DESCRIPTORS: 10-12 terms with MAXIMUM semantic diversity (no synonyms, no overlap)
- ESSENCE: one poetic sentence

Output ONLY valid JSON."""

ARCHETYPE_USER_TEMPLATE = """Transform these {n_clusters} concept clusters into archetypes.

SOURCE CONCEPTS (use as inspiration, do NOT copy directly):
{clusters_text}

STRICT REQUIREMENTS:
1. ZERO WORD REPETITION across all archetypes. If "breath" is in Archetype 1, it cannot appear anywhere else.
2. NO SEMANTIC CLUSTERS within descriptors:
   - WRONG: "oral, spoken, voice, speech" (all about speaking)
   - RIGHT: "utterance, resonance, larynx, aria" (diverse aspects)
3. AVOID COMMON/GENERIC WORDS
   Replace with unusual alternatives.
4. MAXIMUM DIVERSITY per archetype - descriptors should feel like they come from different domains:
   Some example of domains, but not limited to:
   - A body part 
   - An action/verb 
   - A texture/quality 
   - A metaphor/image 
   - An unusual noun

QUALITY CHECK before output:
- Count: does any word appear twice? → REJECT and revise
- Similarity: are any two descriptors synonyms? → replace one with something from a different domain

Output valid JSON:
{{
  "archetypes": [
    {{
      "name": "ARCHETYPE_NAME",
      "descriptors": ["term1", "term2", ...],
      "essence": "One poetic sentence"
    }}
  ]
}}

JSON only."""


class LLMArchetypeRefiner:
    """Use LLM to refine and name discovered clusters into proper archetypes."""
    
    def __init__(
        self,
        backend: Literal["cloudflare", "local"] = "cloudflare",
        model: str = "@cf/meta/llama-3.1-8b-instruct",
        cloudflare_account_id: Optional[str] = None,
        cloudflare_api_token: Optional[str] = None,
        local_model_loader: Optional[Callable] = None,
    ):
        self.backend = backend
        self.model = model
        self.cf_account = cloudflare_account_id
        self.cf_token = cloudflare_api_token
        self.local_loader = local_model_loader
    
    def _call_cloudflare(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Call Cloudflare Workers AI."""
        import requests
        import os
        
        account_id = self.cf_account or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        api_token = self.cf_token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        
        if not account_id or not api_token:
            raise ValueError("Cloudflare credentials not configured")
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{self.model}"
        
        response = requests.post(
            url,
            json={"messages": messages, "max_tokens": max_tokens, "temperature": 0.7},
            headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("success"):
            raise RuntimeError(f"Cloudflare API error: {data.get('errors', [])}")
        
        return data.get("result", {}).get("response", "").strip()
    
    def _call_local(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        """Call local model (requires loader function)."""
        if self.local_loader is None:
            raise ValueError("Local model loader not provided")
        
        tok, mdl = self.local_loader(self.model)
        # Simplified generation (assumes compatible interface)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        
        import torch
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
        
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    def _generate_single_archetype(self, cluster_id: int, concepts: List[str], existing_words: set, semantic_spread: float = 0.5) -> Dict:
        """Generate descriptors for a single cluster, avoiding already-used words.
        
        semantic_spread: 0=focused, 1=divergent
        """
        
        avoid_list = ", ".join(sorted(existing_words)[:50]) if existing_words else "none yet"
        
        # Build spread-dependent instructions
        if semantic_spread < 0.3:
            # FOCUSED: Simple name, coherent descriptors, grounded in source
            spread_instruction = """STYLE: GROUNDED & ACCESSIBLE
NAME FORMAT: 1 word preferred, 2 words maximum
- Good names: "Flame", "Shadow", "Roots", "The Vessel", "Breath"
- Bad names: "The Cartographer of Dreams", "Silken Weaver of Time" (too long)
- Bad names: "Luminari", "Terraephyra" (invented/cryptic words)

DESCRIPTORS: Everyday words, tightly connected to source concepts
- All descriptors should CLEARLY relate to the theme found in source concepts
- Stay grounded - every word should feel like it belongs to the same semantic field
- Example for fire concepts: "FLAME" → warmth, burning, light, smoke, spark, heat, glow, ember, torch, blaze"""
        elif semantic_spread < 0.7:
            # BALANCED: Simple name, but semantically rich descriptors
            spread_instruction = """STYLE: ACCESSIBLE NAME, RICH DESCRIPTORS
NAME FORMAT: 1 word preferred, 2 words maximum  
- Good names: "Threshold", "Marrow", "The Hollow", "Residue", "Patina"
- Bad names: "The Keeper of Forgotten Whispers" (too long)
- Bad names: "The Storyteller", "The Mapmaker" (generic role-names)
- AVOID: Keeper, Walker, Weaver, Maker, Teller, Bearer

DESCRIPTORS: Unusual vocabulary but ALL CONNECTED to source concepts
- Use rare/evocative words: gossamer, patina, sinew, cadence, liminal, chthonic
- Even distant descriptors must be JUSTIFIED by something in the source text
- Create unexpected connections WITHIN the source material's themes
- AVOID generic words: feel, land, air, earth, story, growth, nature
- Example for fire concepts: "CRUCIBLE" → incandescence, transmutation, oxidation, vitreous, phosphor, kiln, temper, slag, anneal, forge-breath"""
        else:
            # DIVERGENT: Cryptic name, surrealist descriptors
            spread_instruction = """STYLE: SURREALIST & MYTHOPOIETIC
NAME FORMAT: 1-2 words, etymologically rich or invented
- Good names: "Pyrrhic", "Velamen", "Ossuary", "The Liminal"

DESCRIPTORS: Transform basic concepts into SPECIFIC, UNUSUAL vocabulary.
The source concepts are just THEMES - never use them directly as descriptors.
Instead, find rare/technical/poetic words that EVOKE those themes obliquely.

TRANSFORMATION EXAMPLES (source concept → good descriptors):
- "time, past, future" → chronometry, patina, palimpsest, vestige, harbinger, sediment
- "life, death, living" → pulse, carrion, germination, ossify, quickening, dormancy  
- "earth, ground, soil" → loam, stratum, rhizome, telluric, alluvial, subterranean
- "light, dark, shadow" → penumbra, phosphene, chiaroscuro, crepuscular, umbral
- "water, flow, liquid" → meniscus, turbidity, ablution, rivulet, briny, vitreous

Each descriptor should come from a DIFFERENT domain: body part, texture, sound, movement, material, tool, scientific term, archaic word.

FORBIDDEN SIMPLE WORDS

QUALITY CHECK before output - for EACH descriptor ask:
1. Would a 10-year-old know this word? If YES → replace with rarer word
2. Is it in the FORBIDDEN list above? If YES → replace
3. Does it have fewer than 6 letters? Consider replacing with more specific term

GOOD descriptors: patina, sinew, cartilage, amber, ferric, gossamer, tremor, resonance, membrane, filament, viscera, tincture, striation, pellicle, tessellation"""
        
        prompt = f"""Generate an archetype from these concepts.

SOURCE CONCEPTS (cluster {cluster_id}):
{', '.join(concepts[:40])}

CRITICAL: The source concepts above are just THEMES for inspiration.
Do NOT use common/abstract words as descriptors. Transform them into specific, evocative vocabulary.

WORDS ALREADY USED IN OTHER ARCHETYPES (DO NOT USE ANY OF THESE):
{avoid_list}

{spread_instruction}

RULES:
1. NO word from the "already used" list above
2. NO synonyms or variants (wind/winds, breath/breathing)
3. NO proper nouns (names, places, religions, cultures)
4. Name should be 1-2 words only
5. TRANSFORM basic concepts into unusual/specific vocabulary - never copy them directly
6. SINGLE-WORD descriptors only. Maximum 2 two-word phrases per archetype. 

Output JSON only:
{{
  "name": "ARCHETYPE NAME",
  "descriptors": ["term1", "term2", "term3", "term4", "term5", "term6", "term7", "term8", "term9", "term10"],
  "essence": "One poetic sentence"
}}"""

        messages = [
            {"role": "system", "content": "You are creating symbolic archetypes grounded in source material. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        if self.backend == "cloudflare":
            response = self._call_cloudflare(messages, max_tokens=512)
        else:
            response = self._call_local(messages, max_tokens=512)
        
        return self._parse_single_archetype(response)
    
    def _parse_single_archetype(self, response: str) -> Dict:
        """Parse a single archetype response."""
        response = response.strip()
        
        # Handle markdown
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
            return data
        except:
            return {}
    
    def _deduplicate_descriptors(self, archetypes: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Remove any duplicate/similar words across archetypes."""
        
        # Build word -> archetype mapping
        word_usage = {}
        for name, descriptors in archetypes.items():
            for desc in descriptors:
                word = desc.lower().strip()
                # Also track stems
                stem = word.rstrip('s').rstrip('ing').rstrip('ed')
                if word not in word_usage:
                    word_usage[word] = []
                word_usage[word].append(name)
                if stem != word and stem not in word_usage:
                    word_usage[stem] = []
                if stem != word:
                    word_usage[stem].append(name)
        
        # Find duplicates
        duplicates = {w: names for w, names in word_usage.items() if len(names) > 1}
        
        if not duplicates:
            return archetypes
        
        # Remove duplicates - keep in first archetype, remove from others
        result = {}
        seen_words = set()
        seen_stems = set()
        
        for name, descriptors in archetypes.items():
            clean_descriptors = []
            for desc in descriptors:
                word = desc.lower().strip()
                stem = word.rstrip('s').rstrip('ing').rstrip('ed')
                
                # Check if word or its stem was already used
                if word in seen_words or stem in seen_stems:
                    continue
                
                clean_descriptors.append(word)
                seen_words.add(word)
                seen_stems.add(stem)
            
            result[name] = clean_descriptors
        
        return result
    
    def _llm_refinement_pass(self, archetypes: Dict[str, List[str]], semantic_spread: float = 0.5) -> Dict[str, List[str]]:
        """
        Second LLM pass: refine all archetypes together to remove redundancy,
        balance descriptors, and ensure distinct archetype identities.
        """
        if len(archetypes) < 2:
            return archetypes
        
        # Build the archetypes summary for the LLM
        archetypes_text = "\n".join([
            f"- {name}: {', '.join(descriptors[:12])}"
            for name, descriptors in archetypes.items()
        ])
        
        # Style guidance based on semantic spread
        if semantic_spread < 0.3:
            style_note = "Names must be 1-2 simple words. Descriptors: common, coherent words."
        elif semantic_spread < 0.7:
            style_note = "Names must be 1-2 words. Descriptors: unusual/evocative but connected."
        else:
            style_note = "Names: 1-2 cryptic/etymological words. Descriptors: surrealist but with invisible thematic threads."
        
        prompt = f"""You are a critical editor refining archetypes. Your job is to ELIMINATE REDUNDANCY ruthlessly.

CURRENT ARCHETYPES:
{archetypes_text}

CRITICAL PROBLEMS TO FIX:

1. NAME LENGTH - names must be 1-2 words ONLY:
   - BAD: "The Cartographer of Dreams", "The Silken Weaver" (too long)
   - GOOD: "Threshold", "Marrow", "The Hollow", "Flame"

2. CONCEPTUAL OVERLAP in names - if multiple archetypes use similar metaphors, RENAME them:
   - BAD: "Cartographer" + "Mapmaker" + "Navigator" (all mapping)
   - BAD: "Weaver" + "Spinner" + "Thread" (all weaving)  
   - BAD: "Storyteller" + "Chronicler" + "Tale" (all narrative)
   - FIX: Give each a COMPLETELY DIFFERENT root metaphor
   
4. DESCRIPTOR DUPLICATES:
   - Remove same/similar words across archetypes
   - Check stems: earth/earthen, speak/spoken, grow/growth

STYLE: {style_note}

CRITICAL: You must output EXACTLY {len(archetypes)} archetypes. Do NOT merge or remove any.
Your job is to RENAME and REFINE, not to reduce the count.

RULES:
- Output exactly {len(archetypes)} archetypes (same count as input)
- Each archetype must have a UNIQUE root metaphor
- NO proper nouns
- Each descriptor in exactly ONE archetype
- 8-12 descriptors per archetype

Output valid JSON:
{{
  "archetypes": [
    {{"name": "ARCHETYPE NAME", "descriptors": ["word1", "word2", ...]}},
    ...
  ]
}}

JSON only."""

        messages = [
            {"role": "system", "content": f"You are refining {len(archetypes)} symbolic archetypes. You must output exactly {len(archetypes)} archetypes. Output valid JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if self.backend == "cloudflare":
                response = self._call_cloudflare(messages, max_tokens=2048)
            else:
                response = self._call_local(messages, max_tokens=2048)
            
            # Parse response
            parsed = self._parse_refinement_response(response)
            # Stricter check: must have at least 80% of original archetypes
            min_required = max(len(archetypes) - 1, int(len(archetypes) * 0.8))
            if parsed and len(parsed) >= min_required:
                return parsed
            else:
                print(f"Refinement pass returned insufficient archetypes, keeping originals")
                return archetypes
                
        except Exception as e:
            print(f"LLM refinement pass failed: {e}, keeping original archetypes")
            return archetypes
    
    def _parse_refinement_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the refinement LLM response."""
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to fix common issues
            response = re.sub(r',\s*}', '}', response)
            response = re.sub(r',\s*]', ']', response)
            try:
                data = json.loads(response)
            except:
                return {}
        
        # Extract archetypes
        result = {}
        archetypes_list = data.get("archetypes", [])
        
        for arch in archetypes_list:
            name = arch.get("name", "").strip().upper()
            descriptors = arch.get("descriptors", [])
            if name and descriptors:
                result[name] = [str(d).strip().lower() for d in descriptors if d]
        
        return result
    
    def refine_clusters(
        self,
        clusters: Dict[int, List[str]],  # cluster_id -> list of concept strings
        cluster_summaries: Optional[Dict[int, str]] = None,  # optional summaries
        semantic_spread: float = 0.5,  # 0=focused/coherent, 1=diverse/divergent
    ) -> Dict[str, List[str]]:
        """
        Take raw clusters and use LLM to derive proper archetypes.
        
        Pipeline:
        1. Per-cluster LLM calls to generate initial archetypes
        2. Programmatic deduplication
        3. Second LLM pass to refine all archetypes together
        
        semantic_spread: Controls diversity of descriptors within each archetype
          - 0.0: Focused/coherent - descriptors cluster around a tight semantic field
          - 0.5: Balanced - mix of related and divergent terms
          - 1.0: Diverse/divergent - descriptors span wildly different domains
        
        Returns: {archetype_name: [descriptors]}
        """
        archetypes = {}
        all_used_words = set()
        
        # PASS 1: Generate each archetype separately
        for cid, concepts in clusters.items():
            try:
                result = self._generate_single_archetype(cid, concepts, all_used_words, semantic_spread)
                
                if result:
                    name = result.get("name", f"CLUSTER_{cid}").strip().upper()
                    descriptors = [str(d).lower().strip() for d in result.get("descriptors", []) if d]
                    
                    if descriptors:
                        archetypes[name] = descriptors
                        # Track all words used
                        for d in descriptors:
                            all_used_words.add(d)
                            # Also add stems
                            stem = d.rstrip('s').rstrip('ing').rstrip('ed')
                            all_used_words.add(stem)
                    else:
                        raise ValueError("No descriptors found in LLM response")
                else:
                    raise ValueError("Empty result from LLM (JSON parse failed)")
            except Exception as e:
                print(f"Error generating archetype for cluster {cid}: {e}")
                # Fallback
                if concepts:
                    name = f"{concepts[0]} / {concepts[1]}".upper() if len(concepts) >= 2 else concepts[0].upper()
                    archetypes[name] = concepts[:10]
        
        # Programmatic deduplication
        archetypes = self._deduplicate_descriptors(archetypes)
        
        # PASS 2: LLM refinement of all archetypes together
        if len(archetypes) >= 2:
            archetypes = self._llm_refinement_pass(archetypes, semantic_spread)
        
        return archetypes
    
    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response, handling common formatting issues."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                response = match.group(1)
        
        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            response = response[start:end]
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # Try to fix common issues
            response = re.sub(r',\s*}', '}', response)  # trailing commas
            response = re.sub(r',\s*]', ']', response)
            try:
                data = json.loads(response)
            except:
                raise ValueError(f"Could not parse LLM response as JSON: {e}\nResponse: {response[:500]}")
        
        # Extract archetypes
        result = {}
        archetypes = data.get("archetypes", [])
        
        for arch in archetypes:
            name = arch.get("name", "").strip().upper()
            descriptors = arch.get("descriptors", [])
            if name and descriptors:
                result[name] = [str(d).strip().lower() for d in descriptors if d]
        
        return result


# ============================================================
# Enhanced Archetype Miner
# ============================================================

class EnhancedArchetypeMiner:
    """
    Full-featured archetype miner with:
    - PDF ingestion + smart chunking
    - Folder image scanning
    - Vision LLM image description
    - Embedding-based clustering
    - LLM-powered archetype refinement
    """
    
    def __init__(
        self,
        text_encoder: Callable[[List[str]], np.ndarray],
        image_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
        llm_refiner: Optional[LLMArchetypeRefiner] = None,
        vision_describer: Optional[VisionDescriber] = None,
    ):
        self.text_enc = text_encoder
        self.img_enc = image_encoder
        self.llm_refiner = llm_refiner
        self.vision_describer = vision_describer
        self.corpus = MinerCorpus()
    
    # ---- Ingestion Methods ----
    
    def add_pdf(
        self,
        pdf_path: str | Path,
        chunking: Literal["paragraph", "sentence", "sliding", "semantic"] = "paragraph",
        **chunk_kwargs
    ) -> int:
        """
        Add a PDF to the corpus.
        
        Returns: number of chunks added
        """
        pages = PDFExtractor.extract_text(pdf_path)
        pdf_name = Path(pdf_path).name
        
        added = 0
        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]
            
            # Apply chunking strategy
            if chunking == "paragraph":
                chunks = TextChunker.chunk_paragraphs(text, **chunk_kwargs)
            elif chunking == "sentence":
                chunks = TextChunker.chunk_sentences(text, **chunk_kwargs)
            elif chunking == "sliding":
                chunks = TextChunker.chunk_sliding_window(text, **chunk_kwargs)
            elif chunking == "semantic":
                chunks = TextChunker.chunk_semantic(
                    text, 
                    embedder=lambda t: self.text_enc.encode(t) if hasattr(self.text_enc, 'encode') else self.text_enc(t),
                    **chunk_kwargs
                )
            else:
                chunks = [text]
            
            for i, chunk_text in enumerate(chunks):
                chunk = TextChunk(
                    id=f"{pdf_name}_p{page_num}_c{i}",
                    text=chunk_text,
                    source=pdf_name,
                    page=page_num,
                    chunk_idx=i,
                    strategy=chunking
                )
                self.corpus.text_chunks.append(chunk)
                added += 1
        
        return added
    
    def add_folder_images(
        self,
        folder: str | Path,
        recursive: bool = True,
        max_files: int = 500,
        describe_images: bool = False,
        description_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Add images from a folder to the corpus.
        
        Args:
            folder: Path to the folder containing images
            recursive: Whether to scan subfolders
            max_files: Maximum number of files to add
            describe_images: If True and vision_describer is set, generate descriptions
            description_callback: Optional callback(current, total) for progress
        
        Returns: number of images added
        """
        images = FolderScanner.scan_images(folder, recursive=recursive, max_files=max_files)
        
        # If vision describer is available and requested, describe images
        if describe_images and self.vision_describer and images:
            def progress_cb(curr, total):
                if description_callback:
                    description_callback(curr, total)
            
            descriptions = self.vision_describer.describe_batch(images, progress_callback=progress_cb)
            
            # Update image items with descriptions
            for img in images:
                if img.path in descriptions:
                    img.description = descriptions[img.path]
        
        self.corpus.images.extend(images)
        return len(images)
    
    def add_text(self, text: str, source: str = "direct", chunk: bool = True) -> int:
        """Add raw text to corpus."""
        if chunk:
            chunks = TextChunker.chunk_paragraphs(text)
        else:
            chunks = [text]
        
        # Use existing corpus size to generate unique IDs
        base_idx = len(self.corpus.text_chunks)
        
        for i, chunk_text in enumerate(chunks):
            tc = TextChunk(
                id=f"{source}_{base_idx + i}",
                text=chunk_text,
                source=source,
                chunk_idx=base_idx + i
            )
            self.corpus.text_chunks.append(tc)
        
        return len(chunks)
    
    # ---- Embedding & Clustering ----
    
    def _embed_corpus(
        self, 
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[List[str], np.ndarray, List[str]]:
        """
        Embed all corpus items.
        
        Args:
            progress_callback: Optional callback(stage, progress) where progress is 0-1
        
        Returns: (ids, embeddings, types)
        """
        ids = []
        embeddings = []
        types = []
        
        # Text chunks
        if self.corpus.text_chunks:
            if progress_callback:
                progress_callback("Embedding text chunks", 0.0)
            
            texts = [c.text for c in self.corpus.text_chunks]
            text_ids = [c.id for c in self.corpus.text_chunks]
            
            print(f"DEBUG: Embedding {len(texts)} text chunks...")
            
            enc = self.text_enc
            if hasattr(enc, 'encode'):
                text_embs = enc.encode(texts)
            else:
                text_embs = enc(texts)
            
            print(f"DEBUG: Raw embeddings shape: {text_embs.shape}, dtype: {text_embs.dtype}")
            
            # Check for NaN or all-zero embeddings
            nan_count = np.isnan(text_embs).any(axis=1).sum()
            zero_count = (np.abs(text_embs).sum(axis=1) < 1e-8).sum()
            if nan_count > 0 or zero_count > 0:
                print(f"DEBUG: WARNING - {nan_count} NaN embeddings, {zero_count} zero embeddings")
            
            text_embs = text_embs / (np.linalg.norm(text_embs, axis=1, keepdims=True) + 1e-8)
            
            print(f"DEBUG: Normalized embeddings - norms range: [{np.linalg.norm(text_embs, axis=1).min():.3f}, {np.linalg.norm(text_embs, axis=1).max():.3f}]")
            
            ids.extend(text_ids)
            embeddings.append(text_embs)
            types.extend(["text"] * len(text_ids))
            
            if progress_callback:
                progress_callback("Text embedding complete", 0.5)
        
        # Images
        if self.corpus.images and self.img_enc is not None:
            if progress_callback:
                progress_callback("Embedding images", 0.5)
            
            img_paths = [img.path for img in self.corpus.images]
            img_ids = [img.id for img in self.corpus.images]
            
            try:
                if hasattr(self.img_enc, 'encode_images'):
                    img_embs = self.img_enc.encode_images(img_paths)
                else:
                    img_embs = self.img_enc(img_paths)
                
                img_embs = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-8)
                
                ids.extend(img_ids)
                embeddings.append(img_embs)
                types.extend(["image"] * len(img_ids))
                
                if progress_callback:
                    progress_callback("Image embedding complete", 0.8)
            except Exception as e:
                print(f"Warning: Image embedding failed: {e}")
        
        if not embeddings:
            return [], np.array([]), []
        
        if progress_callback:
            progress_callback("Combining embeddings", 0.9)
        
        # Combine (pad to same dimension if needed)
        max_dim = max(e.shape[1] for e in embeddings)
        padded = []
        for emb in embeddings:
            if emb.shape[1] < max_dim:
                pad = np.zeros((emb.shape[0], max_dim - emb.shape[1]))
                emb = np.concatenate([emb, pad], axis=1)
            padded.append(emb)
        
        all_embs = np.vstack(padded)
        all_embs = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
        
        return ids, all_embs, types
    
    def _build_similarity_graph(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        k: int = 15,
        min_sim: float = 0.1  # Minimum similarity threshold for edges
    ):
        """Build k-NN similarity graph."""
        if nx is None:
            raise ImportError("networkx required for clustering")
        
        n = len(ids)
        print(f"DEBUG: _build_similarity_graph received {n} ids, embeddings shape={embeddings.shape}")
        
        if n == 0:
            return nx.Graph()
        
        # Compute cosine similarity matrix
        S = embeddings @ embeddings.T
        np.fill_diagonal(S, 0.0)
        
        # Debug: check similarity distribution
        upper_tri = S[np.triu_indices(n, k=1)]
        if len(upper_tri) > 0:
            print(f"DEBUG: Similarity stats - min={upper_tri.min():.3f}, max={upper_tri.max():.3f}, mean={upper_tri.mean():.3f}, median={np.median(upper_tri):.3f}")
        
        # Build k-NN adjacency
        kk = min(k, max(1, n - 1))
        print(f"DEBUG: Using k={kk} for k-NN")
        
        A = np.zeros_like(S)
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        rows = np.arange(n)[:, None]
        A[rows, idx] = S[rows, idx]
        A = np.maximum(A, A.T)  # Make symmetric
        
        # Apply minimum similarity threshold
        A[A < min_sim] = 0
        
        edges_before_threshold = (A > 0).sum() // 2
        print(f"DEBUG: {edges_before_threshold} potential edges after k-NN, min_sim={min_sim}")
        
        G = nx.Graph()
        for u in ids:
            G.add_node(u)
        
        print(f"DEBUG: Added {G.number_of_nodes()} nodes to graph")
        
        # Add edges
        nz = np.argwhere(A > 0)
        edge_count = 0
        for i, j in nz:
            if i < j:
                G.add_edge(ids[i], ids[j], weight=float(A[i, j]))
                edge_count += 1
        
        print(f"DEBUG: Graph - {G.number_of_nodes()} nodes, {edge_count} edges added")
        
        return G
    
    def _cluster_graph(
        self,
        G,
        min_size: int = 3,
        resolution: float = 1.0
    ) -> Dict[int, List[str]]:
        """Cluster the similarity graph using Louvain community detection."""
        if G.number_of_edges() == 0:
            print("DEBUG: No edges in graph, cannot cluster")
            return {}
        
        try:
            # Try Louvain (built into networkx)
            communities = nx.community.louvain_communities(
                G, weight='weight', resolution=resolution, seed=42
            )
            
            print(f"DEBUG: Louvain found {len(communities)} communities: {[len(c) for c in communities]}")
            
            clusters = {}
            for cid, members in enumerate(communities):
                if len(members) >= min_size:
                    clusters[cid] = list(members)
            
            print(f"DEBUG: After min_size={min_size} filter: {len(clusters)} clusters remain")
            
            return clusters
            
        except Exception as e:
            print(f"DEBUG: Louvain failed: {e}, trying connected components")
            # Fallback: connected components
            clusters = {}
            for cid, comp in enumerate(nx.connected_components(G)):
                comp = list(comp)
                if len(comp) >= min_size:
                    clusters[cid] = comp
            return clusters
    
    def _extract_cluster_concepts(
        self,
        cluster_ids: List[str],
        top_k: int = 30
    ) -> List[str]:
        """Extract key concepts/terms from a cluster's text content."""
        # Gather all text
        id_to_text = {c.id: c.text for c in self.corpus.text_chunks}
        texts = [id_to_text.get(cid, "") for cid in cluster_ids if cid in id_to_text]
        
        # For image items, use descriptions if available, else filenames
        id_to_img = {img.id: img for img in self.corpus.images}
        img_items = [id_to_img.get(cid) for cid in cluster_ids if cid in id_to_img]
        
        # Add image descriptions as text sources
        for img in img_items:
            if img:
                if hasattr(img, 'description') and img.description:
                    # Use vision LLM description if available
                    texts.append(img.description)
                elif hasattr(img, 'filename'):
                    # Fallback: extract words from filename
                    name = img.filename.rsplit('.', 1)[0] if '.' in img.filename else img.filename
                    name_words = re.findall(r'[a-zA-Z]{3,}', name.lower())
                    if name_words:
                        texts.append(" ".join(name_words))
        
        if not texts:
            return []
        
        # Extract noun phrases / key terms
        all_phrases = []
        for text in texts:
            # Simple extraction: words and 2-grams
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_phrases.extend(words)
            # 2-grams
            for i in range(len(words) - 1):
                all_phrases.append(f"{words[i]} {words[i+1]}")
        
        # Count and filter
        counts = Counter(all_phrases)
        
        # Remove function words and fragments (keep thematic words for LLM context)
        stopwords = {
            # Articles, pronouns, prepositions - pure function words
            'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'are',
            'was', 'were', 'been', 'being', 'has', 'had', 'not', 'but', 'what',
            'all', 'can', 'her', 'his', 'their', 'they', 'which', 'would', 'there',
            'when', 'who', 'will', 'more', 'some', 'into', 'than', 'them', 'these',
            'then', 'its', 'also', 'only', 'other', 'such', 'could', 'about',
            'our', 'one', 'own', 'those', 'between', 'within', 'through', 'each',
            'how', 'way', 'may', 'must', 'like', 'just', 'over', 'very', 'any',
            'both', 'most', 'after', 'before', 'where', 'while', 'does', 'did',
            'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came', 'see',
            'now', 'new', 'even', 'back', 'well', 'much', 'many', 'here', 'still',
            'yet', 'upon', 'thus', 'onto', 'toward', 'towards', 'among', 'across',
            # Pure fillers
            'myself', 'yourself', 'themselves', 'itself', 'ourselves', 
            'something', 'anything', 'everything', 'nothing', 
            'someone', 'anyone', 'everyone',
            # Fragment phrases to avoid
            'with the', 'from the', 'of the', 'in the', 'to the', 'and the',
            'for the', 'on the', 'at the', 'by the', 'as the', 'is the',
            'that the', 'which the', 'from our', 'with our', 'in our',
        }
        
        filtered = [
            (phrase, count) for phrase, count in counts.most_common(top_k * 3)
            if phrase not in stopwords 
            and len(phrase) > 2
            and not phrase.endswith(' the')
            and not phrase.startswith('the ')
            and not phrase.endswith(' our')
            and not phrase.startswith('our ')
            and not phrase.endswith(' out')
            and not phrase.startswith('out ')
        ]
        
        return [phrase for phrase, _ in filtered[:top_k]]
    
    # ---- Main Pipeline ----
    
    def run(
        self,
        k_neighbors: int = 15,
        min_cluster_size: int = 5,
        resolution: float = 1.0,
        use_llm: bool = True,
        top_concepts_per_cluster: int = 30,
        debug: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, List[str]]:
        """
        Run the full mining pipeline.
        
        Args:
            k_neighbors: Number of neighbors for similarity graph
            min_cluster_size: Minimum items per cluster
            resolution: Louvain resolution parameter
            use_llm: Whether to use LLM for archetype refinement
            top_concepts_per_cluster: Max concepts to extract per cluster
            debug: Print debug info to stdout
            progress_callback: Optional callback(stage_name, progress_0_to_1)
        
        Returns: {archetype_name: [descriptors]}
        """
        if len(self.corpus) == 0:
            if debug:
                print("DEBUG: Empty corpus")
            return {}
        
        # 1. Embed corpus
        if progress_callback:
            progress_callback("Embedding corpus", 0.0)
        
        ids, embeddings, types = self._embed_corpus(progress_callback=progress_callback)
        
        if debug:
            print(f"DEBUG: Embedded {len(ids)} items, shape={embeddings.shape if len(ids) > 0 else 'N/A'}")
        
        if len(ids) == 0:
            return {}
        
        # 2. Build similarity graph
        if progress_callback:
            progress_callback("Building similarity graph", 0.3)
        
        G = self._build_similarity_graph(ids, embeddings, k=k_neighbors)
        
        if debug:
            print(f"DEBUG: Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 3. Cluster
        if progress_callback:
            progress_callback("Clustering", 0.5)
        
        clusters = self._cluster_graph(G, min_size=min_cluster_size, resolution=resolution)
        
        if debug:
            print(f"DEBUG: Found {len(clusters)} clusters: {[(cid, len(m)) for cid, m in clusters.items()]}")
        
        if not clusters:
            return {}
        
        # 4. Extract concepts from each cluster
        if progress_callback:
            progress_callback("Extracting concepts", 0.7)
        
        cluster_concepts = {}
        for cid, members in clusters.items():
            concepts = self._extract_cluster_concepts(members, top_k=top_concepts_per_cluster)
            # Even if no concepts extracted, keep the cluster with placeholders
            if not concepts:
                concepts = [f"cluster_{cid}_item_{i}" for i in range(min(5, len(members)))]
            cluster_concepts[cid] = concepts
            if debug:
                print(f"DEBUG: Cluster {cid} ({len(members)} members) -> {len(concepts)} concepts: {concepts[:5]}")
        
        # 5. Refine with LLM or use simple naming
        if progress_callback:
            progress_callback("Refining with LLM" if use_llm else "Naming clusters", 0.85)
        
        if use_llm and self.llm_refiner is not None:
            try:
                if debug:
                    print("DEBUG: Calling LLM refiner...")
                archetypes = self.llm_refiner.refine_clusters(cluster_concepts)
                if debug:
                    print(f"DEBUG: LLM returned {len(archetypes)} archetypes")
                if archetypes:
                    return archetypes
            except Exception as e:
                if debug:
                    print(f"DEBUG: LLM refinement failed: {e}")
                # Fall through to simple naming
        
        # Fallback: simple naming from top concepts
        archetypes = {}
        for cid, concepts in cluster_concepts.items():
            if len(concepts) >= 2:
                name = f"{concepts[0]} / {concepts[1]}".upper()
            elif concepts:
                name = concepts[0].upper()
            else:
                name = f"CLUSTER_{cid}"
            
            archetypes[name] = concepts[:15]
        
        if debug:
            print(f"DEBUG: Final archetypes: {list(archetypes.keys())}")
        
        return archetypes
    
    def get_embeddings_for_viz(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """Get embeddings for visualization (UMAP/t-SNE in UI)."""
        return self._embed_corpus()
    
    def clear(self):
        """Clear the corpus."""
        self.corpus = MinerCorpus()
