from typing import Optional, List, Union
from pathlib import Path
import numpy as np
import torch


class TextEmbedder:
    """
    Flexible text embedder supporting:
      - Original SentenceTransformer backend
      - Qwen2 / Qwen3 embedding models (EOS pooling by default)
    Also supports optional instruction + context prompting.
    Fallback: hashing-based projection.
    """
    def __init__(self, backend: str = "original", model: Optional[str] = None, dim_fallback: int = 384,
                 device: Optional[str] = None, pooling: str = "eos",
                 default_instruction: Optional[str] = None, default_context: Optional[str] = None):
        self.backend_type = backend.lower()
        self.model_name = model
        self.dim = dim_fallback
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling.lower()
        self.default_instruction = default_instruction
        self.default_context = default_context

        self._backend = None
        self._tokenizer = None
        self._proj = None
        self.audio_capable = False        # <-- ADD

        if self.backend_type == "sentence-transformer":
            self._init_original(model or "sentence-transformers/all-mpnet-base-v2")
        elif self.backend_type == "qwen2":
            self._init_qwen(model or "Qwen/Qwen2-Embedding")
        elif self.backend_type == "qwen3":
            self._init_qwen(model or "Qwen/Qwen3-Embedding-0.6B")
        elif self.backend_type == "cloudflare":
            self._init_cloudflare(model or "@cf/baai/bge-base-en-v1.5")
        elif self.backend_type == "audioclip":           # <-- ADD
            self._init_audioclip(model)                  # <-- ADD
        elif self.backend_type == "clap":
            self._init_clap(model or "laion/clap-htsat-fused")

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose: original | qwen2 | qwen3 | cloudflare | audioclip")

        if self._backend is None and self.backend_type not in ("audioclip", "cloudflare"):  # these use separate vars
            self._init_fallback()

    # ---------- init helpers ----------
    def _init_cloudflare(self, model_name: str):
        """Initialize Cloudflare Workers AI embedding backend."""
        import os
        self._cf_model = model_name
        self._cf_account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self._cf_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
        
        # Also check secrets.toml
        if not self._cf_account or not self._cf_token:
            try:
                try:
                    import tomli
                except ImportError:
                    import tomllib as tomli  # Python 3.11+ built-in
                secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
                if secrets_path.exists():
                    with open(secrets_path, "rb") as f:
                        secrets = tomli.load(f)
                    cf = secrets.get("cloudflare", {})
                    self._cf_account = self._cf_account or cf.get("account_id", "")
                    self._cf_token = self._cf_token or cf.get("api_token", "")
            except Exception as e:
                print(f"[Embedder] Warning: Could not read secrets.toml: {e}")
        
        if not self._cf_account or not self._cf_token:
            raise ValueError("Cloudflare credentials not found. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        
        # Dimension mapping for known models
        dim_map = {
            "@cf/baai/bge-small-en-v1.5": 384,
            "@cf/baai/bge-base-en-v1.5": 768,
            "@cf/baai/bge-large-en-v1.5": 1024,
            "@cf/baai/bge-m3": 1024,
            "@cf/qwen/qwen3-embedding-0.6b": 1024,
            "@cf/google/embeddinggemma-300m": 768,
        }
        self.dim = dim_map.get(model_name, 768)
        self._cf_ready = True
        print(f"[Embedder] Cloudflare ready: {model_name} (dim={self.dim})")
    
    def _call_cloudflare_embed(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """Call Cloudflare embedding API with retry logic."""
        import requests
        import time
        
        url = f"https://api.cloudflare.com/client/v4/accounts/{self._cf_account}/ai/run/{self._cf_model}"
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json={"text": texts},
                    headers={
                        "Authorization": f"Bearer {self._cf_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("success"):
                    raise RuntimeError(f"Cloudflare API error: {data.get('errors', [])}")
                
                # Extract embeddings from response
                result = data.get("result", {})
                
                if isinstance(result, dict) and "data" in result:
                    embeddings = result["data"]
                elif isinstance(result, list):
                    embeddings = result
                else:
                    raise RuntimeError(f"Unexpected Cloudflare response format: {type(result)}")
                
                return np.array(embeddings, dtype=np.float32)
                
            except requests.exceptions.HTTPError as e:
                last_error = e
                if response.status_code == 500 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                    print(f"[Embedder] Cloudflare 500 error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        raise last_error

    def _init_clap(self, model_name: str):
        try:
            from transformers import ClapProcessor, ClapModel
            self.clap_processor = ClapProcessor.from_pretrained(model_name)
            self._clap = ClapModel.from_pretrained(model_name).to(self.device)
            self._clap.eval()
            with torch.no_grad():
                dummy = np.zeros(48000, dtype=np.float32)  # 1s silence @48k
                ain = self.clap_processor(audios=[dummy], sampling_rate=48000, return_tensors="pt")
                ain = {k: v.to(self.device) for k, v in ain.items()}
                a = self._clap.get_audio_features(**ain)
                self.dim = int(a.shape[-1])
            self.audio_capable = True
            self._ac_sr = 48000
            print(f"[Embedder] CLAP loaded: {model_name} (dim={self.dim})")
        except Exception as e:
            raise RuntimeError(f"CLAP load failed: {e}")

    def _init_audioclip(self, model_name: Optional[str] = None):
        """
        AudioCLIP via open_clip text tower + tiny audio projector.
        No torchaudio, no HF dependency.
        """
        try:
            import open_clip
            # TEXT PATH — CLIP text encoder
            self.oc_model, _, self.oc_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
            )
            self.oc_model.eval()
            self.oc_tokenizer = open_clip.get_tokenizer("ViT-B-32")

            # match CLIP text embedding dim
            self.dim = int(self.oc_model.text_projection.shape[1]) if hasattr(self.oc_model, "text_projection") else 512

            # AUDIO PATH — simple conv projector
            self.audio_proj = nn.Sequential(
                nn.Conv1d(1, 64, 5, 2, 2), nn.ReLU(),
                nn.Conv1d(64, 128, 5, 2, 2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(128, self.dim),
            ).to(self.device).eval()

            self.audio_capable = True
            self._ac_sr = 48000
            self._ac_len = self._ac_sr * 10  # seconds

            print(f"[Embedder] AudioCLIP (open_clip) ready, dim={self.dim}.")

        except Exception as e:
            raise RuntimeError(f"AudioCLIP init failed (open_clip): {e}")

    # --- replace your embed_audio_array fallback with this (no torchaudio) ---
    @torch.no_grad()
    def embed_audio_array(self, wave: np.ndarray, sr: int) -> np.ndarray:
        if self.backend_type == "audioclip":
            # resample to 48k if needed
            if sr != self._ac_sr:
                t_old = np.linspace(0, 1, len(wave), endpoint=False)
                t_new = np.linspace(0, 1, int(len(wave) * (self._ac_sr / sr)), endpoint=False)
                wave = np.interp(t_new, t_old, wave).astype(np.float32)
                sr = self._ac_sr
            # center trim or pad to fixed length
            if len(wave) > self._ac_len:
                s = (len(wave) - self._ac_len) // 2
                wave = wave[s:s+self._ac_len]
            elif len(wave) < self._ac_len:
                pad = self._ac_len - len(wave)
                wave = np.pad(wave, (pad//2, pad - pad//2))
            x = torch.from_numpy(wave).float().to(self.device)[None, None, :]  # (1,1,T)
            z = self.audio_proj(x)[0]
            z = z / (z.norm() + 1e-8)
            return z.detach().cpu().float().numpy()

        if self.backend_type == "clap" and hasattr(self, "_clap"):
            ain = self.clap_processor(audios=[wave], sampling_rate=sr, return_tensors="pt")
            ain = {k: v.to(self.device) for k, v in ain.items()}
            z = self._clap.get_audio_features(**ain)[0]
            z = z / (z.norm() + 1e-8)
            return z.detach().cpu().float().numpy()

        raise RuntimeError("embed_audio_array called but this backend has no audio path.")

    def _init_original(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
            self._backend = SentenceTransformer(model_name)
            self.dim = int(self._backend.get_sentence_embedding_dimension())
            print(f"[Embedder] SentenceTransformer loaded ({self.dim}-d).")
        except Exception as e:
            warnings.warn(f"SentenceTransformer load failed: {e}")
            self._backend = None

    def _init_qwen(self, model_name):
        try:
            print(f"[Embedder] Loading Qwen model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Use low_cpu_mem_usage=True to optimize loading
            self._backend = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True
            ).to(self.device)
            self._backend.eval()

            # CPU Optimization: Dynamic Quantization (Int8)
            # Drastically reduces RAM (2.5GB -> ~700MB) and speeds up CPU inference
            if self.device == "cpu":
                try:
                    print("[Embedder] Applying dynamic int8 quantization for CPU...")
                    self._backend = torch.quantization.quantize_dynamic(
                        self._backend, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("[Embedder] Quantization complete.")
                except Exception as e:
                    print(f"[Embedder] Quantization skipped: {e}")
            # probe dim with EOS pooling
            with torch.no_grad():
                toks = self._tokenizer("probe", return_tensors="pt").to(self.device)
                out = self._backend(**toks)
                pooled = self._pool(out.last_hidden_state, toks["attention_mask"])
                self.dim = int(pooled.shape[1])
            print(f"[Embedder] Qwen loaded ({self.dim}-d), pooling={self.pooling}.")
        except Exception as e:
            warnings.warn(f"Qwen load failed: {e}")
            self._backend = None
            self._tokenizer = None

    def _init_fallback(self):
        rng = np.random.default_rng(42)
        self._proj = rng.normal(0, 1 / math.sqrt(self.dim), size=(7000, self.dim))
        print(f"[Embedder] Using hashing fallback ({self.dim}-d).")

    # ---------- pooling ----------
    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden_state: [B, T, H]
        attention_mask   : [B, T] with 1 for real tokens, 0 for pad
        """
        if self.pooling == "mean":
            # mean over non-padding tokens
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            sum_hidden = (last_hidden_state * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = sum_hidden / lengths
            return _l2norm_torch(pooled)

        if self.pooling == "cls":
            # first token (works if model prepends a special token)
            pooled = last_hidden_state[:, 0, :]
            return _l2norm_torch(pooled)

        # "last": last timestep regardless of padding (usually not what you want)
        if self.pooling == "last":
            pooled = last_hidden_state[:, -1, :]
            return _l2norm_torch(pooled)

        # default: "eos" — last *non-pad* token
        idxs = attention_mask.sum(dim=1) - 1  # [B]
        idxs = idxs.clamp(min=0)
        bsz = last_hidden_state.size(0)
        pooled = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), idxs, :]
        return _l2norm_torch(pooled)

    # ---------- input templating ----------
    def _apply_prompt_template(self, texts: List[str],
                            instruction: Optional[str],
                            context: Optional[str]) -> List[str]:
        inst = instruction if instruction is not None else self.default_instruction
        ctx  = context if context is not None else self.default_context

        # Nothing to add? Return unchanged.
        if inst is None and ctx is None:
            return texts

        # Allow a single global context (str) or per-text contexts (list/tuple/ndarray)
        ctx_is_seq = isinstance(ctx, (list, tuple, np.ndarray))
        if ctx_is_seq:
            if len(ctx) != len(texts):
                raise ValueError(f"Context list length ({len(ctx)}) must match texts ({len(texts)})")

        templated = []
        for i, t in enumerate(texts):
            parts = []
            if inst:
                parts.append(str(inst).strip())                # e.g., "Instruction: …"
            if ctx is not None:
                parts.append(f"Context: {ctx[i] if ctx_is_seq else ctx}")
            parts.append(f"Text: {t}")
            templated.append("\n".join(parts))
        return templated


    # ---------- encoding ----------
    def encode(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        context: Optional[str] = None,
        batch_size: int = 32  # Process in chunks to save RAM
    ):
        # --- Cloudflare path (fastest!) ---
        if self.backend_type == "cloudflare" and hasattr(self, "_cf_ready"):
            all_embeddings = []
            # Smaller batches for stability (Qwen3 especially needs this)
            batch_size_cf = 25 if "qwen" in self._cf_model.lower() else 50
            for i in range(0, len(texts), batch_size_cf):
                batch = texts[i:i + batch_size_cf]
                embs = self._call_cloudflare_embed(batch)
                all_embeddings.append(embs)
            if not all_embeddings:
                return np.array([], dtype=np.float32)
            result = np.concatenate(all_embeddings, axis=0)
            # Normalize
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-9)
            return result.astype(np.float32)
        
        # --- Qwen path ---
        if self._backend is not None and self._tokenizer is not None and self.backend_type in ("qwen2","qwen3"):
            inputs = self._apply_prompt_template(texts, instruction, context)
            all_embeddings = []
            
            # Batch processing loop
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]
                with torch.no_grad():
                    toks = self._tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    out = self._backend(**toks)
                    pooled = self._pool(out.last_hidden_state, toks["attention_mask"])
                    all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            
            if not all_embeddings:
                return np.array([], dtype=np.float32)
            return np.concatenate(all_embeddings, axis=0)

        # --- SentenceTransformer (handles batching internally usually, but we can force it if needed) ---
        if self.backend_type == "original" and self._backend is not None:
            return np.asarray(self._backend.encode(texts, normalize_embeddings=True, batch_size=batch_size), dtype=np.float32)

        # --- NEW: AudioCLIP text path (open_clip) ---
        if self.backend_type == "audioclip":
            toks = self.oc_tokenizer(texts).to(self.device)   # list[str] -> token ids
            with torch.no_grad():
                z = self.oc_model.encode_text(toks)           # [B, D]
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            return z.detach().cpu().numpy().astype(np.float32)

        # inside TextEmbedder.encode(...)
        if self.backend_type == "clap" and hasattr(self, "_clap"):
            with torch.no_grad():
                tin = self.clap_processor(text=texts, return_tensors="pt", padding=True)
                tin = {k: v.to(self.device) for k, v in tin.items()}
                z = self._clap.get_text_features(**tin)  # [B, D]
                z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            return z.detach().cpu().numpy().astype(np.float32)


        # --- hashing fallback (last resort) ---
        vocab = 7000
        def vec(t):
            v = np.zeros(vocab, np.float32)
            for tok in t.lower().split():
                v[hash(tok) % vocab] += 1
            return v
        if self._proj is None:
            # safety: create projection if not already set
            self._init_fallback()
        M = np.stack([vec(t) for t in texts]) @ self._proj
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        return M.astype(np.float32)

