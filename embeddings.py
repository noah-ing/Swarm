"""
Embeddings: Real vector embeddings using OpenAI's text-embedding-3-small.

Features:
- Semantic understanding of text
- SQLite cache to avoid re-embedding
- Batch processing for efficiency
- Fallback to simple embeddings if API unavailable
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small output size


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text_hash: str
    embedding: list[float]
    model: str
    cached: bool = False


class EmbeddingService:
    """
    Embedding service with caching and fallback.

    Uses OpenAI's text-embedding-3-small for high-quality semantic embeddings,
    with SQLite caching to minimize API calls and costs.
    """

    def __init__(self, cache_path: Optional[str] = None):
        # Setup cache database
        if cache_path is None:
            cache_dir = Path.home() / ".swarm"
            cache_dir.mkdir(exist_ok=True)
            cache_path = str(cache_dir / "embeddings_cache.db")

        self.cache_path = cache_path
        self._init_cache()

        # Setup OpenAI client
        self.client: Optional[OpenAI] = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)

    def _init_cache(self):
        """Initialize the embedding cache database."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_model
            ON embedding_cache(model)
        """)

        conn.commit()
        conn.close()

    def _hash_text(self, text: str) -> str:
        """Create a hash of the text for caching."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def _get_cached(self, text_hash: str) -> Optional[list[float]]:
        """Retrieve embedding from cache if available."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding FROM embedding_cache
            WHERE text_hash = ? AND model = ?
        """, (text_hash, EMBEDDING_MODEL))

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def _cache_embedding(self, text_hash: str, embedding: list[float]):
        """Cache an embedding for future use."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model)
            VALUES (?, ?, ?)
        """, (text_hash, json.dumps(embedding), EMBEDDING_MODEL))

        conn.commit()
        conn.close()

    def _simple_embedding(self, text: str) -> list[float]:
        """
        Fallback: Create a simple bag-of-words embedding.
        Used when OpenAI API is unavailable.
        """
        words = text.lower().split()
        word_set = set(words)

        # Programming-focused features
        features = [
            "file", "read", "write", "create", "delete", "update", "fix",
            "bug", "error", "test", "function", "class", "import", "api",
            "database", "sql", "json", "http", "request", "response",
            "loop", "condition", "variable", "string", "number", "list",
            "dict", "array", "object", "return", "print", "log", "debug",
            "config", "setting", "env", "path", "directory", "git", "commit",
            "python", "javascript", "typescript", "react", "node", "bash",
            "install", "package", "dependency", "build", "deploy", "run",
            "refactor", "optimize", "performance", "memory", "cache",
            "auth", "user", "password", "token", "session", "cookie",
            "frontend", "backend", "server", "client", "component", "hook",
        ]

        embedding = [1.0 if f in word_set else 0.0 for f in features]
        embedding.append(min(1.0, len(words) / 100))
        embedding.append(min(1.0, len(word_set) / 50))

        # Pad to standard length for compatibility
        while len(embedding) < 64:
            embedding.append(0.0)

        return embedding

    def embed(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """
        Get embedding for text.

        Uses OpenAI API with caching, falls back to simple embedding
        if API is unavailable.

        Args:
            text: Text to embed
            use_cache: Whether to use cached embeddings

        Returns:
            EmbeddingResult with embedding vector
        """
        text_hash = self._hash_text(text)

        # Check cache first
        if use_cache:
            cached = self._get_cached(text_hash)
            if cached:
                return EmbeddingResult(
                    text_hash=text_hash,
                    embedding=cached,
                    model=EMBEDDING_MODEL,
                    cached=True
                )

        # Try OpenAI API
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text[:8000],  # Limit input length
                )
                embedding = response.data[0].embedding

                # Cache the result
                if use_cache:
                    self._cache_embedding(text_hash, embedding)

                return EmbeddingResult(
                    text_hash=text_hash,
                    embedding=embedding,
                    model=EMBEDDING_MODEL,
                    cached=False
                )
            except Exception as e:
                # Fall through to simple embedding
                pass

        # Fallback to simple embedding
        embedding = self._simple_embedding(text)
        return EmbeddingResult(
            text_hash=text_hash,
            embedding=embedding,
            model="simple-bow",
            cached=False
        )

    def embed_batch(self, texts: list[str], use_cache: bool = True) -> list[EmbeddingResult]:
        """
        Embed multiple texts efficiently.

        Batches API calls for texts not in cache.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings

        Returns:
            List of EmbeddingResults in same order as input
        """
        results: list[Optional[EmbeddingResult]] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)

            if use_cache:
                cached = self._get_cached(text_hash)
                if cached:
                    results[i] = EmbeddingResult(
                        text_hash=text_hash,
                        embedding=cached,
                        model=EMBEDDING_MODEL,
                        cached=True
                    )
                    continue

            uncached_indices.append(i)
            uncached_texts.append(text)

        # Batch embed uncached texts
        if uncached_texts and self.client:
            try:
                # OpenAI supports batch embedding
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=[t[:8000] for t in uncached_texts],
                )

                for j, embedding_data in enumerate(response.data):
                    idx = uncached_indices[j]
                    text = uncached_texts[j]
                    text_hash = self._hash_text(text)
                    embedding = embedding_data.embedding

                    if use_cache:
                        self._cache_embedding(text_hash, embedding)

                    results[idx] = EmbeddingResult(
                        text_hash=text_hash,
                        embedding=embedding,
                        model=EMBEDDING_MODEL,
                        cached=False
                    )
            except Exception:
                # Fall through to simple embedding
                pass

        # Fill remaining with simple embeddings
        for i, result in enumerate(results):
            if result is None:
                text = texts[i]
                text_hash = self._hash_text(text)
                embedding = self._simple_embedding(text)
                results[i] = EmbeddingResult(
                    text_hash=text_hash,
                    embedding=embedding,
                    model="simple-bow",
                    cached=False
                )

        return results

    def similarity(self, a: list[float], b: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity (-1 to 1, higher is more similar)
        """
        # Handle different dimensions (real vs simple embeddings)
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_cache_stats(self) -> dict:
        """Get statistics about the embedding cache."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM embedding_cache")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT model, COUNT(*) FROM embedding_cache GROUP BY model
        """)
        by_model = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_cached": total,
            "by_model": by_model,
            "cache_path": self.cache_path,
        }


# Global singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
