import datetime
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient

from research_assistant.config import TAVILY_API_KEY

os.environ.setdefault("HF_TOKEN", "")

_PACKAGE_ROOT = Path(__file__).resolve().parent
_DATA_DIR = _PACKAGE_ROOT / "data"
CHROMA_PATH = _DATA_DIR / "chroma_db"

_tavily_client: TavilyClient | None = None

_embed_model = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your environment or a .env file in the project root."
            )
        _tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    return _tavily_client


def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


class VectorDB:
    def __init__(self, reset: bool = False):
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        if reset:
            try:
                self.client.delete_collection(name="research_kb")
            except Exception:
                pass
        self.collection = self.client.get_or_create_collection(name="research_kb")

    def index_segments(self, text_content: str) -> str:
        if not text_content.strip():
            return "Error: empty content."

        chunks = [text_content[i : i + 1000] for i in range(0, len(text_content), 800)]

        ids = [f"id_{datetime.datetime.now().timestamp()}_{i}" for i in range(len(chunks))]
        embeddings = get_embedding_model().encode(chunks).tolist()

        self.collection.add(documents=chunks, ids=ids, embeddings=embeddings)
        return f"Success: {len(chunks)} segments indexed."

    def query_kb(self, question: str) -> str:
        query_embedding = get_embedding_model().encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=2)
        return "\n".join(results["documents"][0]) if results["documents"] else "No matching information found."


def find_urls(query: str) -> list:
    results = _get_tavily().search(query=query, max_results=3)["results"]
    return [
        {"url": r["url"], "summary": r.get("content", "")[:200]}
        for r in results
    ]


def extract_content(url: str) -> list:
    try:
        raw = _get_tavily().extract(urls=[url])["results"][0]["raw_content"]
        clean = " ".join(raw.split())[:1500]
        return [{"source": url, "content": clean}]
    except Exception as e:
        return [{"source": url, "content": f"Extraction error: {e}"}]
