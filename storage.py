"""
storage.py — Persist and retrieve face embeddings.

Each enrolled person is stored as:
  embeddings_store/
    togbe.json  → { "name": "Togbe", "embeddings": [[0.12, -0.34, ...], ...] }

Why multiple embeddings per person?
  - We store the average of several photos. More photos = better matching.
  - Right now we store each photo's embedding separately and average at match time.
    Later you can pre-average them into a single vector for speed.

Why JSON and not a database yet?
  - Keeps this service stateless and easy to inspect/debug.
  - When you integrate into your Node.js/PostgreSQL stack, you'll move the
    embeddings into a JSONB column (PostgreSQL handles float arrays natively).
"""

import json
import os
from pathlib import Path

_cache: list[dict] | None = None

STORE_DIR = Path("embeddings_store")

def init_store():
    """Create the storage directory if it doesn't exist. 
    
    exist_ok=True just ensures that python does not throw FileExistError if it already exist."""
    STORE_DIR.mkdir(exist_ok=True)

def _invalidate_cache():
    global _cache
    _cache = None

def save_embedding(person_id, name, embedding):
    global _cache
    init_store()
    filepath = STORE_DIR / f"{person_id}.json"
    data = json.loads(filepath.read_text()) if filepath.exists() else {"id": person_id, "name": name, "embeddings": []}
    data["embeddings"].append(embedding)
    filepath.write_text(json.dumps(data))
    _invalidate_cache()


def load_all_embeddings() -> list[dict]:
    global _cache
    if _cache is None:
        init_store()
        _cache = [json.loads(fp.read_text()) for fp in STORE_DIR.glob("*.json")]
    return _cache


def delete_person(person_id: str) -> bool:
    filepath = STORE_DIR / f"{person_id}.json"
    if filepath.exists():
        filepath.unlink()
        _invalidate_cache()
        return True
    return False

def list_enrolled() -> list[dict]:
    """List enrolled people without including the raw embedding vectors."""
    people = load_all_embeddings()
    return [
        {"id": p["id"], "name": p["name"], "photo_count": len(p["embeddings"])}
        for p in people
    ]