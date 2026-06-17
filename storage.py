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

STORE_DIR = Path("embeddings_store")

def init_store():
    """Create the storage directory if it doesn't exist. 
    
    exist_ok=True just ensures that python does not throw FileExistError if it already exist."""
    STORE_DIR.mkdir(exist_ok=True)

def save_embedding(person_id: str, name: str, embedding: list[float]):
    """
    Append a new embedding for a person.
    Call this once per enrolled photo.
    """
    init_store()
    filepath = STORE_DIR / f"{person_id}.json"

    if filepath.exists():
        data = json.loads(filepath.read_text())  
    else:
        data = {"id": person_id, "name": name, "embeddings": []}

    data["embeddings"].append(embedding)
    filepath.write_text(json.dumps(data))


def load_all_embeddings() -> list[dict]:
    """
    Load all enrolled people and their embeddings.
    Returns a list like:
      [{"id": "togbe", "name": "Togbe", "embeddings": [[...], [...]]}, ...]
    """
    init_store()
    people = []

    # Scan through all json files
    for filepath in STORE_DIR.glob("*.json"):
        data = json.loads(filepath.read_text())
        people.append(data)

    return people  # fixed: was inside the loop, returned after first file only

def delete_person(person_id: str) -> bool:
    """Remove a person's embedding file. Returns True if deleted."""
    filepath = STORE_DIR / f"{person_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def list_enrolled() -> list[dict]:
    """List enrolled people without including the raw embedding vectors."""
    people = load_all_embeddings()
    return [
        {"id": p["id"], "name": p["name"], "photo_count": len(p["embeddings"])}
        for p in people
    ]