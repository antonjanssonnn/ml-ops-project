from dataclasses import dataclass
from pathlib import Path

@dataclass
class Chunk:
    id: int
    source: str   # e.g. "001_saturn.txt"
    text: str

def load_documents(data_dir: Path) -> list[Chunk]:
    # NOTE: Using sorted() here to ensure a consistent order of chunks, since Path.iterdir() does not guarantee any particular order. This is important for reproducibility and testing.
    chunks = []
    filtered_data_dir = [chunk for chunk in data_dir.iterdir() if chunk.is_file() and chunk.suffix == ".txt"]
    for i, chunk in enumerate(sorted(filtered_data_dir)):
            chunks.append(Chunk(
                id=i,
                source=chunk.name,
                text=chunk.read_text(encoding="utf-8"),
            ))
    return chunks
