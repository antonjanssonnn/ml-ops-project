from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import json
from dataclasses import asdict

from ml_ops_project.ingest import Chunk, load_documents

def build_index(chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2") -> tuple[faiss.Index, SentenceTransformer, str]:
    sentence_model = SentenceTransformer(model_name) 
    embeddings = sentence_model.encode([chunk.text for chunk in chunks], convert_to_numpy=True) #Encodes the text of each chunk into a vector.
    embeddings.shape[1] #Retrieves the dimensionality of the embeddings (number of features) (8 is the amount of rows that is embedded, and 384 the amount of columns).
    index =faiss.IndexFlatL2(embeddings.shape[1]) #Creates a FAISS index for efficient similarity search using L2 distance.
    index.add(embeddings) #Adds the embeddings to the FAISS index.
    return index, sentence_model, model_name

def save_index(index: faiss.Index, chunks: list[Chunk], model_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index")) #Saves the FAISS index to a file named "faiss.index" in the specified output directory.

    chunks_data = [asdict(chunk) for chunk in chunks]
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": model_name}, f, indent=2)