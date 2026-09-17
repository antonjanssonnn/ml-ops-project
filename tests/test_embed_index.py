from pathlib import Path
from ml_ops_project.ingest import load_documents
from ml_ops_project.embed_index import save_index, build_index

def test_build_index_creates_chunks_and_index():

    # Load documents and create chunks
    chunks = load_documents(Path("data/raw"))

    # Build the index
    index, model = build_index(chunks)

    assert index.ntotal == len(chunks)
    assert model is not None

def test_save_index_creates_files(tmp_path: Path):
    # Load documents and create chunks
    chunks = load_documents(Path("data/raw"))

    # Build the index
    index, model = build_index(chunks)

    # Save the index and chunks to a temporary directory
    save_index(index, chunks, tmp_path)

    # Check that the files were created
    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "chunks.json").exists()