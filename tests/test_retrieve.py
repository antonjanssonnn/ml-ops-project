from ml_ops_project.retrieve import load_index, search
from ml_ops_project.ingest import load_documents
from ml_ops_project.embed_index import build_index, save_index
from pathlib import Path

def test_load_index(tmp_path: Path):
    # Load documents and create chunks
    chunks = load_documents(Path("data/raw"))

    # Build the index
    index, model, model_name = build_index(chunks)

    # Save the index and chunks to a temporary directory
    save_index(index, chunks, model_name, tmp_path)

    # Load the index, chunks, and model from the temporary directory
    loaded_index, loaded_chunks, loaded_model = load_index(tmp_path)

    # Check that the loaded index has the same number of entries as the original index
    assert loaded_index.ntotal == index.ntotal

    # Check that the loaded chunks match the original chunks
    assert len(loaded_chunks) == len(chunks)
    for original_chunk, loaded_chunk in zip(chunks, loaded_chunks):
        assert original_chunk.id == loaded_chunk["id"]
        assert original_chunk.text == loaded_chunk["text"]
        assert original_chunk.source == loaded_chunk["source"]

    # Check that the loaded model is not None
    assert loaded_model._get_name() == model._get_name()

def test_search(tmp_path: Path):
    # Load documents and create chunks
    chunks = load_documents(Path("data/raw"))

    # Build the index
    index, _model, model_name = build_index(chunks)

    # Save the index and chunks to a temporary directory
    save_index(index, chunks, model_name, tmp_path)

    # Load the index, chunks, and model from the temporary directory
    loaded_index, loaded_chunks, loaded_model = load_index(tmp_path)

    # Perform a search using the loaded index and model
    query = "What is Saturn?"
    results = search(query, loaded_index, loaded_model, loaded_chunks, top_k=3)

    # Check that the results are not empty
    assert len(results) > 0

    # Check that the top result is relevant to the query
    top_result = results[0]
    assert "Saturn" in top_result["text"]