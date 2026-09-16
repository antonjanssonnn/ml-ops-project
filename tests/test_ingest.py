from pathlib import Path
from ml_ops_project.ingest import load_documents

def test_load_documents_finds_all_txt_files():
    chunks = load_documents(Path("data/raw"))
    assert len(chunks) == 8

def test_load_documents_ids_are_sequential():
    chunks = load_documents(Path("data/raw"))
    assert [chunk.id for chunk in chunks] == list(range(8))

def test_load_documents_texts_matches_file():
    chunks = load_documents(Path("data/raw"))
    saturn = next(chunk for chunk in chunks if chunk.source == "001_saturn.txt")
    assert "Saturn" in saturn.text