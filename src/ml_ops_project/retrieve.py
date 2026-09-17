from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer

def load_index(index_dir: Path) -> tuple[faiss.Index, list[dict], SentenceTransformer]:
    index = faiss.read_index(str(index_dir/"faiss.index")) #Now we have the index.
    #We do need the chunks as well.

    with open(index_dir / "chunks.json", "r") as f:
        chunks = json.load(f) #Load the chunks from the JSON file.

    with open(index_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
        model_name = metadata.get("model_name")

    sentence_model = SentenceTransformer(model_name) #Load the model using the model name from the metadata.

    return index, chunks, sentence_model



def search(query: str, index: faiss.Index, model: SentenceTransformer, chunks: list[dict], top_k: int = 5) -> list[dict]:

    query_embedding = model.encode([query]) #Encode the query into a vector. (Need to have a query so I can search the index for similar chunks.)
    _distance, indices = index.search(query_embedding, top_k) #Search the index for the top_k most similar chunks to the query embedding.
    # I now want to return the chunks that correspond to the indices of the most similar chunks.
    return [chunks[i] for i in indices[0]] #Return the chunks that are closest (i.e. indiccs[0] is the closest match) Brute forcing, there is most likely better alternatives for bigger projects.
    