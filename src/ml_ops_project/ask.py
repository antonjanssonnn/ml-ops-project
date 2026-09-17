from ml_ops_project.generate import generate_answer
from ml_ops_project.retrieve import load_index, search
from pathlib import Path
from anthropic import Anthropic
import sys


def main():

    question = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")

    index, chunks, model = load_index(Path("index")) # Loading the index, chunks, and model from the specified directory.

    found_chunks = search(question, index, model, chunks, top_k=3) # Searches for the most relevant chunks based on the question.

    client = Anthropic() # Loads the cliennt for the LLM (Claude in this case)
    answer = generate_answer(question, found_chunks, client, model = "claude-haiku-4-5-20251001") # Generates an answer based on the question and the found chunks using the LLM.

    print(answer) # Prints the answer to the console.
    for chunk in found_chunks:
        print(f"Source: {chunk['source']}") # Prints the source of each found chunk to the console.

if __name__ == "__main__":
    main()