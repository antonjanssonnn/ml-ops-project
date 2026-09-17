from anthropic import Anthropic

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_text = "\n\n".join(chunk["text"] for chunk in chunks)

    prompt = f"Provide an answer to the following question based on the provided context. If the context does not contain relevant information, respond with 'I don't know.'\n\n"
    prompt += f"Context:\n{context_text}\n\n"
    prompt += f"Question: {query}\n\n"
    return prompt

def generate_answer(query: str, chunks: list[dict], client: Anthropic, model: str = "claude-haiku-4-5-20251001") -> str:
    prompt = build_prompt(query, chunks)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
