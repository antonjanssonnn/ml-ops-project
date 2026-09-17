from ml_ops_project.generate import build_prompt

def test_build_prompt_includes_context_and_question():
    chunks = [{"text": "Saturn is a planet."}, {"text": "The Eiffel Tower is in Paris."}]
    prompt = build_prompt("What is Saturn?", chunks)

    assert "Saturn is a planet." in prompt
    assert "The Eiffel Tower is in Paris." in prompt
    assert "What is Saturn?" in prompt
