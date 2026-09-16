# What is RAG, and why does it exist?

## The problem: LLMs don't know your data

A large language model's knowledge is frozen at training time and baked into its weights. Two consequences follow directly from that:

1. **It doesn't know anything after its training cutoff, or anything private** — your own notes, a company's internal docs, this repo's dataset. There's no way to ask it about `data/raw/001_saturn.txt` unless that text is somehow in front of it.
2. **It can't cite where an answer came from.** Ask a plain LLM a factual question and it produces an answer from a statistical blend of everything it saw in training — there is no pointer back to a specific source it can show you. When it's wrong, it's usually still fluent and confident: this is what people call "hallucination," and it's not a bug that gets patched out, it's a structural consequence of how the model generates text (predicting plausible next tokens, not looking anything up).

## Two ways to fix this, and why we picked one

**Option A: fine-tune the model on your data.** Retrain (or partially retrain) the model's weights on your documents so the facts get absorbed into it. This works, but it's expensive, slow to update (add one new document → retrain), and still doesn't solve citation — the model still can't tell you *which* document a fact came from, because the fact is now smeared across millions of weights, not stored anywhere discrete.

**Option B: Retrieval-Augmented Generation (RAG).** Instead of teaching the model your data, hand it the data at question time. At query time:

1. **Retrieve** the handful of chunks of your data most relevant to the question (this repo will do this with embeddings + vector search — see [docs/02](02-embeddings-and-vector-search.md)).
2. **Generate** an answer by giving the model the question *plus* those retrieved chunks as context, and instructing it to answer using only that context.

This is what we're building. It wins on exactly the two points fine-tuning loses on: updating the data is just editing a file (no retraining), and every answer can be traced back to the specific chunk(s) it was grounded in — which is also what makes **evaluation** tractable: we can check not just "was the answer right" but "did it retrieve the right source," which is the backbone of the eval harness we'll build in [docs/05](05-ml-ops-and-eval.md).

## Why this matters for the ML-ops side of this project

RAG turns "does the AI know the right answer" into two separable, individually testable questions:

- **Retrieval quality**: given a question, did we fetch the chunk that actually contains the answer? This is a plain information-retrieval metric (hit-rate, precision@k) — no LLM involved, cheap and deterministic to check.
- **Generation quality**: given the *correct* chunk was retrieved, did the model produce a correct, grounded answer from it (rather than ignoring the context or hallucinating on top of it)?

Splitting the pipeline this way is what makes the CI eval workflow in this repo meaningful rather than a black box: when a regression happens, we'll be able to tell whether retrieval broke (e.g. we changed the chunking strategy and now split an answer across two chunks) or generation broke (e.g. a prompt change made the model ignore the retrieved context), instead of just seeing "eval score went down" with no idea why.

## Where this project's dataset fits in

`data/raw/` holds eight short, self-contained, factual articles on unrelated topics (Saturn, the Eiffel Tower, photosynthesis, etc.). Keeping the topics unrelated is deliberate: it means there's exactly one correct source document per question, so when we measure retrieval quality later, a wrong retrieval is unambiguous rather than a judgment call about which of several similar documents was "close enough." `data/eval/eval_questions.json` pairs each topic with a couple of questions and their known-correct answer and source file — that's the ground truth the eval harness will check the pipeline against.
