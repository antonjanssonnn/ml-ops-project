# ml-ops-project

A small, local, public learning project: building a Retrieval-Augmented Generation (RAG) system from scratch, with real ML-ops practices (CI, evaluation, regression checks) around it — kept intentionally tiny so every piece stays understandable.

**Status:** work in progress, built incrementally as a learning exercise. See [docs/](docs/) for the theory behind each design choice.

## What this is

A CLI tool that answers questions using a small checked-in dataset as its only source of truth, via retrieval (find relevant chunks) + generation (ask an LLM to answer using those chunks). Alongside it, a GitHub Actions pipeline that evaluates the pipeline's actual answer quality on every push — not just that the code runs, but that it still gives correct answers.

## Why

Built to learn RAG and ML-ops by doing, with an emphasis on understanding *why* each piece exists, not just copying a tutorial. See [docs/](docs/) for write-ups on each concept as they're introduced.

## Theory

- [What is RAG, and why does it exist?](docs/01-what-is-rag.md)

## Dataset

`data/raw/` holds eight short, unrelated factual articles used as the knowledge base. `data/eval/eval_questions.json` holds a fixed set of questions with known-correct answers and source documents, used to evaluate the pipeline (see the eval theory doc, coming in a later milestone).

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Usage

_(coming in a later milestone)_

## Project structure

_(coming in a later milestone — see the build log in commit history for now)_
