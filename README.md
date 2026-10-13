# Agentic AI for Drug Safety & Discovery

A multi-agent system that runs **entirely on a local LLM** (Ollama) against **real FDA and PubMed data**. It is
implemented twice, in **LangChain** and in **Pydantic AI**, with identical logic.

| Part | Question | How |
|---|---|---|
| **A: Drug Safety** | *Is drug X linked to adverse event Y?* | FAERS disproportionality (ROR, PRR, 95% CI, χ²) → LLM statistician + clinical reviewer → **rule-based QC agent that sends targeted feedback** until the analysis passes → risk level and actions |
| **B: Drug Discovery** | *Which existing drugs could treat disease Z?* | **Agentic RAG** over PubMed: local embeddings, self-graded retrieval, **LLM query rewriting when the evidence is weak**, structured extraction, PMID guardrail, transparent scoring |
| **C: Safety-aware discovery** | *Are the candidates safe enough to pursue?* | Each candidate is screened against FAERS; a writer agent produces a cited brief |

## Highlights
- **Fully local**: chat model and embeddings on Ollama, so no data leaves the machine.
- **Structured outputs**: every LLM answer is a validated Pydantic object.
- **Deterministic where it matters**: statistics, QC rules, scores and risk matrix are plain, testable Python.
- **Guardrails**: citations must exist in the retrieved evidence, and evidence strength must match the literature count.
- **Audit trail**: every agent action is appended to `runs/<run_id>/audit.jsonl`; API responses are cached for reproducibility.

## Quickstart
```bash
ollama pull nemotron-3.5-lightning:30b-mlx
ollama pull nomic-embed-text
OLLAMA_CONTEXT_LENGTH=16384 ollama serve        # needed for the Pydantic AI notebook
pip install langchain-core langchain-ollama "pydantic-ai-slim[openai]" openai requests numpy pandas python-dotenv
```
Open either notebook and run all cells. Optional `.env`: `ENTREZ_EMAIL`, `NCBI_API_KEY`.

| Notebook | Framework |
|---|---|
| `drug_safety_discovery_langchain.ipynb` | LangChain + `ChatOllama` / `OllamaEmbeddings` |
| `drug_safety_discovery_pydantic_ai.ipynb` | Pydantic AI + `OllamaProvider` |

## Results
_Add your own run here: QC score before → after the feedback loop, retrieval confidence before → after query
rewriting, and the top repurposing candidates with their FAERS profile._

## Background
This project grew out of the capstone projects of Udacity's *Agentic AI for Life Sciences* program. It was redesigned
as a single self-contained system and re-implemented in two frameworks on a local model.

> Research/educational project. Not for clinical or regulatory decisions. FAERS counts reflect reporting volume,
> not incidence, and do not establish causation.
