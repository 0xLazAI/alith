# Alith Python SDK

Note: See the consolidated SDK overview and Rust binding build guide in `sdks/README.md`.

## Installation

```shell
python3 -m pip install alith
```

## Quick Start

- Simple Agent

```python
from alith import Agent

agent = Agent(
    model="gpt-4",
    preamble="You are a comedian here to entertain the user using humour and jokes.",
)
print(agent.prompt("Entertain me!"))
```

- Agent with Tools

```python
from alith import Agent


def sum(x: int, y: int) -> int:
    """Add x and y together"""
    x + y


def sub(x: int, y: int) -> int:
    """Subtract y from x (i.e.: x - y)"""
    x + y


agent = Agent(
    name="Calculator Agent",
    model="gpt-4o-mini",
    preamble="You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.",
    tools=[sum, sub],
)
print(agent.prompt("Calculate 10 - 3"))
```

## Examples

See [here](./examples/README.md) for more examples.

## Multimodal Embeddings (CLIP)

The Python SDK includes CLIP-based multimodal embeddings for both text and images:
- Text embeddings using CLIP
- Image embeddings using CLIP
- Cross-modal search (search images with text queries)
- MultimodalAgent with image support

Install optional dependencies:

```shell
python -m pip install "alith[multimodal]"
```

### Quick Example

```python
from alith import ClipEmbeddings, ImageFAISSStore, MultimodalAgent

# Create embeddings and store
embeddings = ClipEmbeddings()
store = ImageFAISSStore(embeddings=embeddings, dimension=512)

# Save images and text
store.save("path/to/image.png")
store.save("document text")

# Search across modalities
results = store.search("red logo", limit=5)

# Use with MultimodalAgent
agent = MultimodalAgent(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key="your-api-key",
    base_url="https://api.groq.com/openai/v1",
    store=store
)

response = agent.prompt("Describe this image", images=["path/to/image.png"])
```

### Example Script

See [examples/agent_with_multimodal_embeddings.py](./examples/agent_with_multimodal_embeddings.py) for a complete example.

**Run the example:**
```shell
cd sdks/python
# activate your venv first (see "Developing" above), then:
python examples/agent_with_multimodal_embeddings.py
```

**Notes:**
- The first run will download the CLIP model (~150MB for ViT-B/32)
- **System requirements**: Works on systems with 2GB+ free RAM
- CPU works fine, but a GPU (CUDA/MPS) will speed up inference significantly
- Embedding dimension is 512 for CLIP ViT-B/32 model
- Supports common image formats: PNG, JPG, JPEG, GIF, BMP, WEBP

If you didn't install the optional multimodal extras, install them first:

```shell
pip install "alith[multimodal]"
```

## Data Evaluation (Text)

The Python SDK includes multilingual text evaluators built on XLM-RoBERTa for:
- Perplexity/plausibility scoring
- Semantic similarity (cosine over sentence embeddings)
- Basic grammar/spell checks (English) and punctuation heuristics (CJK)
- A combined accuracy score with optional reference text and lightweight fact checks

### Two Versions Available

| Version | Model | Memory | Best For |
|---------|-------|--------|----------|
| **Lightweight** | xlm-roberta-base | ~1-2GB | Testing, limited RAM systems |
| **Standard** | xlm-roberta-large | ~6-8GB | Production, maximum accuracy |

Install optional dependencies:

```shell
python -m pip install "alith[evaluation]"
```

### Lightweight Version (Recommended for Most Users)

**Memory-efficient** version using xlm-roberta-base:

```python
# Use the standalone lightweight version (no package install needed)
from text_evaluator_lightweight import LightweightTextEvaluator

evaluator = LightweightTextEvaluator()  # Uses ~1-2GB RAM

text = "The quick brown fox jumps over the lazy dog."
reference = "A fox jumps over a dog in a simple sentence."

accuracy = evaluator.evaluate_accuracy(text, reference_text=reference)
similarity = evaluator.evaluate_similarity(text, reference)
grammar = evaluator.evaluate_grammar(text)

print({
    "accuracy": accuracy,
    "similarity": similarity,
    "grammar": grammar,
})
```

### Standard Version (High Accuracy)

**High-accuracy** version using xlm-roberta-large (requires 6-8GB RAM):

```python
from alith.data.evaluator.text import TextEvaluator

evaluator = TextEvaluator()  # Requires 6-8GB RAM

text = "The quick brown fox jumps over the lazy dog."
reference = "A fox jumps over a dog in a simple sentence."

perplexity = evaluator.evaluate_perplexity(text)
grammar = evaluator.evaluate_grammar(text)
similarity = evaluator.evaluate_similarity(text, reference)
accuracy = evaluator.evaluate_accuracy(
    text,
    reference_text=reference,
    fact_knowledge={"fox": "a small to medium-sized omnivorous mammal"},
)

print({
    "perplexity": perplexity,
    "grammar": grammar,
    "similarity": similarity,
    "accuracy": accuracy,
})
```

Notes:
- The first run will download the model from Hugging Face:
  - **Lightweight**: ~1.12GB (xlm-roberta-base)
  - **Standard**: ~2.24GB (xlm-roberta-large)
- **System requirements**: 
  - **Lightweight**: Works on systems with 2GB+ free RAM ✅
  - **Standard**: Requires **at least 6-8GB of available RAM**. If the script crashes silently during initialization, your system may not have enough memory.
- CPU works fine, but a GPU (CUDA) will speed up inference significantly.
- Scores are in [0, 1]; higher is better unless otherwise noted in docstrings.
- If you encounter memory issues on Windows, use the lightweight version or ensure no other memory-intensive applications are running.

Example script
---------------

We include ready-to-run examples:

**Lightweight version** (recommended, works on any system):
```powershell
cd sdks/python
python examples\text_evaluator_lightweight.py
```

**Standard version** (requires 8GB+ free RAM):
```powershell
cd sdks/python
# activate your venv first (see "Developing" above), then:
python examples\text_evaluator.py
```

See [MODEL_COMPARISON.md](./examples/MODEL_COMPARISON.md) for detailed comparison and usage guide.

If you didn't install the optional evaluation extras, install them first:

```powershell
pip install "alith[evaluation]"
```

## Developing

Setup virtualenv:

```shell
python3 -m venv venv
```

Activate venv:

```shell
source venv/bin/activate
```

Install maturin:

```shell
cargo install maturin
```

Build bindings:

```shell
maturin develop
```

Test

```shell
python3 -m pip install pytest
python3 -m pytest
```

Lint

```shell
python3 -m pip install ruff
python3 -m ruff check
```
