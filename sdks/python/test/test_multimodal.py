"""Comprehensive tests for multimodal features."""

import os
import pytest

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check for required multimodal dependencies
try:
    import clip
    import torch
    from PIL import Image
    import faiss
except ImportError as e:
    missing = getattr(e, 'name', 'multimodal dependencies')
    pytest.skip(
        f"{missing} not available. Install with: pip install -e \".[multimodal]\" "
        "(local) or pip install \"alith[multimodal]\" (PyPI)",
        allow_module_level=True
    )

try:
    from alith import (
        Agent,
        MultimodalAgent,
        ClipEmbeddings,
        ImageFAISSStore,
        FAISSStore,
    )
except ImportError:
    pytest.skip("Multimodal features not available", allow_module_level=True)


# Agent initialization (from agent_with_multimodal_embeddings.py)
# Load API key from environment, use test key if not set (for CI/local testing)
api_key = os.getenv("GROQ_API_KEY") or "test-key-for-unit-tests"
AGENT_CONFIG = {
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "api_key": api_key,
    "base_url": "https://api.groq.com/openai/v1",
}


@pytest.fixture
def embeddings():
    """Create ClipEmbeddings instance."""
    try:
        return ClipEmbeddings(lazy_load=False)
    except ImportError:
        pytest.skip("CLIP not installed")


@pytest.fixture
def store(embeddings):
    """Create ImageFAISSStore instance."""
    return ImageFAISSStore(embeddings=embeddings, dimension=512)


@pytest.fixture
def test_image(tmp_path):
    """Create a test image file."""
    img_path = tmp_path / "test.png"
    Image.new("RGB", (224, 224), color="red").save(img_path)
    return img_path


@pytest.fixture
def agent(store):
    """Create MultimodalAgent with store."""
    return MultimodalAgent(store=store, **AGENT_CONFIG)


# ClipEmbeddings Tests
def test_clip_embeddings_initialization(embeddings):
    """Test ClipEmbeddings initialization."""
    assert embeddings.model_name == "ViT-B/32"
    assert embeddings.normalize is True


def test_embed_texts(embeddings):
    """Test text embedding."""
    texts = ["hello", "world"]
    result = embeddings.embed_texts(texts)
    assert len(result) == 2
    assert len(result[0]) == 512


def test_embed_images(embeddings, test_image):
    """Test image embedding."""
    result = embeddings.embed_images([str(test_image)])
    assert len(result) == 1
    assert len(result[0]) == 512


def test_embed_multimodal(embeddings, test_image):
    """Test multimodal embedding."""
    result = embeddings.embed_multimodal(
        texts=["hello"], images=[str(test_image)]
    )
    assert len(result) == 2
    assert all(len(emb) == 512 for emb in result)


# MultimodalAgent Tests
def test_multimodal_agent_inheritance():
    """Test MultimodalAgent inherits from Agent."""
    agent = MultimodalAgent(**AGENT_CONFIG)
    assert isinstance(agent, Agent)


def test_encode_image(agent, test_image):
    """Test image encoding."""
    encoded = agent._encode_image(str(test_image))
    assert encoded.startswith("data:image/png;base64,")


def test_agent_with_store(agent, store, test_image):
    """Test agent with store integration."""
    store.save(str(test_image))
    assert agent.store is not None


# ImageFAISSStore Tests
def test_store_inheritance(store):
    """Test ImageFAISSStore inherits from FAISSStore."""
    assert isinstance(store, FAISSStore)


def test_save_image_and_text(store, test_image):
    """Test saving images and text."""
    store.save(str(test_image))
    store.save("test document")
    assert len(store.image_paths) == 1
    assert len(store.texts) == 1


def test_search_mixed(store, test_image):
    """Test searching mixed content."""
    store.save(str(test_image))
    store.save("document about images")
    results = store.search("image", limit=5)
    assert len(results) > 0


def test_store_reset(store, test_image):
    """Test store reset."""
    store.save(str(test_image))
    store.save("text")
    store.reset()
    assert len(store.image_paths) == 0
    assert len(store.texts) == 0


# Integration & Backward Compatibility Tests
def test_original_agent_still_works():
    """Test original Agent still works."""
    agent = Agent(model="test", api_key="key", base_url="url")
    assert agent.model == "test"
    assert not isinstance(agent, MultimodalAgent)


def test_faiss_store_still_works(embeddings):
    """Test FAISSStore backward compatibility with text embeddings.
    
    Verifies that the base FAISSStore class still works correctly with text-only
    embeddings, ensuring multimodal features don't break existing functionality.
    Uses ClipEmbeddings which supports both text and images.
    """
    store = FAISSStore(embeddings=embeddings, dimension=512)
    store.save("test document")
    assert len(store.texts) == 1


def test_cross_modal_search(store, test_image):
    """Test cross-modal search."""
    store.save(str(test_image))
    results = store.search("red color image", limit=1)
    assert len(results) > 0

