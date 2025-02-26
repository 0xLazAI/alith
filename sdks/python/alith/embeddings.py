from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path


class Embeddings(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[float]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        pass


try:
    from fastembed_gpu import TextEmbedding  # type: ignore

    FASTEMBED_AVAILABLE = True
except ImportError:
    try:
        from fastembed import TextEmbedding

        FASTEMBED_AVAILABLE = True
    except ImportError:
        FASTEMBED_AVAILABLE = False


class FastEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the model to use
            cache_dir: Directory to cache the model
            gpu: Whether to use GPU acceleration
        """
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "FastEmbed is not installed. Please install it with: "
                "python3 -m pip install fastembed or python3 -m pip install fastembed-gpu for GPU support"
            )

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = list(self.model.embed(texts))
        return embeddings


try:
    from pymilvus import model

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class MilvusEmbeddings(Embeddings):
    def __init__(self):
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Please install it with: "
                "python3 -m pip install pymilvus pymilvus[model]"
            )
        self.model = model.DefaultEmbeddingFunction()
        self.embedding_fn = self.model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_fn.encode_documents(texts)
