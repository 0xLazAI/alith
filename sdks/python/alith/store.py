import shutil
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from .embeddings import Embeddings


class Store(ABC):
    """Abstract base class for a storage backend."""

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4,
    ) -> List[str]:
        """Searches the storage with a query, limiting the results and applying a threshold."""
        pass

    @abstractmethod
    def save(self, value: str) -> None:
        """Saves a value into the storage."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the storage by clearing all stored data."""
        pass


try:
    import chromadb
    import chromadb.errors
    from chromadb import EmbeddingFunction
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True

except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaDBStore(Store):
    path: str = "."
    collection_name: str = "alith"
    embeddings: Optional[Embeddings] = None

    def __init__(
        self,
        path: str = ".",
        collection_name: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is not installed. Please install it with: "
                "python3 -m pip install chromadb"
            )
        self.embeddings = embeddings
        if collection_name:
            self.collection_name = collection_name
        self.path = path
        self.app = chromadb.PersistentClient(
            path=self.path,
            settings=Settings(allow_reset=True),
        )

        class CustomEmbeddingFunction(EmbeddingFunction):
            def __call__(self, texts):
                # embed the documents somehow
                return embeddings.embed_texts(texts)

        from chromadb.utils import embedding_functions

        default_ef = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.app.get_or_create_collection(
            name=self.collection_name,
            embedding_function=(
                CustomEmbeddingFunction() if self.embeddings else default_ef
            ),
        )

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.4
    ) -> List[str]:
        if self.collection:
            fetched = self.collection.query(
                query_texts=[query],
                n_results=limit,
            )
            results = []
            for i in range(len(fetched["ids"][0])):  # type: ignore
                result = {
                    "id": fetched["ids"][0][i],  # type: ignore
                    "metadata": fetched["metadatas"][0][i],  # type: ignore
                    "context": fetched["documents"][0][i],  # type: ignore
                    "score": fetched["distances"][0][i],  # type: ignore
                }
                if result["score"] >= score_threshold:
                    results.append(result)
            results = [result["context"] for result in results]
            return results
        else:
            raise Exception("Collection not initialized")

    def save(self, value: str):
        documents = [value]
        ids = [hashlib.sha256(value.encode("utf-8")).hexdigest()]
        metadatas = [None]
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def save_docs(self, docs: List[str]) -> "ChromaDBStore":
        ids = [hashlib.sha256(doc.encode("utf-8")).hexdigest() for doc in docs]
        metadatas = [None] * len(docs)
        self.collection.upsert(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
        )
        return self

    def reset(self):
        if not self.app:
            self.app = chromadb.PersistentClient(
                path=self.path,
                settings=Settings(allow_reset=True),
            )
        self.app.reset()
        shutil.rmtree(self.path)
        self.app = None
        self.collection = None


try:
    from pymilvus import MilvusClient, model

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class MilvusStore(Store):
    uri: str = "alith.db"
    dimension: int = 768
    collection_name: str = "alith"
    embeddings: Optional[Embeddings]
    embedding_fn: Callable[[List[str]], List[List[float]]]

    def __init__(
        self,
        uri: str = "alith.db",
        dimension: int = 768,
        collection_name: str = "alith",
        embeddings: Optional[Embeddings] = None,
    ):
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Please install it with: "
                "python3 -m pip install pymilvus"
            )
        self.uri = uri
        self.dimension = dimension
        self.collection_name = collection_name
        self.embeddings = embeddings
        if self.embeddings:
            self.embeddings.encode_documents = self.embeddings.embed_texts
            self.embedding_fn = self.embeddings
        else:
            # If connection to https://huggingface.co/ failed, uncomment the following path.
            # import os
            # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            self.model = model.DefaultEmbeddingFunction()
            self.embedding_fn = self.model
        self.client = MilvusClient("alith.db")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
        )

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.4
    ) -> List[str]:
        query_vectors = self.embedding_fn.encode_documents([query])
        results = self.client.search(
            collection_name=self.collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=["text"],
        )
        docs = [d["entity"]["text"] for r in results for d in r]
        return docs

    def save(self, value: str):
        self.save_docs([value])

    def save_docs(self, docs: List[str]) -> "MilvusStore":
        vectors = self.embedding_fn.encode_documents(docs)
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
        return self

    def reset(self):
        self.client.drop_collection(self.collection_name)
