import hashlib
import shutil
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

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
    from pymilvus import MilvusClient, MilvusException, model

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
                "python3 -m pip install pymilvus pymilvus[model]"
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

    def save_docs(
        self, docs: List[str], collection_name: Optional[str] = None
    ) -> "MilvusStore":
        vectors = self.embedding_fn.encode_documents(docs)
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        self.client.insert(
            collection_name=collection_name or self.collection_name, data=data
        )
        return self

    def reset(self):
        self.client.drop_collection(self.collection_name)

    def search_in(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        query_vectors = self.embedding_fn.encode_documents([query])
        results = self.client.search(
            collection_name=collection_name or self.collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=["text"],
        )
        docs = [d["entity"]["text"] for r in results for d in r]
        return docs

    def has_collection(self, collection_name: str) -> bool:
        """Check if the collection exists."""
        try:
            return self.client.has_collection(collection_name)
        except MilvusException:
            return False

    def create_collection(self, collection_name: str) -> "MilvusStore":
        """Create a new collection."""
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dimension,
        )
        return self


try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSStore(Store):
    """FAISS vector store implementation."""
    
    def __init__(
        self,
        dimension: int = 768,
        embeddings: Optional[Embeddings] = None,
        index_type: str = "L2"
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss is not installed. Please install it with: "
                "python3 -m pip install faiss-cpu  # for CPU-only version\n"
                "# or\n"
                "python3 -m pip install faiss-gpu  # for GPU support"
            )
        
        self.dimension = dimension
        self.embeddings = embeddings
        self.index_type = index_type
        
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError("index_type must be either 'L2' or 'IP'")
            
        self.texts: List[str] = []
        
    # ---------------------------------------------------------------------
    # Search methods
    # ---------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4
    ) -> List[str]:
        """Search for similar documents using FAISS with optimized performance."""
        if not self.texts:
            return []
            
        if self.embeddings:
            query_embedding = self.embeddings.embed_texts([query])[0]
        else:
            raise ValueError("Embeddings must be provided for search")
            
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, min(limit * 2, len(self.texts)))
        
        results = []
        results_append = results.append
        
        for i, distance in zip(indices[0], distances[0]):
            if i == -1: 
                continue
                
            if distance == 0.0:
                score = 1.0
            else:
                score = 1.0 / (1.0 + distance)
            
            if score >= score_threshold:
                results_append(self.texts[i])
                if len(results) >= limit:
                    break
                    
        return results

    def search_batch(
        self,
        queries: List[str],
        limit: int = 3,
        score_threshold: float = 0.4
    ) -> List[List[str]]:
        """Batch search for multiple queries at once - much more efficient."""
        if not self.texts or not queries:
            return []
            
        if self.embeddings:
            query_embeddings = self.embeddings.embed_texts(queries)
        else:
            raise ValueError("Embeddings must be provided for search")
            
        query_vectors = np.array(query_embeddings, dtype=np.float32)
        
        distances, indices = self.index.search(query_vectors, min(limit * 2, len(self.texts)))
        
        all_results = []
        for query_idx in range(len(queries)):
            results = []
            results_append = results.append
            
            for i, distance in zip(indices[query_idx], distances[query_idx]):
                if i == -1:
                    continue
                    
                if distance == 0.0:
                    score = 1.0
                else:
                    score = 1.0 / (1.0 + distance)
                
                if score >= score_threshold:
                    results_append(self.texts[i])
                    if len(results) >= limit:
                        break
                        
            all_results.append(results)
            
        return all_results

    def search_with_scores(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4
    ) -> List[tuple]:
        """Search and return results with similarity scores."""
        if not self.texts:
            return []
            
        if self.embeddings:
            query_embedding = self.embeddings.embed_texts([query])[0]
        else:
            raise ValueError("Embeddings must be provided for search")
            
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, min(limit * 2, len(self.texts)))
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i == -1:
                continue
                
            if distance == 0.0:
                score = 1.0
            else:
                score = 1.0 / (1.0 + distance)
            
            if score >= score_threshold:
                results.append((self.texts[i], score))
                if len(results) >= limit:
                    break
                    
        return results

    def search_approximate(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4,
        nprobe: int = 10
    ) -> List[str]:
        """Approximate search using IVF index for very large datasets."""
        if not self.texts:
            return []
            
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = nprobe
            
        if self.embeddings:
            query_embedding = self.embeddings.embed_texts([query])[0]
        else:
            raise ValueError("Embeddings must be provided for search")
            
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, min(limit * 2, len(self.texts)))
        
        results = []
        for i, distance in zip(indices[0], distances[0]):
            if i == -1:
                continue
                
            if distance == 0.0:
                score = 1.0
            else:
                score = 1.0 / (1.0 + distance)
            
            if score >= score_threshold:
                results.append(self.texts[i])
                if len(results) >= limit:
                    break
                    
        return results

    def create_ivf_index(self, nlist: int = 100) -> None:
        """Create an IVF index for better performance on large datasets."""
        if len(self.texts) == 0:
            return
            
        if self.embeddings:
            all_embeddings = self.embeddings.embed_texts(self.texts)
        else:
            raise ValueError("Embeddings must be provided")
            
        vectors = np.array(all_embeddings, dtype=np.float32)
        
        quantizer = faiss.IndexFlatL2(self.dimension) if self.index_type == "L2" else faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        self.index.train(vectors)
        self.index.add(vectors)

    # ---------------------------------------------------------------------
    # Store interface implementation
    # ---------------------------------------------------------------------

    def save(self, value: str) -> None:
        """Save a single document to the store."""
        self.save_docs([value])

    def save_docs(self, docs: List[str]) -> "FAISSStore":
        """Save multiple documents to the store."""
        if not docs:
            return self
            
        if self.embeddings:
            embeddings = self.embeddings.embed_texts(docs)
        else:
            raise ValueError("Embeddings must be provided for saving documents")
            
        vectors = np.array(embeddings).astype('float32')
        
        self.index.add(vectors)
        
        self.texts.extend(docs)
        
        return self

    def reset(self) -> None:
        """Reset the store by clearing all stored data."""
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []

    def has_collection(self, collection_name: str = None) -> bool:
        """Check if the collection exists. For FAISS, this always returns True since we use a single index."""
        return True

    def create_collection(self, collection_name: str = None) -> "FAISSStore":
        """Create a new collection. For FAISS, this resets the current index."""
        self.reset()
        return self

    def search_in(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.4,
        collection_name: str = None
    ) -> List[str]:
        """Search in a specific collection. For FAISS, this is the same as search since we use a single index."""
        return self.search(query, limit, score_threshold)

    def get_stats(self) -> dict:
        """Get statistics about the FAISS store."""
        return {
            "total_documents": len(self.texts),
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0,
            "dimension": self.dimension,
            "index_type": self.index_type
        }

    def save_to_disk(self, path: str) -> None:
        """Save the FAISS index and texts to disk."""
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        faiss.write_index(self.index, f"{path}.index")
        
        with open(f"{path}.json", 'w') as f:
            json.dump(self.texts, f)

    def load_from_disk(self, path: str) -> None:
        """Load the FAISS index and texts from disk."""
        import json
        import os
        
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
            
        if os.path.exists(f"{path}.json"):
            with open(f"{path}.json", 'r') as f:
                self.texts = json.load(f)
