from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import requests


class Embeddings(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        pass
    
    def embed_images(
        self, 
        images: List[Union[str, Path, "PIL.Image.Image"]]  # type: ignore[name-defined]  # noqa: F821
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of images.
        
        Args:
            images: List of image paths (str/Path) or PIL Image objects
            
        Returns:
            List of embedding vectors
            
        Raises:
            NotImplementedError: If image embeddings are not supported
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image embeddings"
        )
    
    def embed_multimodal(
        self,
        texts: Optional[List[str]] = None,
        images: Optional[
            List[Union[str, Path, "PIL.Image.Image"]]  # type: ignore[name-defined]  # noqa: F821
        ] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for text, images, or both.
        Returns embeddings in the order: texts (if provided), then images (if provided).
        
        Args:
            texts: Optional list of text strings
            images: Optional list of image paths or PIL Image objects
            
        Returns:
            Combined list of embedding vectors
        """
        results = []
        if texts:
            results.extend(self.embed_texts(texts))
        if images:
            results.extend(self.embed_images(images))
        return results


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


class RemoteModelEmbeddings(Embeddings):
    def __init__(
        self, model: str, api_key: str, base_url: str, port: Optional[int | str] = None
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.port = port

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.base_url.startswith("http"):
            if self.port:
                url = f"{self.base_url}:{self.port}/embeddings"
            else:
                url = f"{self.base_url}/embeddings"
        else:
            url = f"https://{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": self.model}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            response_datas = response.json().get("data", [])
            embeddings = [data.get("embedding", []) for data in response_datas]
            return embeddings
        else:
            response.raise_for_status()

class OllamaEmbeddings(Embeddings):
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama embedding model.

        Args:
            model: Name of the embedding model (e.g., "nomic-embed-text").
            base_url: Base URL of the Ollama server.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Ollama's local API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        url = f"{self.base_url}/api/embeddings"
        embeddings = []

        for text in texts:
            payload = {
                "model": self.model,
                "prompt": text
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                embeddings.append(data["embedding"])
            else:
                raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

        return embeddings


# CLIP embeddings support
try:
    import clip
    import numpy as np
    import torch
    from PIL import Image as PILImage
    
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    clip = None  # type: ignore
    torch = None  # type: ignore
    np = None  # type: ignore
    PILImage = None  # type: ignore


class ClipEmbeddings(Embeddings):
    """CLIP-based multi-modal embeddings for text and images."""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        lazy_load: bool = True,
    ):
        """Initialize CLIP embeddings model.
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-B/16", "ViT-L/14").
            device: Device to run model on ("cuda", "mps", "cpu", or None for auto).
            normalize: Whether to L2-normalize embeddings.
            batch_size: Batch size for image processing.
            lazy_load: Whether to defer model loading until first use.
        """
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP is not installed. Install with: pip install -e \".[multimodal]\" "
                "(local) or pip install \"alith[multimodal]\" (PyPI)"
            )
        
        self.model_name = model_name
        self.device_str = device
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        
        if not lazy_load:
            self._ensure_loaded()
    
    @property
    def device(self) -> "torch.device":
        """Auto-select device (cuda > mps > cpu)."""
        if self.device_str:
            return torch.device(self.device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _ensure_loaded(self) -> None:
        """Load CLIP model if not already loaded."""
        if self._model is not None and self._preprocess is not None:
            return
        
        self._model, self._preprocess = clip.load(
            self.model_name, 
            device=self.device
        )
        self._model = self._model.eval()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using CLIP text encoder."""
        self._ensure_loaded()
        
        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            embeddings = self._model.encode_text(text_tokens)
            
            if self.normalize:
                norm = embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                embeddings = embeddings / norm
            
            return [emb.cpu().float().numpy() for emb in embeddings]
    
    def embed_images(
        self, 
        images: List[Union[str, Path, "PILImage.Image"]]
    ) -> List[List[float]]:
        """Embed images using CLIP image encoder."""
        self._ensure_loaded()
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i + self.batch_size]
                pil_images = []
                
                for img in batch_images:
                    if isinstance(img, (str, Path)):
                        with PILImage.open(img) as opened_img:
                            pil_images.append(opened_img.copy().convert("RGB"))
                    elif isinstance(img, PILImage.Image):
                        pil_images.append(img.convert("RGB"))
                    else:
                        raise TypeError(f"Unsupported image type: {type(img)}")
                
                preprocessed = [self._preprocess(img) for img in pil_images]
                image_tensors = torch.stack(preprocessed).to(self.device)
                embeddings = self._model.encode_image(image_tensors)
                
                if self.normalize:
                    norm = embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    embeddings = embeddings / norm
                
                batch_embeds = [emb.cpu().float().numpy() for emb in embeddings]
                all_embeddings.extend(batch_embeds)
        
        return all_embeddings