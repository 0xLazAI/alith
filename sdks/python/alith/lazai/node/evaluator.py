from __future__ import annotations

from typing import Literal, Union
from pydantic import BaseModel
from abc import ABC, abstractmethod
import numpy as np
from alith.data.evaluator.text import TextEvaluator

# Data type definition
DataType = Literal["text", "image", "structured"]
evaluator = TextEvaluator()


class DQSWeights(BaseModel):
    w1: float = 0.2  # Weight for Duplication Score
    w2: float = 0.3  # Weight for Accuracy Score
    w3: float = 0.5  # Weight for Context Alignment Score


class DataEvaluator(ABC):
    """
    Interface for data evaluators, defining specifications for various score calculations
    """

    @abstractmethod
    def evaluate_duplication(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Evaluate data duplication (0-1 scale, lower value indicates more duplicates)
        """
        pass

    @abstractmethod
    def evaluate_accuracy(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Evaluate data accuracy (0-1 scale, higher value indicates better quality)
        """
        pass

    @abstractmethod
    def evaluate_context_alignment(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Evaluate context alignment with model objectives (0-1 scale, higher value indicates better alignment)
        """
        pass


class DQSCalculator(ABC):
    """
    Interface for DQS (Data Quality Score) calculator
    """

    def __init__(self, evaluator: DataEvaluator, weights: DQSWeights):
        self.evaluator = evaluator
        self.weights = weights

    @abstractmethod
    def calculate(self, content: Union[bytes, str], data_type: DataType) -> float:
        """
        Calculate the final DQS score
        """
        pass


class MockDataEvaluator(DataEvaluator):
    """
    Mock data evaluator implementing the DataEvaluator interface with simulated logic
    """

    def evaluate_duplication(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate duplication evaluation based on LazAI's technical design:
        - Text: Cosine similarity of embeddings (penalizes similarity >= 0.95)
        - Images: Perceptual hash similarity (penalizes similarity >= 0.7)
        - Structured data: Hash-based duplicate rate
        """
        if data_type == "text":
            # Simulate text embedding similarity check
            similarity = np.random.uniform(0, 0.9)
            return 1.0 if similarity < 0.95 else 0.3  # Penalize high similarity

        elif data_type == "image":
            # Simulate perceptual hash similarity check
            hash_similarity = np.random.uniform(0, 0.8)
            return 1.0 if hash_similarity < 0.7 else 0.2  # Penalize similar images

        else:  # structured
            # Simulate hash-based duplicate detection
            duplicate_rate = np.random.uniform(0, 0.1)
            return 1.0 - duplicate_rate  # Lower duplicates = higher score

    def evaluate_accuracy(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate accuracy evaluation based on LazAI's technical design:
        - Text: Perplexity score (lower is better)
        - Images: Resolution and label consistency
        - Structured data: Completeness and format correctness
        """
        if data_type == "text":
            # Simulate perplexity calculation (lower = better)
            perplexity = np.random.uniform(5, 50)
            return max(0.0, min(1.0, 1 - (perplexity - 5) / 45))

        elif data_type == "image":
            # Simulate resolution and label consistency
            resolution_score = np.random.uniform(0.7, 1.0)
            label_consistency = np.random.uniform(0.6, 1.0)
            return (resolution_score + label_consistency) / 2

        else:  # structured
            # Simulate data completeness and format correctness
            completeness = np.random.uniform(0.8, 1.0)
            format_correctness = np.random.uniform(0.9, 1.0)
            return (completeness + format_correctness) / 2

    def evaluate_context_alignment(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate context alignment evaluation based on LazAI's technical design:
        - Training phase: Impact on loss function reduction
        - Inference phase: Precision, recall, F1-score, and prediction confidence
        """
        if data_type == "text":
            # Simulate training contribution and inference metrics
            training_contribution = np.random.uniform(0.7, 1.0)
            inference_f1 = np.random.uniform(0.8, 1.0)
            return (training_contribution + inference_f1) / 2

        elif data_type == "image":
            # Simulate model alignment for images
            loss_reduction = np.random.uniform(0.6, 1.0)
            prediction_confidence = np.random.uniform(0.8, 1.0)
            return (loss_reduction + prediction_confidence) / 2

        else:  # structured
            # Simulate alignment with model objectives
            importance_prob = np.random.uniform(0.7, 1.0)
            f1_score = np.random.uniform(0.85, 1.0)
            return (importance_prob + f1_score) / 2


class StandardDataEvaluator(DataEvaluator):
    def evaluate_duplication(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate duplication evaluation based on LazAI's technical design:
        - Text: Cosine similarity of embeddings (penalizes similarity >= 0.95)
        - Images: Perceptual hash similarity (penalizes similarity >= 0.7)
        - Structured data: Hash-based duplicate rate
        """
        if data_type == "text":
            # Simulate text embedding similarity check
            similarity = np.random.uniform(0, 0.9)
            return 1.0 if similarity < 0.95 else 0.3  # Penalize high similarity
        else:
            raise NotImplementedError()

    def evaluate_accuracy(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate accuracy evaluation based on LazAI's technical design:
        - Text: Perplexity score (lower is better)
        - Images: Resolution and label consistency
        - Structured data: Completeness and format correctness
        """
        if data_type == "text":
            return evaluator.evaluate_accuracy(
                content.decode("utf-8") if isinstance(content, bytes) else content
            )
        else:
            raise NotImplementedError()

    def evaluate_context_alignment(
        self, content: Union[bytes, str], data_type: DataType
    ) -> float:
        """
        Simulate context alignment evaluation based on LazAI's technical design:
        - Training phase: Impact on loss function reduction
        - Inference phase: Precision, recall, F1-score, and prediction confidence
        """
        if data_type == "text":
            # Simulate training contribution and inference metrics
            training_contribution = np.random.uniform(0.7, 1.0)
            inference_f1 = np.random.uniform(0.8, 1.0)
            return (training_contribution + inference_f1) / 2
        else:
            raise NotImplementedError()


class StandardDQSCalculator(DQSCalculator):
    """
    Standard DQS calculator implementing LazAI's weighted scoring formula:
    `S = w1*DS + w2*AS + w3*CAS`
    """

    def __init__(self, evaluator=StandardDataEvaluator(), weights=DQSWeights()):
        super().__init__(evaluator=evaluator, weights=weights)

    def calculate(self, content: Union[bytes, str], data_type: DataType) -> float:
        ds = self.evaluator.evaluate_duplication(content, data_type)
        as_score = self.evaluator.evaluate_accuracy(content, data_type)
        cas = self.evaluator.evaluate_context_alignment(content, data_type)

        # Apply the weighted formula
        return self.weights.w1 * ds + self.weights.w2 * as_score + self.weights.w3 * cas
