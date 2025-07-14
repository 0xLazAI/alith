import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, XLMRobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import re


class TextEvaluator:
    def __init__(self):
        # Initialize XLM-RoBERTa-large model (supports 100+ languages)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        self.mlm_model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large")
        self.encoder_model = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.mlm_model.eval()
        self.encoder_model.eval()

        # Simple language detector patterns, TODO: add more language patterns.
        # For production, consider using dedicated libraries like langdetect
        self.language_patterns = {
            "chinese": r"[\u4e00-\u9fff]",
            "japanese": r"[\u3040-\u309f\u30a0-\u30ff]",
            "korean": r"[\uac00-\ud7af\u1100-\u11ff]",
            "english": r"[a-zA-Z]",
        }

    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        Note: Use more robust libraries like langdetect in production.
        """
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, text):
                return lang
        return "other"

    def evaluate_perplexity(self, text: str) -> float:
        """
        Calculate text perplexity using masked language modeling.
        Lower scores indicate more plausible text.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < 2:
            return 1.0  # Very short text is considered plausible

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        total_log_prob = 0.0

        # Mask each token sequentially (skip special tokens)
        for i in range(1, len(tokens) - 1):
            masked_input_ids = input_ids.copy()
            masked_input_ids[i] = self.tokenizer.mask_token_id

            with torch.no_grad():
                outputs = self.mlm_model(torch.tensor([masked_input_ids]))
                predictions = outputs.logits[0, i]

            true_token_id = input_ids[i]
            probs = torch.nn.functional.softmax(predictions, dim=-1)
            token_prob = probs[true_token_id].item()
            token_prob = max(token_prob, 1e-10)  # Avoid log(0)
            total_log_prob += torch.log(torch.tensor(token_prob)).item()

        # Convert average log probability to perplexity and normalize to 0-1
        avg_log_prob = total_log_prob / (len(tokens) - 2)
        perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()
        max_perplexity = 100  # Adjustable threshold
        score = max(0.0, 1.0 - (min(perplexity, max_perplexity) / max_perplexity))
        return score

    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        """
        Get sentence embedding using XLM-RoBERTa's [CLS] token representation.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.encoder_model(**inputs)

        # Use [CLS] token as sentence embedding
        return outputs.last_hidden_state[:, 0, :].numpy()

    def evaluate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using cosine similarity.
        """
        embedding1 = self.get_sentence_embedding(text1)
        embedding2 = self.get_sentence_embedding(text2)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity

    def evaluate_grammar(self, text: str) -> float:
        """
        Evaluate text grammar quality (simplified version).
        For production, use language-specific grammar checkers.
        """
        lang = self.detect_language(text)

        # Basic checks based on language
        if len(text) < 5:
            return 0.8  # Too short to evaluate

        # English: Check for basic verb presence
        if lang == "english" and not re.search(
            r"\b(am|is|are|was|were|be|have|has|do|does|did|will|would|should|can|could|may|might|must)\b",
            text.lower(),
        ):
            return 0.7

        # Chinese: Check for basic punctuation
        if lang == "chinese" and not re.search(r"[，。！？]", text):
            return 0.7

        return 0.9  # Default to high score (improve with real grammar checks)

    def evaluate_accuracy(
        self, text: str, reference_text: str = None, fact_knowledge: dict = None
    ) -> float:
        """
        Comprehensive accuracy evaluation for text.

        Args:
            text: Input text to evaluate
            reference_text: Optional reference text for similarity comparison
            fact_knowledge: Optional knowledge base for fact checking

        Returns:
            Combined accuracy score (0-1 range)
        """
        # 1. Plausibility score (based on perplexity)
        plausibility_score = self.evaluate_perplexity(text)

        # 2. Grammar score
        grammar_score = self.evaluate_grammar(text)

        # 3. Similarity score (if reference text provided)
        similarity_score = 1.0
        if reference_text:
            similarity_score = self.evaluate_similarity(text, reference_text)

        # 4. Fact-checking score (if knowledge base provided)
        fact_score = 1.0
        if fact_knowledge:
            fact_errors = 0
            total_checks = 0

            for entity, fact in fact_knowledge.items():
                if entity in text:
                    total_checks += 1
                    # Construct verification question
                    question = f"{entity} is {fact}?"
                    similarity = self.evaluate_similarity(question, text)
                    if similarity < 0.7:  # Adjustable threshold
                        fact_errors += 1

            if total_checks > 0:
                fact_score = max(0.0, 1.0 - (fact_errors / total_checks))

        # Weighted average (adjust weights based on importance)
        weights = {
            "plausibility": 0.4,
            "grammar": 0.2,
            "similarity": 0.2 if reference_text else 0,
            "fact": 0.2 if fact_knowledge else 0,
        }

        # Normalize weights
        weight_sum = sum(weights.values())
        for key in weights:
            weights[key] /= weight_sum

        return (
            weights["plausibility"] * plausibility_score
            + weights["grammar"] * grammar_score
            + weights["similarity"] * similarity_score
            + weights["fact"] * fact_score
        )
