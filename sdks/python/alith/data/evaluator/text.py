import torch
import re
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM, XLMRobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker


class TextEvaluator:
    def __init__(self, device=None):
        """Initialize the evaluator with GPU support and enhanced checks"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TextEvaluator using device: {self.device}")

        # Initialize models and move to GPU
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large")
        self.mlm_model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-large").to(
            self.device
        )
        self.encoder_model = XLMRobertaModel.from_pretrained("xlm-roberta-large").to(
            self.device
        )

        self.mlm_model.eval()
        self.encoder_model.eval()

        # Language detection patterns
        self.language_patterns = {
            "chinese": r"[\u4e00-\u9fff]",
            "japanese": r"[\u3040-\u309f\u30a0-\u30ff]",
            "korean": r"[\uac00-\ud7af\u1100-\u11ff]",
            "english": r"[a-zA-Z]",
        }

        # English spell checker
        self.spell_checker = SpellChecker()

    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, text):
                return lang
        return "other"

    def evaluate_perplexity(self, text: str) -> float:
        """Improved perplexity calculation using full sentence likelihood"""
        if len(text.strip()) == 0:
            return 0.0

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.mlm_model(**inputs, labels=inputs.input_ids)

        # Calculate perplexity from MLM loss
        loss = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(loss)).item()

        # Normalize to 0-1 (lower is better)
        max_perplexity = 50  # Adjust sensitivity
        return max(0.0, 1.0 - min(perplexity, max_perplexity) / max_perplexity)

    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        """Generate sentence embedding using [CLS] token"""
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder_model(**inputs)

        # Return [CLS] embedding as numpy array
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def evaluate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embedding1 = self.get_sentence_embedding(text1)
        embedding2 = self.get_sentence_embedding(text2)
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return similarity

    def evaluate_grammar(self, text: str) -> float:
        """Enhanced grammar/spelling check with language-specific rules"""
        lang = self.detect_language(text)

        # Handle very short text
        if len(text.strip()) < 5:
            return 0.8  # Too short to evaluate

        # English-specific checks
        if lang == "english":
            # Check for basic verbs
            has_verb = bool(
                re.search(
                    r"\b(am|is|are|was|were|be|have|has|do|does|did|will|would|should|can|could|may|might|must)\b",
                    text.lower(),
                )
            )
            verb_score = 0.9 if has_verb else 0.7

            # Check spelling
            words = re.findall(r"\b[a-zA-Z]+\b", text)
            if not words:
                return 0.5  # No words found

            misspelled = self.spell_checker.unknown(words)
            spell_error_rate = len(misspelled) / len(words)
            spell_score = max(0.0, 1.0 - spell_error_rate)

            # Penalize excessive non-alphabet characters
            non_alpha_ratio = len(re.findall(r"[^a-zA-Z\s,.?!]", text)) / max(
                len(text), 1
            )
            non_alpha_penalty = min(1.0, non_alpha_ratio * 2)  # Double penalty

            # Combine scores
            return ((verb_score * 0.4) + (spell_score * 0.4)) * (1 - non_alpha_penalty)

        # Default check for other languages (basic punctuation check)
        if lang in ["chinese", "japanese", "korean"]:
            has_punctuation = bool(re.search(r"[，。！？、；：]", text))
            return 0.9 if has_punctuation else 0.7

        return 0.9  # Default high score for unknown languages

    def evaluate_accuracy(
        self, text: str, reference_text: str = None, fact_knowledge: dict = None
    ) -> float:
        """Comprehensive text accuracy evaluation"""
        # Core metrics
        plausibility_score = self.evaluate_perplexity(text)
        grammar_score = self.evaluate_grammar(text)

        # Similarity to reference text (if provided)
        similarity_score = 1.0
        if reference_text:
            similarity_score = self.evaluate_similarity(text, reference_text)

        # Fact-checking (if knowledge base provided)
        fact_score = 1.0
        if fact_knowledge:
            fact_errors = 0
            total_checks = 0

            for entity, fact in fact_knowledge.items():
                if entity in text:
                    total_checks += 1
                    question = f"{entity} is {fact}?"
                    similarity = self.evaluate_similarity(question, text)
                    if similarity < 0.7:  # Threshold for fact mismatch
                        fact_errors += 1

            if total_checks > 0:
                fact_score = max(0.0, 1.0 - (fact_errors / total_checks))

        # Weighted average (adjust weights based on importance)
        weights = {
            "plausibility": 0.4,
            "grammar": 0.3,  # Increased weight for grammar
            "similarity": 0.2 if reference_text else 0,
            "fact": 0.1 if fact_knowledge else 0,
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
