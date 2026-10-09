"""
Kneser-Ney language model based recaser.

This implements a truecaser using a Kneser-Ney smoothed n-gram language model.
The approach is:

1. Train a language model on properly cased text
2. For each lowercased word, find all casing variants seen in training
3. Use the LM to score each variant in context and pick the best one

The Kneser-Ney smoothing formula:
    P_KN(w|h) = max(c(h,w) - D, 0) / c(h) + λ(h) * P_KN(w|h')
    
Where:
    - D is the discount (typically 0.75)
    - λ(h) is the backoff weight
    - h' is the shorter history (for backoff)

References:
    - Kneser & Ney (1995) - Improved backing-off for M-gram language modeling
    - Chen & Goodman (1999) - An empirical study of smoothing techniques
    - classic statistical truecasing pipelines
"""

import os
import json
import math
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set


# ══════════════════════════════════════════════════════════════════════════════
#  Kneser-Ney N-gram Language Model
# ══════════════════════════════════════════════════════════════════════════════
class KneserNeyLM:
    """
    Kneser-Ney smoothed n-gram language model.
    
    This implementation uses modified Kneser-Ney smoothing with interpolation.
    """
    
    def __init__(self, order: int = 3, discount: float = 0.75):
        """
        Args:
            order: N-gram order (default 3 for trigram)
            discount: Discount parameter D (default 0.75)
        """
        self.order = order
        self.discount = discount
        
        # N-gram counts: ngram_counts[n][tuple] = count
        # n=1 is unigram, n=2 is bigram, etc.
        self.ngram_counts: Dict[int, Counter] = {i: Counter() for i in range(1, order + 1)}
        
        # Continuation counts for Kneser-Ney (how many unique contexts a word appears in)
        # continuation_counts[n][word] = number of unique (n-1)-gram contexts word appears in
        self.continuation_counts: Dict[int, Counter] = {i: Counter() for i in range(1, order + 1)}
        
        # Total counts and vocabulary
        self.total_unigrams = 0
        self.vocab: Set[str] = set()
        
        # Precomputed backoff weights
        self.backoff_weights: Dict[int, Dict[Tuple, float]] = {}
        self.context_totals: Dict[int, Dict[Tuple, int]] = {}
        self.context_num_types: Dict[int, Dict[Tuple, int]] = {}
        self.total_continuations = 0
        self._log_prob_cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        
        # Special tokens
        self.BOS = "<s>"
        self.EOS = "</s>"
        self.UNK = "<unk>"
    
    def train(self, sentences: List[List[str]]):
        """
        Train the language model on tokenized sentences.
        
        Args:
            sentences: List of tokenized sentences (each sentence is a list of words)
        """
        print(f"Training {self.order}-gram Kneser-Ney LM...")
        
        # Count n-grams
        for sentence in sentences:
            # Add BOS/EOS markers
            tokens = [self.BOS] * (self.order - 1) + sentence + [self.EOS]
            
            for i in range(len(tokens)):
                # Count n-grams of all orders
                for n in range(1, self.order + 1):
                    if i >= n - 1:
                        ngram = tuple(tokens[i - n + 1:i + 1])
                        self.ngram_counts[n][ngram] += 1
                        
                # Build vocabulary
                if tokens[i] not in (self.BOS, self.EOS):
                    self.vocab.add(tokens[i])

        self.total_unigrams = sum(self.ngram_counts[1].values())

        # Compute unique continuation counts after the full n-gram tables exist.
        self._compute_continuation_counts()
        self._compute_backoff_weights()
        
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Unigrams: {len(self.ngram_counts[1]):,}")
        print(f"  Bigrams: {len(self.ngram_counts[2]):,}")
        if self.order >= 3:
            print(f"  Trigrams: {len(self.ngram_counts[3]):,}")
    
    def _compute_continuation_counts(self):
        """Compute unique left-context counts for each word."""
        self.continuation_counts = {i: Counter() for i in range(1, self.order + 1)}
        for n in range(2, self.order + 1):
            for ngram in self.ngram_counts[n]:
                self.continuation_counts[n][ngram[-1]] += 1

        self.total_continuations = len(self.ngram_counts.get(2, {}))
        self._log_prob_cache.clear()

    def _compute_backoff_weights(self):
        """Precompute interpolation weights for each context."""
        self.context_totals = {}
        self.context_num_types = {}
        self.backoff_weights = {}

        for n in range(2, self.order + 1):
            self.backoff_weights[n] = {}

            # Group by context (all but last word)
            context_counts = defaultdict(int)
            context_types = defaultdict(int)

            for ngram, count in self.ngram_counts[n].items():
                context = ngram[:-1]
                context_counts[context] += count
                context_types[context] += 1

            self.context_totals[n] = dict(context_counts)
            self.context_num_types[n] = dict(context_types)

            # Compute lambda for each context
            for context, total in context_counts.items():
                num_types = context_types[context]
                # λ(h) = D * N_+(h•) / c(h)
                # where N_+(h•) is number of unique words following context h
                self.backoff_weights[n][context] = (self.discount * num_types) / total
        self._log_prob_cache.clear()

    def _get_unigram_prob(self, word: str) -> float:
        """
        Get the unigram backoff distribution.

        Standard Kneser-Ney uses the number of distinct left contexts of a word
        rather than its raw frequency. If bigram continuation counts are not
        available, fall back to a smoothed unigram distribution.
        """
        if self.total_continuations > 0:
            cont_count = self.continuation_counts[2].get(word, 0)
            if cont_count > 0:
                return cont_count / self.total_continuations

        count = self.ngram_counts[1].get((word,), 0)
        if count > 0 and self.total_unigrams > 0:
            return count / self.total_unigrams

        if len(self.vocab) == 0:
            return 1.0 / (len(self.vocab) + 1)

        return 1.0 / (len(self.vocab) + 1)

    def log_prob(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Compute log probability of word given context using Kneser-Ney smoothing.
        
        Args:
            word: The word to compute probability for
            context: Tuple of context words (history)
        
        Returns:
            Log probability (base e)
        """
        if self.order > 1:
            context = tuple(context[-(self.order - 1):])
        else:
            context = ()

        key = (word, context)
        cached = self._log_prob_cache.get(key)
        if cached is not None:
            return cached

        value = self._log_prob_recursive(word, context)
        self._log_prob_cache[key] = value
        return value

    def _log_prob_recursive(self, word: str, context: Tuple[str, ...]) -> float:
        """Internal recursive Kneser-Ney scoring with cached top-level calls."""
        # Determine effective n-gram order
        n = min(len(context) + 1, self.order)

        if n == 1:
            prob = self._get_unigram_prob(word)
            return math.log(prob) if prob > 0 else -100.0

        # Get the relevant context (last n-1 words)
        effective_context = context[-(n-1):] if len(context) >= n - 1 else context
        ngram = effective_context + (word,)

        # Get counts
        ngram_count = self.ngram_counts[n].get(ngram, 0)
        context_tuple = effective_context
        context_count = self.context_totals.get(n, {}).get(context_tuple, 0)

        if context_count == 0:
            # Backoff to lower order
            return self._log_prob_recursive(word, effective_context[1:] if effective_context else ())

        # Kneser-Ney formula
        # P(w|h) = max(c(h,w) - D, 0) / c(h) + λ(h) * P(w|h')
        discounted = max(ngram_count - self.discount, 0) / context_count

        # Get backoff weight
        lambda_h = self.backoff_weights.get(n, {}).get(context_tuple, 0.0)

        # Recursively get lower-order probability
        lower_context = effective_context[1:] if effective_context else ()
        lower_prob = math.exp(self._log_prob_recursive(word, lower_context))

        prob = discounted + lambda_h * lower_prob

        return math.log(prob) if prob > 0 else -100.0
    
    def score_sentence(self, tokens: List[str]) -> float:
        """
        Score a sentence (list of tokens) using the LM.
        
        Returns:
            Total log probability of the sentence
        """
        # Add BOS markers
        context = [self.BOS] * (self.order - 1)
        total_log_prob = 0.0
        
        for word in tokens + [self.EOS]:
            total_log_prob += self.log_prob(word, tuple(context))
            context = context[1:] + [word]
        
        return total_log_prob
    
    def save(self, path: str):
        """Save the model to a file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "order": self.order,
                "discount": self.discount,
                "ngram_counts": dict(self.ngram_counts),
                "continuation_counts": dict(self.continuation_counts),
                "total_unigrams": self.total_unigrams,
                "vocab": self.vocab,
                "backoff_weights": self.backoff_weights,
                "context_totals": self.context_totals,
                "context_num_types": self.context_num_types,
                "total_continuations": self.total_continuations,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "KneserNeyLM":
        """Load a model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        lm = cls(order=data["order"], discount=data["discount"])
        lm.ngram_counts = data["ngram_counts"]
        lm.continuation_counts = data["continuation_counts"]
        lm.total_unigrams = data["total_unigrams"]
        lm.vocab = data["vocab"]
        lm.backoff_weights = data["backoff_weights"]
        lm.context_totals = data.get("context_totals", {})
        lm.context_num_types = data.get("context_num_types", {})
        lm.total_continuations = data.get("total_continuations", len(lm.ngram_counts.get(2, {})))
        if not lm.context_totals:
            lm._compute_continuation_counts()
            lm._compute_backoff_weights()
        
        return lm


# ══════════════════════════════════════════════════════════════════════════════
#  Truecaser using Kneser-Ney LM
# ══════════════════════════════════════════════════════════════════════════════
class KNRecaser:
    """
    Truecaser/Recaser using Kneser-Ney language model.
    
    This follows the standard statistical truecasing setup:
    1. Build a mapping from lowercased words to all their casing variants
    2. Train a language model on properly cased text
    3. For each word, score all variants in context and pick the best
    """
    
    def __init__(
        self,
        lm: KneserNeyLM,
        casing_variants: Dict[str, Dict[str, int]],
        decode_beam_size: int = 6,
        max_candidates_per_token: int = 8,
        lexical_weight: float = 0.5,
    ):
        """
        Args:
            lm: Trained Kneser-Ney language model
            casing_variants: Map from lowercase word to {cased_variant: count}
        """
        self.lm = lm
        self.casing_variants = casing_variants
        self.decode_beam_size = max(1, decode_beam_size)
        self.max_candidates_per_token = max(1, max_candidates_per_token)
        self.lexical_weight = lexical_weight

        # Precompute the most frequent casing for each word (fallback)
        self.most_frequent_casing: Dict[str, str] = {}
        self.sorted_variants: Dict[str, List[Tuple[str, int]]] = {}
        self.variant_log_probs: Dict[str, Dict[str, float]] = {}
        for lower, variants in casing_variants.items():
            if variants:
                ranked = sorted(variants.items(), key=lambda x: (-x[1], x[0]))
                self.sorted_variants[lower] = ranked
                self.most_frequent_casing[lower] = ranked[0][0]
                total = sum(count for _, count in ranked)
                if total > 0:
                    self.variant_log_probs[lower] = {
                        variant: math.log(count / total)
                        for variant, count in ranked
                    }

    def _candidate_variants(
        self,
        word: str,
        context: List[str],
        max_candidates: int,
    ) -> List[str]:
        """Return the candidate casings to score for a token."""
        lower = word.lower()
        ranked = self.sorted_variants.get(lower)
        is_sentence_start = (not context) or (context[-1] in ".!?")

        if ranked is None:
            if not any(ch.isalpha() for ch in word):
                return [word]
            candidates = [lower]
            if is_sentence_start:
                capitalized = lower.capitalize()
                if capitalized not in candidates:
                    candidates.insert(0, capitalized)
            return candidates

        candidates = [variant for variant, _ in ranked[:max_candidates]]
        if is_sentence_start and any(ch.isalpha() for ch in word):
            capitalized = lower.capitalize()
            if capitalized not in candidates:
                candidates.append(capitalized)
        return candidates

    def get_best_casing(self, word: str, context: List[str]) -> str:
        """
        Get the best casing for a word given its context.
        
        Args:
            word: The word to case (can be any case)
            context: List of preceding words (already cased)
        
        Returns:
            Best casing of the word
        """
        lower = word.lower()

        candidates = self._candidate_variants(
            word,
            context,
            max_candidates=self.max_candidates_per_token,
        )
        if len(candidates) == 1:
            return candidates[0]

        context_tuple = tuple(context[-(self.lm.order - 1):]) if context else ()
        best_variant = None
        best_score = float("-inf")

        for variant in candidates:
            lexical_score = self.variant_log_probs.get(lower, {}).get(variant, 0.0)
            score = self.lm.log_prob(variant, context_tuple) + self.lexical_weight * lexical_score
            if score > best_score:
                best_score = score
                best_variant = variant

        if best_variant is not None:
            return best_variant
        if lower in self.most_frequent_casing:
            return self.most_frequent_casing[lower]
        return self._default_casing(word, context)
    
    def _default_casing(self, word: str, context: List[str]) -> str:
        """Default casing for unknown words."""
        # If at sentence start (empty context or after sentence-ending punctuation)
        if not context or (context and context[-1] in ".!?"):
            return word.capitalize()
        return word.lower()
    
    def recase(self, text: str) -> str:
        """
        Recase a sentence.
        
        Args:
            text: Input text (can be any case, typically lowercased)
        
        Returns:
            Properly cased text
        """
        words = text.strip().split()
        if not words:
            return text

        beam_size = self.decode_beam_size
        if beam_size <= 1:
            recased = []
            context = []
            for word in words:
                best = self.get_best_casing(word, context)
                recased.append(best)
                context.append(best)
            return " ".join(recased)

        beams: List[Tuple[float, Tuple[str, ...]]] = [(0.0, tuple())]

        for word in words:
            new_beams: List[Tuple[float, Tuple[str, ...]]] = []
            for score, history in beams:
                history_list = list(history)
                context_tuple = tuple(history[-(self.lm.order - 1):]) if self.lm.order > 1 else ()
                lower = word.lower()
                candidates = self._candidate_variants(
                    word,
                    history_list,
                    max_candidates=self.max_candidates_per_token,
                )

                for variant in candidates:
                    lexical_score = self.variant_log_probs.get(lower, {}).get(variant, 0.0)
                    variant_score = self.lm.log_prob(variant, context_tuple)
                    new_beams.append((
                        score + variant_score + self.lexical_weight * lexical_score,
                        history + (variant,),
                    ))

            new_beams.sort(key=lambda item: item[0], reverse=True)
            beams = new_beams[:beam_size]

        finished_beams: List[Tuple[float, Tuple[str, ...]]] = []
        for score, history in beams:
            context_tuple = tuple(history[-(self.lm.order - 1):]) if self.lm.order > 1 else ()
            finished_beams.append((
                score + self.lm.log_prob(self.lm.EOS, context_tuple),
                history,
            ))

        best_history = max(finished_beams, key=lambda item: item[0])[1]
        return " ".join(best_history)
    
    def recase_batch(self, texts: List[str]) -> List[str]:
        """Recase a batch of sentences."""
        return [self.recase(text) for text in texts]
    
    def save(self, model_dir: str):
        """Save the recaser to a directory."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LM
        self.lm.save(os.path.join(model_dir, "lm.pkl"))
        
        # Save casing variants
        with open(os.path.join(model_dir, "casing_variants.json"), "w", encoding="utf-8") as f:
            json.dump(self.casing_variants, f, ensure_ascii=False, indent=2)
        
        # Save config
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({
                "type": "kneser_ney",
                "lm_order": self.lm.order,
                "vocab_size": len(self.lm.vocab),
                "num_casing_variants": len(self.casing_variants),
                "decode_beam_size": self.decode_beam_size,
                "max_candidates_per_token": self.max_candidates_per_token,
                "lexical_weight": self.lexical_weight,
            }, f, indent=2)
    
    @classmethod
    def load(cls, model_dir: str) -> "KNRecaser":
        """Load a recaser from a directory."""
        # Load LM
        lm = KneserNeyLM.load(os.path.join(model_dir, "lm.pkl"))

        # Load casing variants
        with open(os.path.join(model_dir, "casing_variants.json"), "r", encoding="utf-8") as f:
            casing_variants = json.load(f)

        config_path = os.path.join(model_dir, "config.json")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        return cls(
            lm,
            casing_variants,
            decode_beam_size=config.get("decode_beam_size", 6),
            max_candidates_per_token=config.get("max_candidates_per_token", 8),
            lexical_weight=config.get("lexical_weight", 0.5),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Training Functions
# ══════════════════════════════════════════════════════════════════════════════
def build_casing_variants(sentences: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build a mapping from lowercase words to their casing variants.
    
    Args:
        sentences: List of properly cased sentences
    
    Returns:
        Dict mapping lowercase word -> {cased_variant: count}
    """
    variants: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for sentence in sentences:
        for word in sentence.split():
            lower = word.lower()
            variants[lower][word] += 1
    
    # Convert to regular dict
    return {k: dict(v) for k, v in variants.items()}


def train_kn_recaser(
    sentences: List[str],
    order: int = 3,
    discount: float = 0.75,
    decode_beam_size: int = 6,
    max_candidates_per_token: int = 8,
    lexical_weight: float = 0.5,
) -> KNRecaser:
    """
    Train a Kneser-Ney based recaser.
    
    Args:
        sentences: List of properly cased sentences
        order: N-gram order for the LM
        discount: Kneser-Ney discount parameter
    
    Returns:
        Trained KNRecaser
    """
    print("Building casing variants...")
    casing_variants = build_casing_variants(sentences)
    print(f"  Found {len(casing_variants):,} unique lowercased words")
    
    # Tokenize sentences for LM training
    tokenized = [sent.split() for sent in sentences]
    
    # Train LM
    lm = KneserNeyLM(order=order, discount=discount)
    lm.train(tokenized)

    return KNRecaser(
        lm,
        casing_variants,
        decode_beam_size=decode_beam_size,
        max_candidates_per_token=max_candidates_per_token,
        lexical_weight=lexical_weight,
    )
