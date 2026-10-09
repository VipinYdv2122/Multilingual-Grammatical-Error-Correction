"""
Recaser utilities for GEC pipeline.

This module provides functions to load and apply the Kneser-Ney LM-based
recaser used by the GEC pipeline.

Usage:
    from scripts.recaser_utils import load_recaser, apply_recaser

    recaser = load_recaser(lang="english")
    recased_text = recaser.recase("i went to new york yesterday .")
    # Output: "I went to New York yesterday ."
"""

import os
from typing import List, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kn_recaser import KNRecaser


def get_recaser_path(lang: str, base_dir: str = "tokenizers/recaser") -> str:
    """Get the path to the recaser model directory."""
    return os.path.join(base_dir, lang)


def recaser_exists(lang: str, base_dir: str = "tokenizers/recaser") -> bool:
    """Check if a trained recaser exists for the given language."""
    model_dir = get_recaser_path(lang, base_dir)
    required_files = ["lm.pkl", "casing_variants.json", "config.json"]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)


def load_recaser(
    lang: str,
    base_dir: str = "tokenizers/recaser",
) -> Optional[KNRecaser]:
    """
    Load a trained Kneser-Ney recaser for the given language.
    
    Args:
        lang: Language name (e.g., "english", "hindi")
        base_dir: Base directory for recaser models
    
    Returns:
        KNRecaser object, or None if model doesn't exist
    """
    model_dir = get_recaser_path(lang, base_dir)
    
    if not recaser_exists(lang, base_dir):
        return None
    
    return KNRecaser.load(model_dir)


def apply_recaser(text: str, recaser: Optional[KNRecaser]) -> str:
    """
    Apply recaser to a single sentence.
    
    Args:
        text: Input text (lowercased)
        recaser: KNRecaser object (or None for fallback)
    
    Returns:
        Recased text
    """
    if recaser is None:
        return capitalize_first_letter(text)
    return recaser.recase(text)


def apply_recaser_batch(texts: List[str], recaser: Optional[KNRecaser]) -> List[str]:
    """
    Apply recaser to a batch of sentences.
    
    Args:
        texts: List of input texts (lowercased)
        recaser: KNRecaser object (or None for fallback)
    
    Returns:
        List of recased texts
    """
    if recaser is None:
        return [capitalize_first_letter(t) for t in texts]
    return recaser.recase_batch(texts)


def capitalize_first_letter(text: str) -> str:
    """Simple fallback: capitalize first letter of sentence."""
    text = text.strip()
    if text:
        return text[0].upper() + text[1:]
    return text


def detokenize_text(text: str) -> str:
    """
    Simple detokenization: fix common spacing issues around punctuation.
    """
    text = text.strip()
    # Fix spacing around punctuation
    text = text.replace(" .", ".").replace(" ,", ",").replace(" !", "!")
    text = text.replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
    text = text.replace(" '", "'").replace(" n't", "n't")
    text = text.replace("( ", "(").replace(" )", ")")
    text = text.replace("[ ", "[").replace(" ]", "]")
    # Fix contractions
    text = text.replace(" 's", "'s").replace(" 're", "'re")
    text = text.replace(" 've", "'ve").replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d").replace(" 'm", "'m")
    return text
