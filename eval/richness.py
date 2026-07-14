"""Richness metrics (spaCy); corpus aggregation happens in summarize.py."""

import spacy

_NLP = None


def _nlp(config: dict):
    global _NLP
    if _NLP is None:
        name = config.get("spacy_model", "en_core_web_sm")
        _NLP = spacy.load(name, disable=["ner", "lemmatizer", "parser"])
    return _NLP


def richness(caption: str, config: dict) -> dict:
    doc = _nlp(config)(caption)
    alpha = [t for t in doc if t.is_alpha]
    words = [t.text.lower() for t in alpha]
    n_tokens = len(words)
    types = sorted(set(words))
    n_noun = sum(1 for t in alpha if t.pos_ in ("NOUN", "PROPN"))
    n_verb = sum(1 for t in alpha if t.pos_ == "VERB")
    n_adj = sum(1 for t in alpha if t.pos_ == "ADJ")
    word_count = len(caption.split())
    return {
        "word_count": word_count,
        "below_floor": word_count < config.get("min_words", 30),
        "n_tokens": n_tokens,
        "n_types": len(types),
        "ttr": (len(types) / n_tokens) if n_tokens else 0.0,
        "pos": {"noun": n_noun, "verb": n_verb, "adj": n_adj, "other": n_tokens - n_noun - n_verb - n_adj},
        "types": types,
    }
