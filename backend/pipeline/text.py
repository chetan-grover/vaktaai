"""Transcript helpers shared by content, timeline and fusion: sentences, quote lookup, sentiment."""
from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"[a-z0-9']+")
SENTENCE_END = (".", "?", "!")

POSITIVE = {"good", "great", "happy", "love", "excited", "wonderful", "amazing", "proud", "success", "win",
            "best", "hope", "glad", "enjoy", "fantastic", "beautiful", "grateful", "thrilled", "easy", "better"}
NEGATIVE = {"bad", "sad", "terrible", "awful", "fail", "failed", "failure", "lost", "loss", "died", "death",
            "pain", "problem", "worse", "worst", "angry", "fear", "afraid", "sorry", "tragic", "hard", "crisis",
            "unfortunately", "disaster", "hurt", "difficult", "poor", "sick", "fired", "worried"}


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def sentences_from_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group timestamped words into sentences at . ? ! boundaries."""
    sentences, current = [], []
    for w in words:
        current.append(w)
        if w["w"].rstrip('"\'').endswith(SENTENCE_END):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return [{"text": " ".join(w["w"] for w in s).strip(), "start": s[0]["start"], "end": s[-1]["end"]}
            for s in sentences]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def locate(quote: str, words: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """Start/end time of the first occurrence of `quote` in the timestamped words (matches its first tokens)."""
    q = tokens(quote)
    if not q or not words:
        return None, None
    wt = [tokens(w["w"])[0] if tokens(w["w"]) else "" for w in words]
    probe = q[: min(4, len(q))]
    for i in range(len(wt) - len(probe) + 1):
        if wt[i:i + len(probe)] == probe:
            j = min(i + len(q) - 1, len(words) - 1)
            return words[i]["start"], words[j]["end"]
    return None, None


def lexicon_sentiment(text: str) -> float | None:
    """(pos - neg) / hits in [-1, 1], None when no sentiment words."""
    toks = tokens(text)
    pos = sum(t in POSITIVE for t in toks)
    neg = sum(t in NEGATIVE for t in toks)
    return (pos - neg) / (pos + neg) if pos + neg else None
