"""F9 content structure + E4 / F20 rhetoric analysis.

Rhetoric comes from the QLoRA-tuned LLM served behind an OpenAI-compatible endpoint
(VAKTA_LLM_URL, e.g. training/serve_rhetoric.py or Ollama). Output is validated against
RhetoricReport with pydantic, retried with the validation error, and falls back to a
transparent lexicon/regex analyser so a report is always produced.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .settings import get_settings
from .text import locate, sentences_from_words, split_sentences, tokens
from .utils import rnd, trapezoid

DEVICE_TYPES = ["anaphora", "epistrophe", "tricolon", "rhetorical_question", "antithesis", "alliteration",
                "metaphor", "story", "statistic", "quote", "callback", "repetition", "other"]
TRANSITIONS: dict[str, list[str]] = {
    "sequence": ["first", "firstly", "second", "secondly", "third", "next", "then", "finally", "lastly", "to begin"],
    "contrast": ["however", "but", "on the other hand", "yet", "although", "instead", "in contrast", "still"],
    "cause": ["because", "therefore", "so", "as a result", "consequently", "thus", "hence", "that's why"],
    "example": ["for example", "for instance", "such as", "imagine", "take"],
    "addition": ["also", "moreover", "furthermore", "in addition", "and another", "plus"],
    "conclusion": ["in conclusion", "to sum up", "overall", "ultimately", "in short", "to conclude", "so remember"],
}
ETHOS = {"experience", "years", "expert", "trained", "studied", "research", "worked", "team", "trust", "credible",
         "professor", "doctor", "engineer", "founded", "led", "built", "certified", "know", "seen", "personally"}
PATHOS = {"feel", "felt", "love", "fear", "hope", "imagine", "dream", "heart", "pain", "joy", "proud", "afraid",
          "angry", "sad", "happy", "children", "family", "story", "lonely", "tears", "beautiful", "scared", "care"}
LOGOS = {"because", "therefore", "data", "evidence", "percent", "research", "study", "result", "results",
         "statistic", "statistics", "proves", "cause", "thus", "hence", "average", "million", "billion", "number",
         "measured", "analysis", "fact", "facts", "reason", "logic", "compare"}

RUN_ON_WORDS = 80  # longer "sentences" are ASR stretches without punctuation

RUBRIC_PROMPT = """You are a speech-rhetoric analyst. Analyse the speech transcript and answer ONLY with JSON:
{"devices": [{"type": one of %s, "text": exact quote, "explanation": short}],
 "appeals": {"ethos": 0-1, "pathos": 0-1, "logos": 0-1},
 "appeal_evidence": {"ethos": [quotes], "pathos": [quotes], "logos": [quotes]},
 "transitions": [{"text": exact quote, "category": one of %s}],
 "transition_quality": 0-1,
 "suggestions": [{"original": exact quote, "rewrite": improved sentence, "reason": short}],
 "summary": one or two sentences}
Appeals must sum to 1. Quote the transcript exactly. Give 1-3 suggestions.""" % (DEVICE_TYPES, list(TRANSITIONS))


class Device(BaseModel):
    type: str
    text: str
    explanation: str = ""
    start: float | None = None
    end: float | None = None

    @field_validator("type")
    @classmethod
    def _norm_type(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_").replace("-", "_")
        return v if v in DEVICE_TYPES else "other"


class Appeals(BaseModel):
    ethos: float = Field(ge=0, le=1)
    pathos: float = Field(ge=0, le=1)
    logos: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _normalise(self) -> "Appeals":
        total = self.ethos + self.pathos + self.logos
        if total > 0:
            self.ethos, self.pathos, self.logos = (round(x / total, 3) for x in (self.ethos, self.pathos, self.logos))
        return self


class TransitionItem(BaseModel):
    text: str
    category: str
    start: float | None = None


class Suggestion(BaseModel):
    original: str
    rewrite: str
    reason: str = ""


class RhetoricReport(BaseModel):
    devices: list[Device] = []
    appeals: Appeals
    appeal_evidence: dict[str, list[str]] = {}
    transitions: list[TransitionItem] = []
    transition_quality: float = Field(ge=0, le=1)
    suggestions: list[Suggestion] = []
    summary: str = ""


# --------------------------------------------------------------------------- structure (F9)
HOOKS = {
    "question": lambda s: s.rstrip().endswith("?"),
    "statistic": lambda s: bool(re.search(r"\d|percent|million|billion", s.lower())),
    "story": lambda s: bool(re.search(r"\b(when i|years ago|one day|i remember|last (year|week|month)|imagine)\b", s.lower())),
    "quote": lambda s: '"' in s or "once said" in s.lower(),
}
WEAK_OPEN = re.compile(r"^(hi|hello|good (morning|afternoon|evening)|my name is|today i('m| am) going to talk|so today)\b", re.I)
CLOSE_CTA = re.compile(r"\b(let's|let us|i urge|i challenge|start|try|join|go and|remember|take action|next time)\b", re.I)
CLOSE_SUMMARY = re.compile(r"\b(in conclusion|to sum up|to conclude|in short|so remember|that's why|overall)\b", re.I)


def classify_opener(sentences: list[str]) -> dict[str, Any]:
    first = " ".join(sentences[:2])
    for kind, test in HOOKS.items():
        if test(sentences[0] if kind == "question" else first):
            return {"type": kind, "text": sentences[0], "score": 1.0}
    weak = bool(WEAK_OPEN.search(sentences[0]))
    return {"type": "weak-greeting" if weak else "plain", "text": sentences[0], "score": 0.3 if weak else 0.6}


def classify_closer(sentences: list[str]) -> dict[str, Any]:
    tail = [s for s in sentences[-3:] if not re.fullmatch(r"(thank you|thanks)[.! ]*", s.strip().lower())] or sentences[-1:]
    text = " ".join(tail)
    if CLOSE_CTA.search(text):
        return {"type": "call-to-action", "text": tail[-1], "score": 1.0}
    if CLOSE_SUMMARY.search(text):
        return {"type": "summary", "text": tail[-1], "score": 0.8}
    return {"type": "abrupt", "text": tail[-1], "score": 0.35}


def repeated_phrases(text: str, n: int = 3, min_count: int = 3) -> list[dict[str, Any]]:
    toks = tokens(text)
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return [{"phrase": " ".join(g), "count": c} for g, c in grams.most_common(5) if c >= min_count]


def structure_checks(sentences: list[str]) -> dict[str, Any]:
    lengths = [len(tokens(s)) for s in sentences]
    return {
        "sentence_count": len(sentences),
        "mean_sentence_words": rnd(sum(lengths) / len(lengths), 1) if lengths else None,
        "long_sentences": [s for s, n in zip(sentences, lengths) if n > 35][:5],
        "opener": classify_opener(sentences),
        "closer": classify_closer(sentences),
        "repetition": repeated_phrases(" ".join(sentences)),
    }


# --------------------------------------------------------------------------- heuristic rhetoric
def _first(s: str, n: int) -> tuple[str, ...]:
    return tuple(tokens(s)[:n])


def _last(s: str, n: int) -> tuple[str, ...]:
    return tuple(tokens(s)[-n:])


def find_devices(sentences: list[str]) -> list[Device]:
    devices: list[Device] = []
    for i in range(1, len(sentences)):
        a, b = sentences[i - 1], sentences[i]
        if len(tokens(a)) > 2 and _first(a, 2) == _first(b, 2):
            devices.append(Device(type="anaphora", text=f"{a} {b}", explanation=f"Consecutive sentences open with '{' '.join(_first(a, 2))}'."))
        if len(tokens(a)) > 2 and _last(a, 2) == _last(b, 2):
            devices.append(Device(type="epistrophe", text=f"{a} {b}", explanation=f"Consecutive sentences end with '{' '.join(_last(a, 2))}'."))
    for s in sentences:
        if s.rstrip().endswith("?"):
            devices.append(Device(type="rhetorical_question", text=s, explanation="Question posed to the audience."))
        if re.search(r"\b[\w']+(?: [\w']+){0,3}, [\w']+(?: [\w']+){0,3},? and [\w']+", s):
            devices.append(Device(type="tricolon", text=s, explanation="A list of three for rhythm and emphasis."))
        if re.search(r"\bnot\b[^.?!]{1,50}\bbut\b", s, re.I):
            devices.append(Device(type="antithesis", text=s, explanation="Contrasts two ideas ('not ... but ...')."))
        if re.search(r"\d+(\.\d+)?\s*(%|percent)|\b\d[\d,]*\s+(million|billion|people|years)", s, re.I):
            devices.append(Device(type="statistic", text=s, explanation="Uses a number as evidence."))
        if HOOKS["story"](s):
            devices.append(Device(type="story", text=s, explanation="Personal or narrative framing."))
    return devices


def appeal_scores(sentences: list[str]) -> tuple[Appeals, dict[str, list[str]]]:
    counts = {"ethos": 0.0, "pathos": 0.0, "logos": 0.0}
    evidence: dict[str, list[tuple[int, str]]] = {k: [] for k in counts}
    for s in sentences:
        toks = tokens(s)
        hits = {
            "ethos": sum(t in ETHOS for t in toks) + (1 if re.search(r"\b(as a|i've|i have been|in my)\b", s, re.I) else 0),
            "pathos": sum(t in PATHOS for t in toks),
            "logos": sum(t in LOGOS for t in toks) + len(re.findall(r"\d", s)) * 0.5,
        }
        for k, h in hits.items():
            counts[k] += h
            if h:
                evidence[k].append((h, s))
    total = sum(counts.values()) + 3
    appeals = Appeals(**{k: (v + 1) / total for k, v in counts.items()})
    return appeals, {k: [s for _, s in sorted(v, reverse=True)[:2]] for k, v in evidence.items()}


def find_transitions(sentences: list[str]) -> tuple[list[TransitionItem], float]:
    found: list[TransitionItem] = []
    for s in sentences[1:]:
        head = " ".join(tokens(s)[:4])
        for cat, phrases in TRANSITIONS.items():
            hit = next((p for p in phrases if re.match(rf"(\w+ ){{0,2}}{re.escape(p)}\b", head)), None)
            if hit:
                found.append(TransitionItem(text=hit, category=cat))
                break
    if len(sentences) < 2:
        return found, 0.5
    share = len(found) / (len(sentences) - 1)
    variety = len({t.category for t in found}) / len(TRANSITIONS)
    quality = 0.7 * (trapezoid(share, 0.0, 0.2, 0.55, 0.9) or 0.0) + 0.3 * variety
    return found, round(quality, 3)


def heuristic_suggestions(structure: dict[str, Any], sentences: list[str]) -> list[Suggestion]:
    out: list[Suggestion] = []
    opener = structure["opener"]
    if opener["score"] < 0.7:
        out.append(Suggestion(original=opener["text"], rewrite="Open with a question, a striking number or a one-line story before introducing yourself.",
                              reason="The first sentence is a greeting or plain statement, not a hook."))
    closer = structure["closer"]
    if closer["score"] < 0.7:
        out.append(Suggestion(original=closer["text"], rewrite="End with one sentence telling the audience exactly what to do or remember.",
                              reason="The speech ends without a summary or call to action."))
    for s in structure["long_sentences"][:1]:
        words = s.split()
        parts = re.split(r",\s*(?:and|but|so)\s+", s, maxsplit=1)
        if len(words) > RUN_ON_WORDS:
            # Usually an unpunctuated stretch of speech, not one written sentence: quote the start only.
            out.append(Suggestion(original=" ".join(words[:25]) + " …",
                                  rewrite="Break this stretch into short sentences of 15-20 words and pause at each full stop.",
                                  reason=f"About {len(words)} words without a sentence break; listeners lose the thread."))
        elif len(parts) > 1:
            rewrite = ". ".join(p.strip().rstrip(".") for p in parts) + "."
            out.append(Suggestion(original=s, rewrite=rewrite, reason="Over 35 words; split it so listeners can follow."))
    return out[:3]


def heuristic_rhetoric(sentences: list[str], structure: dict[str, Any]) -> RhetoricReport:
    appeals, evidence = appeal_scores(sentences)
    transitions, quality = find_transitions(sentences)
    devices = find_devices(sentences)
    dominant = max(("ethos", "pathos", "logos"), key=lambda k: getattr(appeals, k))
    summary = (f"{len(devices)} rhetorical devices found; the speech leans on {dominant}; "
               f"{len(transitions)} signposted transitions.")
    return RhetoricReport(devices=devices, appeals=appeals, appeal_evidence=evidence, transitions=transitions,
                          transition_quality=quality, suggestions=heuristic_suggestions(structure, sentences),
                          summary=summary)


# --------------------------------------------------------------------------- LLM path
def extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def llm_rhetoric(transcript: str, url: str, model: str, retries: int = 2) -> RhetoricReport:
    import httpx

    messages = [{"role": "system", "content": RUBRIC_PROMPT}, {"role": "user", "content": transcript}]
    last_error = ""
    for _ in range(retries + 1):
        resp = httpx.post(f"{url.rstrip('/')}/v1/chat/completions", timeout=120.0,
                          json={"model": model, "messages": messages, "temperature": 0.2,
                                "response_format": {"type": "json_object"}})
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return RhetoricReport.model_validate(extract_json(content))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:500]
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Invalid JSON for the schema: {last_error}. Reply with corrected JSON only."}]
    raise ValueError(f"LLM output failed validation: {last_error}")


def attach_times(report: RhetoricReport, words: list[dict[str, Any]]) -> RhetoricReport:
    for d in report.devices:
        d.start, d.end = locate(d.text, words)
    for t in report.transitions:
        t.start = locate(t.text, words)[0]
    return report


class ContentAnalyzer(Analyzer):
    name = "content"
    requires = ("asr",)

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        asr = self.need(ctx, "asr")
        words = asr.get("words", [])
        if len(words) < 5:
            raise AnalyzerSkipped("transcript too short for content analysis")
        timed = sentences_from_words(words)
        sentences = [s["text"] for s in timed] if len(timed) > 1 else split_sentences(asr["text"])
        structure = structure_checks(sentences)
        settings = get_settings()
        backend, note = "heuristic", None
        report = None
        if settings.llm_url:
            try:
                report = llm_rhetoric(asr["text"], settings.llm_url, settings.llm_model)
                backend = f"llm:{settings.llm_model}"
            except Exception as exc:  # noqa: BLE001 - endpoint down / invalid JSON
                note = f"LLM unavailable, used heuristic analyser ({type(exc).__name__})"
        if report is None:
            report = heuristic_rhetoric(sentences, structure)
        attach_times(report, words)
        return {"structure": structure, "rhetoric": report.model_dump(), "sentences": timed,
                "note": note, "models": {"rhetoric": backend}}
