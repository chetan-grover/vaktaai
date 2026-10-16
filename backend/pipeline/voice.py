"""F4 voice features (pitch, pace, pauses, energy) and F5/S2 filler detection incl. Hinglish."""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .ingest import load_audio
from .utils import nanmean, nanstd, per_second, rnd, window_starts

PAUSE_MIN_S = 0.4
LONG_PAUSE_S = 2.0

# Always fillers, whatever the context.
HARD_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "hmm", "mm", "mhm"}
# Fillers only when set off by a pause or comma ("I like pizza" is not a filler).
SOFT_FILLERS = {"like", "so", "basically", "actually", "literally", "right", "okay", "ok", "well", "anyway"}
MULTI_FILLERS = {("you", "know"), ("i", "mean"), ("kind", "of"), ("sort", "of")}
# S2: Hinglish discourse fillers.
HINGLISH_FILLERS = {"matlab", "toh", "na", "yaar", "haan", "accha", "acha", "bas", "woh", "vo", "arey", "arre"}
SOFT_GAP_S = 0.25

_PUNCT = re.compile(r"[^\w']+", re.UNICODE)


def clean(token: str) -> str:
    return _PUNCT.sub("", token.lower())


def _gap_before(words: list[dict], i: int) -> float:
    return words[i]["start"] - words[i - 1]["end"] if i > 0 else math.inf


def _gap_after(words: list[dict], i: int) -> float:
    return words[i + 1]["start"] - words[i]["end"] if i + 1 < len(words) else math.inf


def detect_fillers(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fillers: list[dict[str, Any]] = []
    i = 0
    while i < len(words):
        token = clean(words[i]["w"])
        nxt = clean(words[i + 1]["w"]) if i + 1 < len(words) else ""
        if (token, nxt) in MULTI_FILLERS and words[i + 1]["w"].rstrip().endswith(","):
            fillers.append(_filler(words[i], words[i + 1], f"{token} {nxt}", "en"))
            i += 2
            continue
        set_off = (words[i]["w"].endswith(",") or _gap_before(words, i) > SOFT_GAP_S
                   or _gap_after(words, i) > SOFT_GAP_S)
        if token in HARD_FILLERS:
            fillers.append(_filler(words[i], words[i], token, "en"))
        elif token in HINGLISH_FILLERS and set_off:
            fillers.append(_filler(words[i], words[i], token, "hinglish"))
        elif token in SOFT_FILLERS and set_off:
            fillers.append(_filler(words[i], words[i], token, "en"))
        i += 1
    return fillers


def _filler(first: dict, last: dict, text: str, lang: str) -> dict[str, Any]:
    return {"text": text, "start": first["start"], "end": last["end"], "lang": lang}


def detect_pauses(words: list[dict[str, Any]], min_gap: float = PAUSE_MIN_S) -> list[dict[str, Any]]:
    pauses = []
    for prev, cur in zip(words, words[1:]):
        gap = cur["start"] - prev["end"]
        if gap >= min_gap:
            pauses.append({"start": rnd(prev["end"]), "end": rnd(cur["start"]), "duration": rnd(gap)})
    return pauses


def wpm_windows(words: list[dict[str, Any]], duration: float, size: float = 5.0) -> list[dict[str, Any]]:
    starts = np.array([w["start"] for w in words], dtype=float)
    windows = []
    for s in window_starts(duration, size, size):
        count = int(((starts >= s) & (starts < s + size)).sum())
        windows.append({"start": rnd(s), "end": rnd(min(s + size, duration)), "words": count,
                        "wpm": rnd(count * 60.0 / size, 1)})
    return windows


def pitch_track(samples: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """F0 in Hz every 10 ms (NaN where unvoiced), via Praat."""
    import parselmouth

    snd = parselmouth.Sound(samples.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    return pitch.xs(), f0


def energy_track(samples: np.ndarray, sr: int, frame_s: float = 0.025, hop_s: float = 0.01
                 ) -> tuple[np.ndarray, np.ndarray]:
    frame, hop = int(sr * frame_s), int(sr * hop_s)
    if len(samples) < frame:
        return np.array([]), np.array([])
    n = 1 + (len(samples) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(samples[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    t = (np.arange(n) * hop + frame / 2) / sr
    return t, db


def semitone_stats(f0: np.ndarray) -> dict[str, float | None]:
    voiced = f0[~np.isnan(f0)]
    if voiced.size < 20:
        return {"mean_hz": None, "std_hz": None, "std_st": None, "range_st": None}
    st = 12 * np.log2(voiced / np.median(voiced))
    lo, hi = np.percentile(st, [5, 95])
    return {"mean_hz": rnd(voiced.mean(), 1), "std_hz": rnd(voiced.std(), 1),
            "std_st": rnd(st.std(), 2), "range_st": rnd(hi - lo, 2)}


def speech_mask(db: np.ndarray) -> np.ndarray:
    """Frames louder than the noise floor + 12 dB count as speech."""
    if db.size == 0:
        return db.astype(bool)
    floor = np.percentile(db, 10)
    return db > floor + 12


class VoiceAnalyzer(Analyzer):
    name = "voice"
    requires = ("ingest", "asr")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        if not ingest.get("has_audio"):
            raise AnalyzerSkipped("recording has no audio track")
        duration = float(ingest["duration_s"])
        samples, sr = load_audio(ctx.audio_path)
        asr = ctx.read("asr") or {}
        words = asr.get("words", [])

        e_t, e_db = energy_track(samples, sr)
        speaking = speech_mask(e_db)
        try:
            p_t, f0 = pitch_track(samples, sr)
            pitch_backend = "parselmouth"
        except ImportError:
            p_t, f0, pitch_backend = np.array([]), np.array([]), "unavailable"

        pauses = detect_pauses(words)
        fillers = detect_fillers(words)
        windows = wpm_windows(words, duration, 5.0)
        speech_span = (words[-1]["end"] - words[0]["start"]) if len(words) > 1 else 0.0
        paused = sum(p["duration"] for p in pauses)
        minutes = max(duration / 60.0, 1e-6)
        speech_db = e_db[speaking] if speaking.any() else e_db
        has_words = bool(words)  # without a transcript, word-based metrics are unknown, not zero

        summary = {
            "word_count": len(words),
            "wpm": rnd(len(words) * 60 / speech_span, 1) if speech_span > 0 else None,
            "articulation_wpm": rnd(len(words) * 60 / max(speech_span - paused, 1e-6), 1) if speech_span > 0 else None,
            "wpm_std": rnd(nanstd([w["wpm"] for w in windows if w["words"] > 0]), 1),
            "pause_count": (len(pauses)) if has_words else None,
            "pauses_per_min": (rnd(len(pauses) / minutes, 2)) if has_words else None,
            "mean_pause_s": rnd(nanmean([p["duration"] for p in pauses]), 2),
            "long_pause_count": (sum(p["duration"] >= LONG_PAUSE_S for p in pauses)) if has_words else None,
            "pause_ratio": rnd(paused / speech_span, 3) if speech_span > 0 else None,
            "filler_count": (len(fillers)) if has_words else None,
            "filler_rate_per_min": (rnd(len(fillers) / minutes, 2)) if has_words else None,
            "hinglish_filler_count": (sum(f["lang"] == "hinglish" for f in fillers)) if has_words else None,
            "energy_mean_db": rnd(float(np.mean(speech_db)), 2) if speech_db.size else None,
            "energy_std_db": rnd(float(np.std(speech_db)), 2) if speech_db.size else None,
            "speech_ratio": rnd(float(speaking.mean()), 3) if speaking.size else None,
            **{f"pitch_{k}": v for k, v in semitone_stats(f0).items()},
        }
        return {
            "summary": summary,
            "pauses": pauses,
            "fillers": fillers,
            "wpm_windows": windows,
            "series": {
                "pitch_hz": per_second(p_t, f0, duration),
                "energy_db": per_second(e_t[speaking], e_db[speaking], duration) if speaking.any()
                else per_second(e_t, e_db, duration),
            },
            "models": {"pitch": pitch_backend},
        }
