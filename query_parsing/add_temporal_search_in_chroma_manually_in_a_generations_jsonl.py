#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
)

load_dotenv()

# =============================================================================
# Paths / env
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_BASE_PATH = os.getenv("CHROMA_BASE_PATH", "data/chroma")
COLLECTION_NAME = os.getenv("DEFAULT_CHROMA_INDEX", "chroma_index")
CHROMA_DIR = (PROJECT_ROOT / ".." / CHROMA_BASE_PATH / COLLECTION_NAME).resolve()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large").strip()

GOOGLE_EMBEDDING_API_KEY = os.getenv("GOOGLE_EMBEDDING_API_KEY", "").strip()
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
).strip()

EMBEDDING_REQUESTS_PER_MINUTE = int(os.getenv("EMBEDDING_REQUESTS_PER_MINUTE", "80"))
if EMBEDDING_REQUESTS_PER_MINUTE <= 0:
    raise ValueError("EMBEDDING_REQUESTS_PER_MINUTE doit être > 0")

# Retry avec intervalle exponentiel tronqué pour les erreurs temporaires
# typiquement 429 / RESOURCE_EXHAUSTED côté Google.
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "8"))
EMBEDDING_INITIAL_BACKOFF_SECONDS = float(os.getenv("EMBEDDING_INITIAL_BACKOFF_SECONDS", "1.0"))
EMBEDDING_MAX_BACKOFF_SECONDS = float(os.getenv("EMBEDDING_MAX_BACKOFF_SECONDS", "64.0"))
EMBEDDING_RETRY_DEADLINE_SECONDS = float(os.getenv("EMBEDDING_RETRY_DEADLINE_SECONDS", "300.0"))

if EMBEDDING_MAX_RETRIES < 0:
    raise ValueError("EMBEDDING_MAX_RETRIES doit être >= 0")
if EMBEDDING_INITIAL_BACKOFF_SECONDS <= 0:
    raise ValueError("EMBEDDING_INITIAL_BACKOFF_SECONDS doit être > 0")
if EMBEDDING_MAX_BACKOFF_SECONDS <= 0:
    raise ValueError("EMBEDDING_MAX_BACKOFF_SECONDS doit être > 0")
if EMBEDDING_RETRY_DEADLINE_SECONDS <= 0:
    raise ValueError("EMBEDDING_RETRY_DEADLINE_SECONDS doit être > 0")

# =============================================================================
# Conservative year-only temporal parser -> builds year_int filter for Chroma
# =============================================================================

MIN_REASONABLE_YEAR = 1700
CURRENT_YEAR = date.today().year
MAX_REASONABLE_YEAR = CURRENT_YEAR
ASSUME_TWO_DIGIT_DECADES_ARE_1900S = True

YEAR_PATTERN = r"(1\d{3}|20\d{2})"
MONTH_PATTERN = (
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)

SOURCE_NOUN_PATTERN = (
    r"(?:annonce|annonces|avis|registre|registres|publication|publications|"
    r"article|articles|journal|journaux|feuille|bulletin|bulletins|"
    r"acte|actes|affiche|affiches|édition|edition|numéro|numero|"
    r"inventaire|inventaires|rapport|rapports|registre municipal|"
    r"registres municipaux)"
)


def is_reasonable_year(year: Optional[int]) -> bool:
    return year is not None and MIN_REASONABLE_YEAR <= year <= MAX_REASONABLE_YEAR


def extract_reasonable_years(text: str) -> List[int]:
    years: List[int] = []
    for m in re.finditer(rf"\b{YEAR_PATTERN}\b", text):
        y = int(m.group(1))
        if is_reasonable_year(y):
            years.append(y)
    return years


def extract_last_reasonable_year(text: str) -> Optional[int]:
    years = extract_reasonable_years(text)
    return years[-1] if years else None


def build_year_int_filter(
    *,
    eq: Optional[int] = None,
    lt: Optional[int] = None,
    lte: Optional[int] = None,
    gt: Optional[int] = None,
    gte: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if eq is not None:
        if is_reasonable_year(eq):
            return {"year_int": eq}
        return None

    lower = MIN_REASONABLE_YEAR
    upper = CURRENT_YEAR

    if start is not None:
        lower = max(lower, start)
    if end is not None:
        upper = min(upper, end)

    if gt is not None:
        lower = max(lower, gt + 1)
    if gte is not None:
        lower = max(lower, gte)

    if lt is not None:
        upper = min(upper, lt - 1)
    if lte is not None:
        upper = min(upper, lte)

    if lower > upper:
        return {"year_int": {"$in": []}}

    return {"$and": [{"year_int": {"$gte": lower}}, {"year_int": {"$lte": upper}}]}


def build_time_filter_payload(
    *,
    eq: Optional[int] = None,
    lt: Optional[int] = None,
    lte: Optional[int] = None,
    gt: Optional[int] = None,
    gte: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    source_text: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    chroma_filter = build_year_int_filter(
        eq=eq, lt=lt, lte=lte, gt=gt, gte=gte, start=start, end=end
    )
    if chroma_filter is None:
        return None

    label = source_text
    if label is None:
        if eq is not None:
            label = f"in {eq}"
        elif lt is not None:
            label = f"before {lt}"
        elif gt is not None:
            label = f"after {gt}"
        elif start is not None and end is not None:
            label = f"between {start} and {end}"
        elif start is not None:
            label = f"from {start}"
        elif end is not None:
            label = f"up to {end}"

    return {
        "time_filter": chroma_filter,
        "cleaned_filter": label,
    }


def safe_year_eq_payload(year: Optional[int], source_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not is_reasonable_year(year):
        return None
    return build_time_filter_payload(eq=year, source_text=source_text or f"in {year}")


def try_century_filter(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower().strip()

    def century_bounds(century: int) -> Tuple[int, int]:
        start = (century - 1) * 100
        end = start + 99
        return start, end

    def build_single_century_result(start: int, end: int, century_label: str, full_text: str) -> Optional[Dict[str, Any]]:
        if re.search(r"\b(before|avant)\b", full_text):
            return build_time_filter_payload(
                start=MIN_REASONABLE_YEAR,
                end=min(start - 1, CURRENT_YEAR),
                source_text=f"before {century_label}",
            )

        if re.search(r"\b(after|après|apres)\b", full_text):
            return build_time_filter_payload(
                start=max(end + 1, MIN_REASONABLE_YEAR),
                end=CURRENT_YEAR,
                source_text=f"after {century_label}",
            )

        return build_time_filter_payload(
            start=max(start, MIN_REASONABLE_YEAR),
            end=min(end, CURRENT_YEAR),
            source_text=century_label,
        )

    m = re.search(
        r"\bbetween\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+and\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)\s+centur(?:y|ies)\b",
        t,
    )
    if m:
        c1, c2 = int(m.group(1)), int(m.group(2))
        if c1 <= c2:
            start = (c1 - 1) * 100
            end = (c2 - 1) * 100 + 99
            return build_time_filter_payload(
                start=max(start, MIN_REASONABLE_YEAR),
                end=min(end, CURRENT_YEAR),
                source_text=f"between {c1}th century and {c2}th century",
            )

    m = re.search(
        r"\bentre\s+(?:le\s+)?(\d{1,2})\s*(?:e|er)?\s+et\s+(?:le\s+)?(\d{1,2})\s*(?:e|er)?\s+siècle\b",
        t,
    )
    if m:
        c1, c2 = int(m.group(1)), int(m.group(2))
        if c1 <= c2:
            start = (c1 - 1) * 100
            end = (c2 - 1) * 100 + 99
            return build_time_filter_payload(
                start=max(start, MIN_REASONABLE_YEAR),
                end=min(end, CURRENT_YEAR),
                source_text=f"between {c1}th century and {c2}th century",
            )

    roman_map = {
        "xviie": 17,
        "xviiie": 18,
        "xixe": 19,
        "xxe": 20,
    }

    m = re.search(
        r"\bentre\s+(?:le\s+)?(xviie|xviiie|xixe|xxe)\s+et\s+(?:le\s+)?(xviie|xviiie|xixe|xxe)\s+siècle\b",
        t,
    )
    if m:
        c1 = roman_map[m.group(1)]
        c2 = roman_map[m.group(2)]
        if c1 <= c2:
            start = (c1 - 1) * 100
            end = (c2 - 1) * 100 + 99
            return build_time_filter_payload(
                start=max(start, MIN_REASONABLE_YEAR),
                end=min(end, CURRENT_YEAR),
                source_text=f"between {c1}th century and {c2}th century",
            )

    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+century\b", t)
    if m:
        century = int(m.group(1))
        start, end = century_bounds(century)
        if end >= MIN_REASONABLE_YEAR:
            return build_single_century_result(start, end, f"{century}th century", t)

    m = re.search(r"\b(\d{1,2})\s*(?:e|er)?\s+siècle\b", t)
    if m:
        century = int(m.group(1))
        start, end = century_bounds(century)
        if end >= MIN_REASONABLE_YEAR:
            return build_single_century_result(start, end, f"{century}th century", t)

    for roman, century in roman_map.items():
        if roman in t and "siècle" in t:
            start, end = century_bounds(century)
            if end >= MIN_REASONABLE_YEAR:
                return build_single_century_result(start, end, f"{century}th century", t)

    return None


def try_decade_filter(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower().strip()

    m = re.search(rf"\bann[eé]es\s+{YEAR_PATTERN}\b", t)
    if m:
        start = int(m.group(1))
        if start % 10 == 0 and is_reasonable_year(start):
            return build_time_filter_payload(
                start=start,
                end=start + 9,
                source_text=f"{start}s",
            )

    m = re.search(rf"\b{YEAR_PATTERN}s\b", t)
    if m:
        start = int(m.group(1))
        if start % 10 == 0 and is_reasonable_year(start):
            return build_time_filter_payload(
                start=start,
                end=start + 9,
                source_text=f"{start}s",
            )

    m = re.search(r"\bann[eé]es\s+(\d{2})\b", t)
    if m and ASSUME_TWO_DIGIT_DECADES_ARE_1900S:
        decade = int(m.group(1))
        start = 1900 + decade
        if start % 10 == 0 and is_reasonable_year(start):
            return build_time_filter_payload(
                start=start,
                end=start + 9,
                source_text=f"{start}s",
            )

    m = re.search(r"\b(\d{2})s\b", t)
    if m and ASSUME_TWO_DIGIT_DECADES_ARE_1900S:
        decade = int(m.group(1))
        start = 1900 + decade
        if start % 10 == 0 and is_reasonable_year(start):
            return build_time_filter_payload(
                start=start,
                end=start + 9,
                source_text=f"{start}s",
            )

    return None


def remove_time_expressions(text: str) -> str:
    cleaned = text

    patterns = [
        r"\bbetween\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+and\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+centur(?:y|ies)\b",
        r"\bentre\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+et\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bentre\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+et\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        r"\bbefore\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century\b",
        r"\bafter\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century\b",
        r"\bavant\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\b(?:après|apres)\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bentre\s+\d{4}\s+et\s+\d{4}\b",
        r"\bbetween\s+\d{4}\s+and\s+\d{4}\b",
        r"\bavant\s+\d{4}\b",
        r"\bbefore\s+\d{4}\b",
        r"\b(?:après|apres)\s+\d{4}\b",
        r"\bafter\s+\d{4}\b",
        r"\ben\s+\d{4}\b",
        r"\bin\s+\d{4}\b",
        r"\bann[eé]es\s+(?:\d{2}|1\d{3}|20\d{2})\b",
        r"\b(?:1\d{3}|20\d{2})s\b",
        r"\b\d{2}s\b",
        r"\b\d{1,2}(?:st|nd|rd|th)\s+century\b",
        r"\b\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\b(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        rf"\b(?:dat(?:e|é)e?|dated)\s+(?:du|de|le|on|of)\s+[^.?!;]{{0,40}}?{YEAR_PATTERN}\b",
        rf"\bdu\s+\d{{1,2}}\s+(?:au\s+\d{{1,2}}\s+)?{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        rf"\ble\s+\d{{1,2}}\s+{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        rf"\b{SOURCE_NOUN_PATTERN}\b(?:[^.?!;]{{0,40}}?)\bde\s+{YEAR_PATTERN}\b",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?;:")
    return cleaned


def try_general_regex_time_filter(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower().strip()

    m = re.search(rf"\bbetween\s+{YEAR_PATTERN}\s+and\s+{YEAR_PATTERN}\b", t)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if is_reasonable_year(y1) and is_reasonable_year(y2) and y1 <= y2:
            return build_time_filter_payload(
                start=y1,
                end=y2,
                source_text=f"between {y1} and {y2}",
            )

    m = re.search(rf"\bentre\s+{YEAR_PATTERN}\s+et\s+{YEAR_PATTERN}\b", t)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if is_reasonable_year(y1) and is_reasonable_year(y2) and y1 <= y2:
            return build_time_filter_payload(
                start=y1,
                end=y2,
                source_text=f"between {y1} and {y2}",
            )

    m = re.search(r"\b(?:before|avant)\b(?P<date_part>[^.?!;]{0,50})", t, flags=re.IGNORECASE)
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return build_time_filter_payload(
                lt=y,
                source_text=f"before {y}",
            )

    m = re.search(r"\b(?:after|après|apres)\b(?P<date_part>[^.?!;]{0,50})", t, flags=re.IGNORECASE)
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return build_time_filter_payload(
                gt=y,
                source_text=f"after {y}",
            )

    m = re.search(
        rf"\b(?:in|en)\s+(?P<date_part>(?:[^\W\d_]+\s+){{0,3}}{YEAR_PATTERN})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    m = re.search(
        rf"\b(?:dat(?:e|é)e?|dated)\s+(?:du|de|le|on|of)\s+[^.?!;]{{0,40}}?{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    m = re.search(
        rf"\bdu\s+\d{{1,2}}\s+(?:au\s+\d{{1,2}}\s+)?{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    m = re.search(
        rf"\ble\s+\d{{1,2}}\s+{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    m = re.search(
        rf"\b{SOURCE_NOUN_PATTERN}\b(?:[^.?!;]{{0,40}}?)\bde\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    years = sorted(set(extract_reasonable_years(t)))
    if len(years) == 1:
        has_context_cue = re.search(
            r"\b(?:selon|dans|d[’']après|dapres|according to)\b",
            t,
            flags=re.IGNORECASE,
        )
        has_source_noun = re.search(
            rf"\b{SOURCE_NOUN_PATTERN}\b",
            t,
            flags=re.IGNORECASE,
        )
        if has_context_cue and has_source_noun:
            y = years[0]
            return safe_year_eq_payload(y, f"in {y}")

    return None


def parse_query_for_temporal_chroma(query: str) -> Dict[str, Any]:
    century_result = try_century_filter(query)
    if century_result is not None:
        return {
            "original_query": query,
            "clean_query": remove_time_expressions(query) or query,
            "time_filter": century_result["time_filter"],
            "cleaned_filter": century_result["cleaned_filter"],
            "filter_source": "century_regex",
        }

    decade_result = try_decade_filter(query)
    if decade_result is not None:
        return {
            "original_query": query,
            "clean_query": remove_time_expressions(query) or query,
            "time_filter": decade_result["time_filter"],
            "cleaned_filter": decade_result["cleaned_filter"],
            "filter_source": "decade_regex",
        }

    general_result = try_general_regex_time_filter(query)
    if general_result is not None:
        return {
            "original_query": query,
            "clean_query": remove_time_expressions(query) or query,
            "time_filter": general_result["time_filter"],
            "cleaned_filter": general_result["cleaned_filter"],
            "filter_source": "general_regex",
        }

    return {
        "original_query": query,
        "clean_query": query,
        "time_filter": None,
        "cleaned_filter": None,
        "filter_source": None,
    }

# =============================================================================
# Embeddings / rate limit
# =============================================================================

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute doit être > 0")
        self.min_interval = 60.0 / requests_per_minute
        self.last_call_time: Optional[float] = None

    def wait(self) -> float:
        now = time.monotonic()

        if self.last_call_time is None:
            self.last_call_time = now
            return 0.0

        elapsed = now - self.last_call_time
        remaining = self.min_interval - elapsed

        waited = 0.0
        if remaining > 0:
            time.sleep(remaining)
            waited = remaining

        self.last_call_time = time.monotonic()
        return waited


def resolve_embedding_provider() -> str:
    provider = EMBEDDING_PROVIDER

    if not provider:
        lowered = COLLECTION_NAME.lower()
        if "gemini" in lowered or "google" in lowered:
            provider = "google"
        elif "openai" in lowered:
            provider = "openai"

    if provider not in {"openai", "google"}:
        raise ValueError(
            f"EMBEDDING_PROVIDER invalide: '{EMBEDDING_PROVIDER}'. "
            f"Valeurs attendues: 'openai' ou 'google'."
        )
    return provider


def resolve_embedding_model(provider: str) -> str:
    if provider == "openai":
        return OPENAI_EMBEDDING_MODEL
    if provider == "google":
        return GOOGLE_EMBEDDING_MODEL
    raise ValueError(f"Provider non supporté: {provider}")


def build_embedding_function():
    provider = resolve_embedding_provider()

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY manquant dans le .env")
        return OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_EMBEDDING_MODEL,
        )

    if provider == "google":
        if not GOOGLE_EMBEDDING_API_KEY:
            raise ValueError("GOOGLE_EMBEDDING_API_KEY manquant dans le .env")
        return GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_EMBEDDING_API_KEY,
            model_name=GOOGLE_EMBEDDING_MODEL,
        )

    raise ValueError("Provider d'embedding non supporté.")


def is_retryable_embedding_error(exc: BaseException) -> bool:
    """
    Retourne True pour les erreurs temporaires côté provider.
    Le cas principal ici est Google 429 / RESOURCE_EXHAUSTED.
    On reste volontairement large pour couvrir aussi les timeouts et 503.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    retryable_markers = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "ratelimit",
        "too many requests",
        "temporarily unavailable",
        "unavailable",
        "deadline_exceeded",
        "deadline exceeded",
        "timeout",
        "timed out",
        "503",
        "500",
    )
    return any(marker in text for marker in retryable_markers)


def call_embedding_with_retry(embedding_fn, text: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Appelle l'API d'embedding avec exponential backoff tronqué + jitter.

    Important : une tentative qui échoue puis réussit après retry n'est PAS
    comptée comme une erreur finale. Elle est simplement signalée via
    retry_count et succeeded_after_retry.
    """
    retry_count = 0
    retry_sleep_seconds = 0.0
    first_error_type: Optional[str] = None
    first_error_message: Optional[str] = None
    started_at = time.monotonic()

    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        try:
            embedding = embedding_fn([text])
            return embedding, {
                "retry_count": retry_count,
                "retry_sleep_ms": round(retry_sleep_seconds * 1000.0, 2),
                "succeeded_after_retry": retry_count > 0,
                "first_error_type": first_error_type,
                "first_error_message": first_error_message,
            }
        except Exception as exc:
            if first_error_type is None:
                first_error_type = type(exc).__name__
                first_error_message = str(exc)

            elapsed = time.monotonic() - started_at
            retries_left = EMBEDDING_MAX_RETRIES - attempt

            if retries_left <= 0 or not is_retryable_embedding_error(exc):
                raise

            # Intervalle exponentiel tronqué : 1, 2, 4, 8... plafonné,
            # avec jitter aléatoire pour éviter de retomber en même temps.
            base_delay = min(
                EMBEDDING_INITIAL_BACKOFF_SECONDS * (2 ** attempt),
                EMBEDDING_MAX_BACKOFF_SECONDS,
            )
            delay = base_delay + random.uniform(0.0, 1.0)

            remaining_deadline = EMBEDDING_RETRY_DEADLINE_SECONDS - elapsed
            if remaining_deadline <= 0:
                raise
            delay = min(delay, remaining_deadline)

            retry_count += 1
            retry_sleep_seconds += delay
            print(
                f"[RETRY embedding] tentative {attempt + 1} échouée "
                f"({type(exc).__name__}: {str(exc)[:250]}). "
                f"Nouvelle tentative dans {delay:.2f}s..."
            )
            time.sleep(delay)

    raise RuntimeError("Retry embedding terminé dans un état inattendu.")


def extract_query_embedding(
    embedding_fn,
    text: str,
    rate_limiter: RateLimiter,
) -> Tuple[List[float], Dict[str, Any]]:
    wait_seconds = rate_limiter.wait()
    rate_limit_wait_ms = wait_seconds * 1000.0

    embedding_api_start = time.perf_counter()
    embedding, retry_meta = call_embedding_with_retry(embedding_fn, text)
    embedding_api_ms = (time.perf_counter() - embedding_api_start) * 1000.0

    if embedding is None:
        raise ValueError("Embedding vide renvoyé par la fonction d'embedding.")

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    if isinstance(embedding, list) and len(embedding) == 0:
        raise ValueError("Embedding vide renvoyé par la fonction d'embedding.")

    first = embedding[0] if isinstance(embedding, list) and len(embedding) > 0 else None

    if hasattr(first, "tolist"):
        first = first.tolist()

    if isinstance(first, list):
        final_embedding = [float(x) for x in first]
    elif isinstance(embedding, list) and isinstance(first, (int, float)):
        final_embedding = [float(x) for x in embedding]
    else:
        raise ValueError(
            f"Format d'embedding inattendu. Type={type(embedding)}, valeur={repr(embedding)[:500]}"
        )

    timing = {
        "rate_limit_wait_ms": round(rate_limit_wait_ms, 2),
        "embedding_api_ms": round(embedding_api_ms, 2),
        "embedding_retry_count": int(retry_meta["retry_count"]),
        "embedding_retry_sleep_ms": float(retry_meta["retry_sleep_ms"]),
        "embedding_succeeded_after_retry": bool(retry_meta["succeeded_after_retry"]),
        "embedding_first_error_type": retry_meta.get("first_error_type"),
        "embedding_first_error_message": retry_meta.get("first_error_message"),
    }
    return final_embedding, timing


# =============================================================================
# Embedding file helpers
# =============================================================================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def build_embedding_key(text: str, provider: str, model: str) -> str:
    raw = f"{provider}::{model}::{normalize_text(text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_embeddings_file(path: Path) -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL invalide dans le fichier embeddings {path} à la ligne {line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                continue

            key = str(row.get("key", "")).strip()
            embedding = row.get("embedding")

            if key and isinstance(embedding, list) and embedding:
                cache[key] = [float(x) for x in embedding]

    return cache


def get_embedding_with_optional_file_cache(
    *,
    embedding_fn,
    text: str,
    rate_limiter: RateLimiter,
    provider: str,
    model: str,
    embedding_file_cache: Optional[Dict[str, List[float]]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    key = build_embedding_key(text, provider, model)

    if embedding_file_cache is not None and key in embedding_file_cache:
        return embedding_file_cache[key], {
            "source": "file_cache",
            "cache_key": key,
            "rate_limit_wait_ms": 0.0,
            "embedding_api_ms": 0.0,
            "embedding_retry_count": 0,
            "embedding_retry_sleep_ms": 0.0,
            "embedding_succeeded_after_retry": False,
            "embedding_first_error_type": None,
            "embedding_first_error_message": None,
        }

    embedding, timing = extract_query_embedding(
        embedding_fn=embedding_fn,
        text=text,
        rate_limiter=rate_limiter,
    )
    return embedding, {
        "source": "api",
        "cache_key": key,
        "rate_limit_wait_ms": timing["rate_limit_wait_ms"],
        "embedding_api_ms": timing["embedding_api_ms"],
        "embedding_retry_count": timing["embedding_retry_count"],
        "embedding_retry_sleep_ms": timing["embedding_retry_sleep_ms"],
        "embedding_succeeded_after_retry": timing["embedding_succeeded_after_retry"],
        "embedding_first_error_type": timing["embedding_first_error_type"],
        "embedding_first_error_message": timing["embedding_first_error_message"],
    }


# =============================================================================
# JSONL / Chroma helpers
# =============================================================================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL invalide dans {path} à la ligne {line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_retrieved_chunks_from_chroma(
    results: Dict[str, Any],
    retrieval_variant: str,
) -> List[Dict[str, Any]]:
    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    chunks: List[Dict[str, Any]] = []

    retrieval_source = "temporal" if retrieval_variant == "temporal" else "chroma"

    for rank, doc in enumerate(documents):
        meta = metadatas[rank] if rank < len(metadatas) and isinstance(metadatas[rank], dict) else {}
        meta_out = dict(meta)
        meta_out["retrieval_source"] = retrieval_source
        meta_out["retrieval_variant"] = retrieval_variant
        meta_out["retrieval_rank"] = rank

        chunk = {
            "id": ids[rank] if rank < len(ids) else None,
            "text": doc,
            "metadata": meta_out,
            "distance": distances[rank] if rank < len(distances) else None,
            "rank": rank,
            "retrieval_rank": rank,
            "retrieval_source": retrieval_source,
        }
        chunks.append(chunk)

    return chunks

# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajoute des chunks Chroma temporels dans un generations.jsonl existant."
    )
    parser.add_argument("--input-jsonl", required=True, help="Fichier generations.jsonl existant")
    parser.add_argument("--output-jsonl", required=True, help="Fichier JSONL de sortie")
    parser.add_argument("--top-k-temporal", type=int, default=100, help="Nombre de chunks Chroma temporels à ajouter")
    parser.add_argument(
        "--embeddings-file",
        required=False,
        help="Fichier JSONL d'embeddings pré-générés. Si absent, comportement identique à aujourd'hui.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    embeddings_file_path = Path(args.embeddings_file) if args.embeddings_file else None

    provider = resolve_embedding_provider()
    model = resolve_embedding_model(provider)

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"INPUT_JSONL = {input_path}")
    print(f"OUTPUT_JSONL = {output_path}")
    print(f"EMBEDDINGS_FILE = {embeddings_file_path}")
    print(f"CHROMA_DIR = {CHROMA_DIR}")
    print(f"COLLECTION_NAME = {COLLECTION_NAME}")
    print(f"EMBEDDING_PROVIDER = {provider}")
    print(f"EMBEDDING_MODEL = {model}")
    print(f"TOP_K_TEMPORAL = {args.top_k_temporal}")

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    if embeddings_file_path is not None and not embeddings_file_path.exists():
        raise FileNotFoundError(f"Fichier embeddings introuvable: {embeddings_file_path}")

    if not CHROMA_DIR.exists():
        raise ValueError(f"Le dossier Chroma n'existe pas: {CHROMA_DIR}")

    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError("Aucune ligne valide dans le fichier d'entrée.")

    embedding_fn = build_embedding_function()
    rate_limiter = RateLimiter(EMBEDDING_REQUESTS_PER_MINUTE)

    embedding_file_cache = None
    if embeddings_file_path is not None:
        embedding_file_cache = load_embeddings_file(embeddings_file_path)
        print(f"Embeddings chargés depuis le fichier : {len(embedding_file_cache)}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collections = client.list_collections()
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        raise ValueError(
            f"La collection '{COLLECTION_NAME}' n'existe pas dans '{CHROMA_DIR}'. "
            f"Collections disponibles: {collection_names}"
        )

    collection = client.get_collection(name=COLLECTION_NAME)

    out_rows: List[Dict[str, Any]] = []

    temporal_queries_run = 0
    temporal_queries_skipped = 0
    total_added_chunks = 0
    total_embedding_api_ms = 0.0
    total_rate_limit_wait_ms = 0.0
    total_chroma_temporal_ms = 0.0
    total_embedding_file_hits = 0
    total_embedding_api_calls = 0
    total_embedding_retry_successes = 0
    total_embedding_retry_attempts = 0
    total_embedding_retry_sleep_ms = 0.0
    temporal_final_failures = 0
    final_error_rows: List[Dict[str, Any]] = []

    for row in tqdm(rows, desc="Temporal Chroma augmentation", unit="q"):
        out = dict(row)
        question = str(out.get("question", "") or "").strip()

        existing_chunks = safe_list(out.get("retrieved_chunks"))
        method_metadata = safe_dict(out.get("method_metadata"))
        latency_ms = safe_dict(out.get("latency_ms"))

        if not question:
            temporal_queries_skipped += 1
            method_metadata["temporal_chroma_augmentation"] = {
                "applied": False,
                "reason": "empty_question",
                "added_chunks_count": 0,
            }
            out["method_metadata"] = method_metadata
            out_rows.append(out)
            continue

        temporal_info = parse_query_for_temporal_chroma(question)
        temporal_filter = temporal_info.get("time_filter")
        temporal_clean_query = str(temporal_info.get("clean_query", question) or question).strip()

        if temporal_filter is None:
            temporal_queries_skipped += 1
            method_metadata["temporal_chroma_augmentation"] = {
                "applied": False,
                "reason": "no_temporal_expression_detected",
                "time_filter": None,
                "cleaned_filter": None,
                "filter_source": None,
                "added_chunks_count": 0,
            }
            out["method_metadata"] = method_metadata
            out_rows.append(out)
            continue

        temporal_queries_run += 1

        query_for_embedding = temporal_clean_query if temporal_clean_query else question

        try:
            temporal_embedding, embedding_meta = get_embedding_with_optional_file_cache(
                embedding_fn=embedding_fn,
                text=query_for_embedding,
                rate_limiter=rate_limiter,
                provider=provider,
                model=model,
                embedding_file_cache=embedding_file_cache,
            )

            temporal_embedding_api_ms = float(embedding_meta["embedding_api_ms"])
            temporal_rate_limit_wait_ms = float(embedding_meta["rate_limit_wait_ms"])
            temporal_embedding_retry_count = int(embedding_meta.get("embedding_retry_count", 0) or 0)
            temporal_embedding_retry_sleep_ms = float(embedding_meta.get("embedding_retry_sleep_ms", 0.0) or 0.0)

            if embedding_meta["source"] == "file_cache":
                total_embedding_file_hits += 1
            else:
                total_embedding_api_calls += 1

            if bool(embedding_meta.get("embedding_succeeded_after_retry", False)):
                total_embedding_retry_successes += 1
            total_embedding_retry_attempts += temporal_embedding_retry_count
            total_embedding_retry_sleep_ms += temporal_embedding_retry_sleep_ms

            chroma_temporal_start = time.perf_counter()
            temporal_results = collection.query(
                query_embeddings=[temporal_embedding],
                n_results=args.top_k_temporal,
                include=["documents", "distances", "metadatas"],
                where=temporal_filter,
            )
            chroma_temporal_ms = (time.perf_counter() - chroma_temporal_start) * 1000.0

            temporal_chunks = build_retrieved_chunks_from_chroma(
                temporal_results,
                retrieval_variant="temporal",
            )
        except Exception as exc:
            temporal_final_failures += 1
            temporal_queries_skipped += 1
            error_row = {
                "question": question,
                "time_filter": temporal_filter,
                "cleaned_filter": temporal_info.get("cleaned_filter"),
                "filter_source": temporal_info.get("filter_source"),
                "temporal_clean_query": query_for_embedding,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            final_error_rows.append(error_row)

            method_metadata["temporal_chroma_augmentation"] = {
                "applied": False,
                "reason": "temporal_embedding_or_chroma_failed_after_retries",
                "time_filter": temporal_filter,
                "cleaned_filter": temporal_info.get("cleaned_filter"),
                "filter_source": temporal_info.get("filter_source"),
                "temporal_clean_query": query_for_embedding,
                "added_chunks_count": 0,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            out["method_metadata"] = method_metadata
            out["latency_ms"] = latency_ms
            out_rows.append(out)
            continue

        out["retrieved_chunks"] = existing_chunks + temporal_chunks

        method_metadata["temporal_chroma_augmentation"] = {
            "applied": True,
            "time_filter": temporal_filter,
            "cleaned_filter": temporal_info.get("cleaned_filter"),
            "filter_source": temporal_info.get("filter_source"),
            "temporal_clean_query": query_for_embedding,
            "top_k_temporal": args.top_k_temporal,
            "added_chunks_count": len(temporal_chunks),
            "embedding_source": embedding_meta["source"],
            "embedding_cache_key": embedding_meta["cache_key"],
            "embedding_retry_count": temporal_embedding_retry_count,
            "embedding_retry_sleep_ms": round(temporal_embedding_retry_sleep_ms, 2),
            "embedding_succeeded_after_retry": bool(embedding_meta.get("embedding_succeeded_after_retry", False)),
            "embedding_first_error_type": embedding_meta.get("embedding_first_error_type"),
            "embedding_first_error_message": embedding_meta.get("embedding_first_error_message"),
        }
        out["method_metadata"] = method_metadata

        latency_ms["temporal_rate_limit_wait_ms"] = round(
            float(latency_ms.get("temporal_rate_limit_wait_ms", 0.0) or 0.0) + temporal_rate_limit_wait_ms,
            2,
        )
        latency_ms["temporal_embedding_api_ms"] = round(
            float(latency_ms.get("temporal_embedding_api_ms", 0.0) or 0.0) + temporal_embedding_api_ms,
            2,
        )
        latency_ms["temporal_embedding_retry_sleep_ms"] = round(
            float(latency_ms.get("temporal_embedding_retry_sleep_ms", 0.0) or 0.0) + temporal_embedding_retry_sleep_ms,
            2,
        )
        latency_ms["chroma_temporal_query_ms"] = round(
            float(latency_ms.get("chroma_temporal_query_ms", 0.0) or 0.0) + chroma_temporal_ms,
            2,
        )
        latency_ms["real_pipeline_ms"] = round(
            float(latency_ms.get("real_pipeline_ms", 0.0) or 0.0)
            + temporal_embedding_api_ms
            + chroma_temporal_ms,
            2,
        )
        out["latency_ms"] = latency_ms

        total_added_chunks += len(temporal_chunks)
        total_embedding_api_ms += temporal_embedding_api_ms
        total_rate_limit_wait_ms += temporal_rate_limit_wait_ms
        total_chroma_temporal_ms += chroma_temporal_ms

        out_rows.append(out)

    write_jsonl(output_path, out_rows)

    error_output_path = output_path.with_name(f"{output_path.stem}.errors.jsonl")
    if final_error_rows:
        write_jsonl(error_output_path, final_error_rows)

    print(f"\nFichier généré : {output_path}")
    if final_error_rows:
        print(f"Fichier d'erreurs finales : {error_output_path}")
    print("\n===== RÉSUMÉ GLOBAL =====")
    print(f"Questions traitées                : {len(out_rows)}")
    print(f"Requêtes temporelles exécutées    : {temporal_queries_run}")
    print(f"Requêtes temporelles ignorées     : {temporal_queries_skipped}")
    print(f"Embeddings depuis fichier         : {total_embedding_file_hits}")
    print(f"Embeddings via API                : {total_embedding_api_calls}")
    print(f"Embeddings réussis après retry    : {total_embedding_retry_successes}")
    print(f"Tentatives retry embedding        : {total_embedding_retry_attempts}")
    print(f"Échecs finaux temporels           : {temporal_final_failures}")
    print(f"Chunks temporels ajoutés          : {total_added_chunks}")
    print(f"Somme rate limit wait             : {total_rate_limit_wait_ms:.2f} ms ({total_rate_limit_wait_ms / 1000:.2f} s)")
    print(f"Somme sleep retries embedding     : {total_embedding_retry_sleep_ms:.2f} ms ({total_embedding_retry_sleep_ms / 1000:.2f} s)")
    print(f"Somme embedding API temporel      : {total_embedding_api_ms:.2f} ms ({total_embedding_api_ms / 1000:.2f} s)")
    print(f"Somme requêtes Chroma temporelles : {total_chroma_temporal_ms:.2f} ms ({total_chroma_temporal_ms / 1000:.2f} s)")

    if len(out_rows) > 0:
        print("\n===== MOYENNES PAR QUESTION =====")
        print(f"Moyenne chunks temporels ajoutés  : {total_added_chunks / len(out_rows):.2f}")
        print(f"Moyenne embedding API temporel    : {total_embedding_api_ms / len(out_rows):.2f} ms")
        print(f"Moyenne requête Chroma temporelle : {total_chroma_temporal_ms / len(out_rows):.2f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())