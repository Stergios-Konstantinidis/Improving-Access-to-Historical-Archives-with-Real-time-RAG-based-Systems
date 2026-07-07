import os
import re
from typing import Any, Dict, Optional
from datetime import date

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_env(env_path: Optional[str] = None) -> None:
    """
    Wrapper demandé par le projet.
    Charge les variables depuis un fichier .env.
    """
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path, override=False)


load_env()

print("LOADED time_filter_parser.py - REGEX ONLY VERSION (CONSERVATIVE YEAR-ONLY)")

MIN_REASONABLE_YEAR = 1700
CURRENT_YEAR = date.today().year
MAX_REASONABLE_YEAR = CURRENT_YEAR
ASSUME_TWO_DIGIT_DECADES_ARE_1900S = True

# Valeurs attendues :
# - "year"     -> filtre sur le champ string "year"
# - "year_int" -> filtre sur le champ numérique "year_int"
CHROMA_TIME_FILTER_MODE = os.getenv("CHROMA_TIME_FILTER_MODE", "year").strip().lower()

YEAR_PATTERN = r"(1\d{3}|20\d{2})"
MONTH_PATTERN = (
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    r"septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december)"
)

# Noms “sources / documents” qui rendent un motif "de 1913" raisonnablement fiable.
# On évite volontairement des mots comme "résultats", pour ne pas filtrer trop agressivement
# des questions du type "... comparé à ses résultats de 1974 ?"
SOURCE_NOUN_PATTERN = (
    r"(?:annonce|annonces|avis|registre|registres|publication|publications|"
    r"article|articles|journal|journaux|feuille|bulletin|bulletins|"
    r"acte|actes|affiche|affiches|édition|edition|numéro|numero|"
    r"inventaire|inventaires|rapport|rapports|registre municipal|"
    r"registres municipaux)"
)


def is_reasonable_year(year: Optional[int]) -> bool:
    return year is not None and MIN_REASONABLE_YEAR <= year <= MAX_REASONABLE_YEAR


def build_year_string_values(start_year: int, end_year_inclusive: int) -> list[str]:
    if start_year > end_year_inclusive:
        return []
    return [str(y) for y in range(start_year, end_year_inclusive + 1)]


def extract_reasonable_years(text: str) -> list[int]:
    years = []
    for m in re.finditer(rf"\b{YEAR_PATTERN}\b", text):
        y = int(m.group(1))
        if is_reasonable_year(y):
            years.append(y)
    return years


def extract_last_reasonable_year(text: str) -> Optional[int]:
    years = extract_reasonable_years(text)
    return years[-1] if years else None


def safe_year_eq_payload(year: Optional[int], source_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not is_reasonable_year(year):
        return None
    return build_time_filter_payload(
        eq=year,
        source_text=source_text or f"in {year}",
    )


def build_year_numeric_filter(
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

    clauses = []

    if lower is not None:
        clauses.append({"year_int": {"$gte": lower}})

    if upper is not None:
        clauses.append({"year_int": {"$lte": upper}})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def build_year_string_filter(
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
            return {"year": str(eq)}
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
        return {"year": {"$in": []}}

    return {"year": {"$in": build_year_string_values(lower, upper)}}


def build_year_filter(
    *,
    eq: Optional[int] = None,
    lt: Optional[int] = None,
    lte: Optional[int] = None,
    gt: Optional[int] = None,
    gte: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    mode = CHROMA_TIME_FILTER_MODE

    if mode == "year_int":
        return build_year_numeric_filter(
            eq=eq,
            lt=lt,
            lte=lte,
            gt=gt,
            gte=gte,
            start=start,
            end=end,
        )

    return build_year_string_filter(
        eq=eq,
        lt=lt,
        lte=lte,
        gt=gt,
        gte=gte,
        start=start,
        end=end,
    )


def build_year_range_filter(start_year: int, end_year_exclusive: int) -> Optional[Dict[str, Any]]:
    return build_year_string_filter(start=start_year, end=end_year_exclusive - 1)


def recursively_convert_year_ints_to_strings(obj: Any) -> Any:
    if isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            if key == "year":
                if isinstance(value, int):
                    converted[key] = str(value)
                elif isinstance(value, list):
                    converted[key] = [str(v) if isinstance(v, int) else v for v in value]
                elif isinstance(value, dict):
                    inner = {}
                    for op, inner_value in value.items():
                        if isinstance(inner_value, int):
                            inner[op] = str(inner_value)
                        elif isinstance(inner_value, list):
                            inner[op] = [str(v) if isinstance(v, int) else v for v in inner_value]
                        else:
                            inner[op] = inner_value
                    converted[key] = inner
                else:
                    converted[key] = value
            else:
                converted[key] = recursively_convert_year_ints_to_strings(value)
        return converted

    if isinstance(obj, list):
        return [recursively_convert_year_ints_to_strings(v) for v in obj]

    return obj


def build_cleaned_time_filter_label(
    *,
    eq: Optional[int] = None,
    lt: Optional[int] = None,
    lte: Optional[int] = None,
    gt: Optional[int] = None,
    gte: Optional[int] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    source_text: Optional[str] = None,
) -> Optional[str]:
    if eq is not None:
        return f"in {eq}"

    if lt is not None:
        return f"before {lt}"

    if lte is not None:
        return f"up to {lte}"

    if gt is not None:
        return f"after {gt}"

    if gte is not None:
        return f"from {gte}"

    if start is not None and end is not None:
        if start == end:
            return f"in {start}"
        return f"between {start} and {end}"

    if start is not None:
        return f"from {start}"

    if end is not None:
        return f"up to {end}"

    return source_text


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
    chroma_filter = build_year_filter(
        eq=eq,
        lt=lt,
        lte=lte,
        gt=gt,
        gte=gte,
        start=start,
        end=end,
    )

    if chroma_filter is None:
        return None

    cleaned_filter = build_cleaned_time_filter_label(
        eq=eq,
        lt=lt,
        lte=lte,
        gt=gt,
        gte=gte,
        start=start,
        end=end,
        source_text=source_text,
    )

    return {
        "time_filter": chroma_filter,
        "cleaned_filter": cleaned_filter,
    }


def try_century_filter(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower().strip()

    def century_bounds(century: int) -> tuple[int, int]:
        start = (century - 1) * 100
        end = start + 99
        return start, end

    def build_single_century_result(
        start: int,
        end: int,
        century_label: str,
        full_text: str
    ) -> Optional[Dict[str, Any]]:
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
        r"\bduring\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century\b",
        r"\bin\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)\s+century\b",
        r"\bavant\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\b(?:après|apres)\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bpendant\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bau\s+\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bdans\s+(?:le\s+)?\d{1,2}\s*(?:e|er)?\s+siècle\b",
        r"\bavant\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        r"\b(?:après|apres)\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        r"\bpendant\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        r"\bau\s+(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
        r"\bdans\s+(?:le\s+)?(?:xviie|xviiie|xixe|xxe)\s+siècle\b",
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

    # 1) Entre deux années
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

    # 2) Avant / après
    m = re.search(
        r"\b(?:before|avant)\b(?P<date_part>[^.?!;]{0,50})",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return build_time_filter_payload(
                lt=y,
                source_text=f"before {y}",
            )

    m = re.search(
        r"\b(?:after|après|apres)\b(?P<date_part>[^.?!;]{0,50})",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return build_time_filter_payload(
                gt=y,
                source_text=f"after {y}",
            )

    # 3) En 1890 / in 1890
    m = re.search(
        rf"\b(?:in|en)\s+(?P<date_part>(?:[^\W\d_]+\s+){{0,3}}{YEAR_PATTERN})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group("date_part"))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    # 4) datée du 25 février 1825 / daté du 11 août 1854 / dated on ...
    m = re.search(
        rf"\b(?:dat(?:e|é)e?|dated)\s+(?:du|de|le|on|of)\s+[^.?!;]{{0,40}}?{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    # 5) du 15 mai 1830 / du 22 au 28 août 1855
    m = re.search(
        rf"\bdu\s+\d{{1,2}}\s+(?:au\s+\d{{1,2}}\s+)?{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    # 6) le 10 janvier 1858
    m = re.search(
        rf"\ble\s+\d{{1,2}}\s+{MONTH_PATTERN}\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    # 7) annonce de 1913 / publication de 1891 / registre de 1807 / avis public de 1794
    # Prudence : on ne prend ce cas que si l'année suit un nom "source/document"
    m = re.search(
        rf"\b{SOURCE_NOUN_PATTERN}\b(?:[^.?!;]{{0,40}}?)\bde\s+{YEAR_PATTERN}\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        y = extract_last_reasonable_year(m.group(0))
        if y is not None:
            return safe_year_eq_payload(y, f"in {y}")

    # 8) Fallback prudent :
    # s'il n'y a qu'une seule année ET que la question contient à la fois
    # un marqueur de cadrage ("selon", "dans", etc.) + un nom de source/document,
    # on accepte cette année.
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


def merge_chroma_filters(
    existing_filters: Optional[Dict[str, Any]],
    new_filter: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not existing_filters:
        return new_filter
    if not new_filter:
        return existing_filters
    return {"$and": [existing_filters, new_filter]}


def parse_query_for_chroma(query: str) -> Dict[str, Any]:
    print("parse_query_for_chroma called with:", repr(query))

    century_result = try_century_filter(query)
    if century_result is not None:
        return {
            "original_query": query,
            # Si tu veux vraiment retirer l'expression temporelle du texte de recherche,
            # décommente la ligne suivante et commente "clean_query": query
            # "clean_query": remove_time_expressions(query) or query,
            "clean_query": query,
            "time_filter": century_result["time_filter"],
            "cleaned_filter": century_result["cleaned_filter"],
            "filter_source": "century_regex",
        }

    decade_result = try_decade_filter(query)
    if decade_result is not None:
        return {
            "original_query": query,
            # "clean_query": remove_time_expressions(query) or query,
            "clean_query": query,
            "time_filter": decade_result["time_filter"],
            "cleaned_filter": decade_result["cleaned_filter"],
            "filter_source": "decade_regex",
        }

    general_result = try_general_regex_time_filter(query)
    if general_result is not None:
        return {
            "original_query": query,
            # "clean_query": remove_time_expressions(query) or query,
            "clean_query": query,
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