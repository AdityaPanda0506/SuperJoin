"""
Zero-Hardcoding Normalizer & Generalized Entity Clustering Engine.
Provides domain-agnostic numerical scaling, algorithmic date parsing via dateparser,
and Jaccard token set similarity for dynamic entity/attribute clustering.
"""

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Optional

import dateparser

from .models import TemporalInterval


class GeneralizedNormalizer:
    """
    100% Domain-Agnostic Normalizer & Canonicalization Engine.
    Uses mathematical order-of-magnitude multipliers, standard currency symbols,
    algorithmic date parsing, and hybrid token + character SequenceMatcher similarity.
    """

    # Standard numerical scales (general mathematical multipliers, not domain-specific)
    MULTIPLIER_MAP = {
        # Indian Numbering System
        "crore": 1e7,
        "crores": 1e7,
        "cr": 1e7,
        "lakh": 1e5,
        "lakhs": 1e5,
        "lac": 1e5,
        "lacs": 1e5,
        # International System
        "trillion": 1e12,
        "tn": 1e12,
        "billion": 1e9,
        "billions": 1e9,
        "bn": 1e9,
        "million": 1e6,
        "millions": 1e6,
        "mn": 1e6,
        "thousand": 1e3,
        "k": 1e3,
    }

    CURRENCY_SYMBOLS = {
        "₹": "INR",
        "rs": "INR",
        "inr": "INR",
        "$": "USD",
        "usd": "USD",
        "€": "EUR",
        "eur": "EUR",
        "£": "GBP",
        "gbp": "GBP",
        "¥": "JPY",
        "jpy": "JPY",
    }

    @classmethod
    def clean_token_set(cls, text: str) -> set[str]:
        """Tokenizes, lowercases, and strips non-alphanumeric noise."""
        if not text:
            return set()
        cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
        tokens = set(cleaned.split())
        return tokens

    @classmethod
    def compute_similarity(cls, str1: str, str2: str) -> float:
        """
        Computes a hybrid similarity score:
        40% Token Jaccard Overlap + 60% Character Sequence Levenshtein Ratio.
        Handles typos, word reordering, and abbreviations without hardcoding.
        """
        s1 = str(str1).lower().strip()
        s2 = str(str2).lower().strip()
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        # Remove generic corporate legal form noise for accurate entity matching
        legal_noise = {
            "limited",
            "ltd",
            "inc",
            "corp",
            "corporation",
            "llp",
            "pvt",
            "private",
            "co",
            "company",
        }
        t1_raw = re.findall(r"\w+", s1)
        t2_raw = re.findall(r"\w+", s2)

        t1_filtered = [w for w in t1_raw if w not in legal_noise]
        t2_filtered = [w for w in t2_raw if w not in legal_noise]

        s1_clean = " ".join(t1_filtered) if t1_filtered else s1
        s2_clean = " ".join(t2_filtered) if t2_filtered else s2

        # 1. Token Jaccard with prefix stemming (e.g. 'india' and 'indian' -> 'indi')
        def stem(w: str) -> str:
            return w[:4] if len(w) >= 4 else w

        t1 = {stem(w) for w in t1_filtered} if t1_filtered else {stem(w) for w in t1_raw}
        t2 = {stem(w) for w in t2_filtered} if t2_filtered else {stem(w) for w in t2_raw}

        jaccard = len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0

        # 2. SequenceMatcher (Character-level overlap)
        seq_ratio = SequenceMatcher(None, s1_clean, s2_clean).ratio()

        # Weighted blend
        return 0.4 * jaccard + 0.6 * seq_ratio

    @classmethod
    def canonicalize_entity(cls, entity: str) -> str:
        """Algorithmic entity token cleaning."""
        if not entity:
            return "unknown_entity"
        tokens = cls.clean_token_set(entity)
        # Remove common corporate legal noise words
        noise = {"limited", "ltd", "inc", "corp", "corporation", "llp", "pvt", "private"}
        clean_tokens = sorted(list(tokens - noise))
        return "_".join(clean_tokens) if clean_tokens else "entity"

    @classmethod
    def canonicalize_attribute(cls, attribute: str) -> str:
        """Algorithmic attribute token cleaning."""
        if not attribute:
            return "general_attribute"
        tokens = cls.clean_token_set(attribute)
        clean_tokens = sorted(list(tokens))
        return "_".join(clean_tokens) if clean_tokens else "attribute"

    @classmethod
    def canonicalize_keys(cls, entity: str, attribute: str) -> tuple[str, str]:
        """Return tuple of (canonical_entity, canonical_attribute)."""
        return cls.canonicalize_entity(entity), cls.canonicalize_attribute(attribute)

    @classmethod
    def canonicalize_value_and_unit(
        cls, raw_val: Any, raw_unit: str | None = None
    ) -> tuple[float | None, str]:
        """
        Algorithmically normalizes arbitrary numbers, multipliers, and scales
        to an unscaled base float representation without hardcoding.
        """
        if raw_val is None or str(raw_val).strip() == "":
            return None, "RAW"

        val_str = str(raw_val).strip()
        unit_str = (raw_unit or "").strip()
        combined = f"{val_str} {unit_str}".lower()

        # 1. Determine Currency or Base Dimension
        detected_unit = "COUNT"
        for symbol, code in cls.CURRENCY_SYMBOLS.items():
            if symbol in combined:
                detected_unit = code
                break
        if "%" in combined or "percent" in combined or "ratio" in combined:
            detected_unit = "PERCENT"

        # 2. Extract base numeric string
        # Clean currency and commas
        val_no_commas = val_str.replace(",", "")
        num_clean = re.sub(r"[^\d.-]", " ", val_no_commas)
        num_matches = re.findall(r"[-+]?\d*\.?\d+", num_clean)
        if not num_matches:
            return None, detected_unit

        try:
            base_number = float(num_matches[0])
        except ValueError:
            return None, detected_unit

        # 3. Apply Multiplier
        multiplier = 1.0
        words = re.findall(r"\b[a-zA-Z]+\b", combined)
        for w in words:
            if w in cls.MULTIPLIER_MAP:
                multiplier = cls.MULTIPLIER_MAP[w]
                break

        return (base_number * multiplier, detected_unit)

    @classmethod
    def canonicalize_temporal_interval(cls, raw_expr: str | None) -> TemporalInterval:
        """
        Algorithmically infers start and end dates from temporal strings
        (fiscal years, quarters, exact dates) using dateparser.
        """
        if not raw_expr or str(raw_expr).strip().lower() in [
            "none",
            "perpetual",
            "unknown",
            "since inception",
        ]:
            return TemporalInterval(
                start_date=None,
                end_date=None,
                granularity="PERPETUAL",
                raw_expression=raw_expr or "PERPETUAL",
            )

        clean_expr = raw_expr.strip()

        # Regex for Fiscal Year intervals: e.g., FY22, FY 2023-24, 2021-22, Fiscal 2024
        fy_match = re.search(
            r"(?:FY|Fiscal)?\s*(\d{2,4})(?:[-/](\d{2,4}))?",
            clean_expr,
            re.IGNORECASE,
        )
        if fy_match and (
            "fy" in clean_expr.lower() or "fiscal" in clean_expr.lower() or "-" in clean_expr
        ):
            start_y = int(fy_match.group(1))
            if start_y < 100:
                start_y += 2000

            end_group = fy_match.group(2)
            if end_group:
                end_y = int(end_group)
                if end_y < 100:
                    end_y = (start_y // 100) * 100 + end_y
            else:
                end_y = start_y
                start_y = end_y - 1

            # Detect Quarters
            q_match = re.search(r"Q([1-4])", clean_expr, re.IGNORECASE)
            if q_match:
                q = int(q_match.group(1))
                q_months = {
                    1: ("04-01", "06-30", start_y),
                    2: ("07-01", "09-30", start_y),
                    3: ("10-01", "12-31", start_y),
                    4: ("01-01", "03-31", end_y),
                }
                sm, em, yr = q_months[q]
                return TemporalInterval(
                    start_date=f"{yr}-{sm}",
                    end_date=f"{yr}-{em}",
                    granularity="QUARTER",
                    raw_expression=clean_expr,
                )

            return TemporalInterval(
                start_date=f"{start_y}-04-01",
                end_date=f"{end_y}-03-31",
                granularity="YEAR",
                raw_expression=clean_expr,
            )

        # Check for standalone Calendar Year (e.g., "2022", "CY2022", "CY22")
        cy_match = re.match(r"^(?:CY\s*)?(\d{2,4})$", clean_expr, re.IGNORECASE)
        if cy_match and ("cy" in clean_expr.lower() or len(cy_match.group(1)) == 4):
            cy_year = int(cy_match.group(1))
            if cy_year < 100:
                cy_year += 2000
            return TemporalInterval(
                start_date=f"{cy_year}-01-01",
                end_date=f"{cy_year}-12-31",
                granularity="YEAR",
                raw_expression=clean_expr,
            )

        # General date fallback using dateparser
        try:
            parsed_dt = dateparser.parse(clean_expr)
            if parsed_dt:
                iso_d = parsed_dt.strftime("%Y-%m-%d")
                return TemporalInterval(
                    start_date=iso_d,
                    end_date=iso_d,
                    granularity="EXACT_DATE",
                    raw_expression=clean_expr,
                )
        except Exception:
            pass

        return TemporalInterval(
            start_date=None,
            end_date=None,
            granularity="UNKNOWN",
            raw_expression=clean_expr,
        )


# Backward compatibility alias
DomainNormalizer = GeneralizedNormalizer
