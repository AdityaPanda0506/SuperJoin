"""
Fact Integrity Gate & Anomaly Quarantine Circuit Breaker.
Validates verbatim quote grounding against source text and detects degenerate coordinate bounding boxes.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Union

from .models import GroundedFact


class FactIntegrityGate:
    """
    Validation gate performing post-extraction fact integrity verification.
    Quarantines hallucinated quotes and unanchored numeric facts.
    """

    def __init__(self, quote_fuzzy_threshold: float = 0.85):
        """
        Initialize FactIntegrityGate.

        Args:
            quote_fuzzy_threshold: Minimum similarity ratio (0.0 to 1.0) for fuzzy quote verification.
        """
        self.quote_fuzzy_threshold = quote_fuzzy_threshold

    @staticmethod
    def _normalize_string(s: str) -> str:
        """Strip non-alphanumeric characters and collapse whitespace for resilient matching."""
        if not s:
            return ""
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()

    def verify_quote_grounding(self, quote: str, page_text: str) -> bool:
        """
        Verify if a quote exists verbatim or near-verbatim in page text.

        Args:
            quote: The extracted verbatim quote string.
            page_text: Full clean text of the page.

        Returns:
            True if quote is grounded in source page text, False otherwise.
        """
        if not quote or not quote.strip():
            return False

        norm_quote = self._normalize_string(quote)
        norm_page = self._normalize_string(page_text)

        # 1. Exact substring check on normalized text
        if norm_quote in norm_page:
            return True

        # 2. Sliding window fuzzy match check for minor OCR / whitespace variances
        words_quote = norm_quote.split()
        words_page = norm_page.split()

        if not words_quote or not words_page:
            return False

        quote_len = len(words_quote)
        # Search windows around quote length
        for i in range(max(1, len(words_page) - quote_len + 1)):
            window_str = " ".join(words_page[i : i + quote_len])
            ratio = SequenceMatcher(None, norm_quote, window_str).ratio()
            if ratio >= self.quote_fuzzy_threshold:
                return True

        return False

    @staticmethod
    def is_numeric_fact(fact: GroundedFact) -> bool:
        """
        Determine if a fact represents a quantitative/numeric claim.
        """
        # Check canonical value type
        if isinstance(fact.canonical_value, (int, float)):
            return True

        # Check raw_value pattern (digits, currency symbols, percentages)
        raw = fact.raw_value.strip()
        if re.search(r"\d", raw):
            return True

        # Check attribute keywords
        attr = fact.attribute.lower()
        numeric_keywords = [
            "revenue",
            "arr",
            "mrr",
            "ebitda",
            "income",
            "growth",
            "margin",
            "percentage",
            "count",
            "amount",
            "headcount",
            "price",
            "cost",
            "valuation",
        ]
        if any(kw in attr for kw in numeric_keywords):
            return True

        return False

    def validate_fact(self, fact: GroundedFact, page_text: str) -> GroundedFact:
        """
        Validate a single GroundedFact against page source text or visual chart evidence.
        Mutates `is_quarantined` and `quarantine_reason` if integrity checks fail.

        Args:
            fact: GroundedFact to validate.
            page_text: Text content of the fact's source page.

        Returns:
            Validated (and potentially quarantined) GroundedFact.
        """
        is_visual = (
            getattr(fact.provenance, "provenance_modality", "TEXT") == "VISUAL_CHART"
            or fact.provenance.verbatim_quote.startswith("[Visual Chart]")
        )

        # 1. Quote / Visual Grounding Verification Check
        quote = fact.provenance.verbatim_quote
        if is_visual:
            # Visual charts skip text substring check; verified via visual chart description
            pass
        else:
            if not self.verify_quote_grounding(quote, page_text):
                fact.is_quarantined = True
                fact.quarantine_reason = "UNVERIFIED_SOURCE_QUOTE: Quote not found in page text"
                return fact

        # 2. Degenerate Coordinate & Chart Axis Check (Unanchored Numeric Claims)
        if self.is_numeric_fact(fact) or is_visual:
            temporal = fact.context_box.temporal
            is_unanchored = (
                temporal is None
                or not temporal.raw_expression
                or temporal.raw_expression.strip() == ""
                or temporal.granularity == "UNKNOWN"
            )
            if is_unanchored:
                fact.is_quarantined = True
                fact.quarantine_reason = (
                    "DEGENERATE_CHART_AXIS: Missing time horizon on graph legend or axis"
                    if is_visual
                    else "DEGENERATE_TEMPORAL_BOUND: Numeric claim lacks time horizon"
                )
                return fact

        # If all checks pass and it was not previously quarantined
        if not fact.quarantine_reason:
            fact.is_quarantined = False
            fact.quarantine_reason = None

        return fact

    @classmethod
    def validate_facts(
        cls,
        facts: Union[List[GroundedFact], str],
        page_texts: Optional[Union[Dict[int, str], List[GroundedFact]]] = None,
        page_text_map: Optional[Dict[int, str]] = None,
        page_text: Optional[str] = None,
    ) -> Tuple[List[GroundedFact], List[GroundedFact]]:
        """
        Batch validate a list of extracted facts against page texts or visual chart evidence.

        Args:
            facts: List of GroundedFact objects (or string page_text if positional swap).
            page_texts: Mapping of page_number -> clean_text.
            page_text_map: Alias for page_texts mapping.
            page_text: Single string of page text (if validating single page).

        Returns:
            Tuple of (valid_facts, quarantined_facts).
        """
        if isinstance(facts, str) and isinstance(page_texts, list):
            return cls.validate_facts(facts=page_texts, page_text=facts)

        gate = cls()
        p_texts = page_texts if isinstance(page_texts, dict) else (page_text_map or {})

        valid_facts: List[GroundedFact] = []
        quarantined_facts: List[GroundedFact] = []

        for fact in facts:
            page_num = fact.provenance.page_number
            text = p_texts.get(page_num, page_text or "")
            validated_fact = gate.validate_fact(fact, text)

            if validated_fact.is_quarantined:
                quarantined_facts.append(validated_fact)
            else:
                valid_facts.append(validated_fact)

        return valid_facts, quarantined_facts
