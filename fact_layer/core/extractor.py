"""
Structured Extraction Pipeline using Google GenAI SDK (google-genai).
Extracts grounded, context-bounded hyper-tuple facts with token optimization, Indian financial/operational metric support, and Pydantic schema validation.
"""

import json
import os
import re
from typing import Optional

from google import genai
from google.genai import types

from .models import (
    ContextBoundingBox,
    GroundedFact,
    PageExtractionBatch,
    Provenance,
    TemporalInterval,
)
from .parser import PageChunk

SYSTEM_PROMPT = """
You are an expert financial and macroeconomic data extraction engine.
Extract grounded atomic facts as structured JSON objects.

STRICT CONSTRAINTS ON 'entity':
1. The 'entity' field MUST be the formal name of the company, central bank, government body, or sovereign state (e.g., 'Delhivery Limited', 'Reserve Bank of India', 'Ministry of Finance', 'Government of India').
2. NEVER set 'entity' to the title of the document, report name, publication year, table header, or citation string (e.g., DO NOT use '01 India Economic Survey 2024 25', 'PIB press release', or 'January 2025 round').
3. If the sentence reports a national macroeconomic figure without an explicit body named, set 'entity' to 'Government of India' or 'Indian Economy'.

EXTRACTION DIRECTIVES:
1. ONLY extract actionable, falsifiable business, financial, or operational claims across any domain:
   - Financials: Revenues, Income, Loss, EBITDA, Market Size, Proceeds (in ₹, INR, USD, Crores, Millions, Billions, Percentages).
   - Operational Metrics: PIN Code coverage, Express Parcel volumes, Freight tonnage, Workforce strength, Logistics area (sq ft), Fleet size, Active customers.
   - Corporate Governance: Executive appointments, Board roles, Founder shareholdings, Registered offices.
2. FORBID extracting trivial boilerplate, page headers/footers, page numbers, legal disclaimers, or table headers without values.
3. GROUNDING IS MANDATORY: Every fact MUST include a `verbatim_quote` containing an EXACT, continuous snippet from the source text supporting the fact.
4. CONTEXT BOUNDING BOX:
   - `temporal`: Qualify start_date/end_date (ISO-8601 YYYY-MM-DD or YYYY if present), granularity ('EXACT_DATE', 'MONTH', 'QUARTER', 'YEAR', 'PERPETUAL', 'UNKNOWN'), and raw_expression (e.g., 'FY24', 'Fiscal 2022', 'as of March 31, 2024', 'since inception').
   - `scope`: Qualification scope (e.g. 'GLOBAL', 'India', 'Consolidated', 'Express Parcel', 'PTL').
   - `canonical_unit`: Standardized unit/scale (e.g. 'INR', 'USD', 'INR_CRORES', 'INR_MILLIONS', 'PERCENT', 'COUNT', 'PIN_CODES', 'PARCELS', 'TONNES', 'SQ_FT').
5. CANONICAL VALUE NORMALIZATION:
   - For numeric claims (e.g., "₹81,415Mn", "18,793 Pin codes", "2.8Bn parcels"), normalize `canonical_value` to float/int (e.g., 81415000000.0, 18793, 2800000000).
6. OUTPUT SCHEMA: Return ONLY structured JSON adhering strictly to the PageExtractionBatch schema.
"""


class FactExtractor:
    """
    LLM extraction engine with token budget management, Indian metric support, and Pydantic schema enforcement.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """
        Initialize FactExtractor.

        Args:
            api_key: Optional Google GenAI API key. If not provided, falls back to GEMINI_API_KEY environment variable.
            model_name: Model identifier (default: 'gemini-2.5-flash').
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[Warning] Failed to initialize GenAI client: {e}. Fallback mode active.")
        else:
            print(
                "[Info] GEMINI_API_KEY not found. FactExtractor operating in fallback/simulation mode."
            )

    @classmethod
    def extract(
        cls,
        clean_text: str,
        source_doc_name: str,
        page_number: int,
        doc_hash: str,
    ) -> list[GroundedFact]:
        """
        Classmethod helper to extract grounded facts from a single page's text.
        """
        extractor = cls()
        chunk = PageChunk(
            doc_name=source_doc_name,
            doc_hash=doc_hash,
            page_number=page_number,
            clean_text=clean_text,
            has_tables=False,
            token_estimate=max(1, int(len(clean_text.split()) * 1.3)),
        )
        batch = extractor.extract_facts_from_chunk(chunk)
        return batch.facts

    def extract_facts_from_chunk(self, chunk: PageChunk) -> PageExtractionBatch:
        """
        Extract grounded facts from a single page chunk using Gemini structured outputs.
        Opportunistically routes pages with embedded graphics/charts through Multimodal Vision.

        Args:
            chunk: PageChunk containing page text, doc name, hash, page number, and optional image_bytes.

        Returns:
            PageExtractionBatch containing extracted GroundedFact objects.
        """
        if not chunk.clean_text and not chunk.has_images:
            return PageExtractionBatch(facts=[])

        # If LLM client is available, run GenAI text/vision extraction
        if self.client:
            try:
                contents = []
                # Opportunistic Track B: Attach rendered 150 DPI page image if graphics/charts detected
                if chunk.has_images and chunk.image_bytes:
                    contents.append(
                        types.Part.from_bytes(
                            data=chunk.image_bytes,
                            mime_type="image/jpeg",
                        )
                    )

                prompt_content = (
                    f"SOURCE DOCUMENT: {chunk.doc_name} (Page {chunk.page_number})\n"
                    f"MODALITY: {'VISUAL_CHART & TEXT' if chunk.has_images else 'TEXT_ONLY'}\n"
                    f"PAGE TEXT:\n{chunk.clean_text or '[Visual Page]'}\n\n"
                    f"INSTRUCTION: Extract grounded numeric and business facts. For visual charts/graphs, "
                    f"read X-axis time horizons and Y-axis scales, and prefix `verbatim_quote` with '[Visual Chart]: '."
                )
                contents.append(prompt_content)

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PageExtractionBatch,
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )

                if response.text:
                    parsed_batch = PageExtractionBatch.model_validate_json(response.text)
                    # Enforce correct provenance fields from chunk
                    for fact in parsed_batch.facts:
                        fact.provenance.source_doc_name = chunk.doc_name
                        fact.provenance.doc_hash = chunk.doc_hash
                        fact.provenance.page_number = chunk.page_number
                        if (
                            "[visual chart]" in fact.provenance.verbatim_quote.lower()
                            or chunk.has_images
                        ):
                            fact.provenance.provenance_modality = "VISUAL_CHART"
                        else:
                            fact.provenance.provenance_modality = getattr(
                                chunk, "provenance_modality", "TEXT"
                            )
                    return parsed_batch

            except Exception as e:
                print(f"[Error] GenAI extraction failed for page {chunk.page_number}: {e}")

        # Fallback heuristic / deterministic parsing when API key is missing or call fails
        return self._heuristic_fallback_extraction(chunk)

    def extract_facts_from_chunks(self, chunks: list[PageChunk]) -> list[GroundedFact]:
        """
        Extract facts across all page chunks in a document.

        Args:
            chunks: List of PageChunk objects.

        Returns:
            Flattened list of all extracted GroundedFact objects.
        """
        all_facts: list[GroundedFact] = []
        for chunk in chunks:
            batch = self.extract_facts_from_chunk(chunk)
            all_facts.extend(batch.facts)
        return all_facts

    def _heuristic_fallback_extraction(self, chunk: PageChunk) -> PageExtractionBatch:
        """
        Deterministic heuristic extractor for financial, operational, and macroeconomic documents.
        Recognizes Crores, Lakhs, Millions, Billions, PIN codes, Parcel Volume, Percentages, and Roles across multiline layouts.
        """
        facts: list[GroundedFact] = []
        text = chunk.clean_text

        # Extract dynamic entity candidate from text or document name stem
        clean_stem = (
            re.sub(r"^\d+\s*[\-_]?", "", chunk.doc_name)
            .replace("-excerpt.pdf", "")
            .replace(".pdf", "")
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )
        if any(
            term in clean_stem.lower()
            for term in ["economic survey", "pib", "press release", "bulletin", "budget"]
        ):
            doc_entity = "Government of India"
        else:
            doc_entity = clean_stem.title()
        org_match = re.search(
            r"\b([A-Z][A-Za-z0-9\&\.\s]{2,35}\s+(?:Limited|Ltd|Corporation|Corp|Bank|Ministry|Department|Government|Inc|LLP|Authority|Council|Institute|Company))\b",
            text,
        )
        dynamic_entity = org_match.group(1).strip() if org_match else doc_entity

        # Heuristic 1: Stat Box / KPI items (multiline friendly)
        kpi_patterns = [
            # Parcel shipments
            (
                r"([>~]?\d+(?:\.\d+)?\s*(?:Bn|billion|Mn|million)?\s*(?:\(\d+\))?\s*\n?\s*Express parcel shipments[A-Za-z\s\n]*)",
                dynamic_entity,
                "Express Parcel Volume",
                "PARCELS",
            ),
            # PIN codes
            (
                r"(\d{1,3}(?:,\d{3})*|\d+)\s*(?:\(\d+\))?\s*\n?\s*(?:Pin codes covered|PIN codes covered|pin codes)",
                dynamic_entity,
                "PIN Code Coverage",
                "PIN_CODES",
            ),
            # Freight tonnage
            (
                r"([>~]?\d+(?:\.\d+)?\s*(?:Mn|million|K)?\s*tonnes\s*(?:\(\d+(?:,\d+)?\))?\s*\n?\s*Part-truckload freight[A-Za-z\s\n]*)",
                dynamic_entity,
                "PTL Freight Delivered",
                "TONNES",
            ),
            # Workforce
            (
                r"(\d{1,3}(?:,\d{3})*|\d+)\s*(?:\(\d+(?:,\d+)?\))?\s*\n?\s*(?:Workforce strength|employees|workforce)",
                dynamic_entity,
                "Workforce Strength",
                "COUNT",
            ),
            # Logistics area
            (
                r"(\d+(?:\.\d+)?\s*(?:Mn|million)?\s*Sq ft\s*(?:\(\d+\))?\s*\n?\s*Logistics area[A-Za-z\s\n]*)",
                dynamic_entity,
                "Logistics Area Under Management",
                "SQ_FT",
            ),
            # Active customers
            (
                r"([>~]?\d{1,3}(?:,\d{3})*|\d+)\s*(?:\(\d+(?:,\d+)?\))?\s*\n?\s*(?:Active customers)",
                dynamic_entity,
                "Active Customers",
                "COUNT",
            ),
            # Annual report summary figures
            (
                r"(\d+(?:\.\d+)?\s*(?:Mn|million|Bn|billion|K)?\s*Express parcels shipped)",
                dynamic_entity,
                "Express Parcel Volume",
                "PARCELS",
            ),
            (
                r"((?:₹|Rs\.?|INR|\$)\s*\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})*)\s*(?:Mn|million|Cr|crore|crores|Bn|billion)?\s*Revenue from services",
                dynamic_entity,
                "Revenue from Services",
                "INR",
            ),
            (
                r"((?:₹|Rs\.?|INR|\$)\s*\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})*)\s*(?:Mn|million|Cr|crore|crores|Bn|billion)?\s*EBITDA\b",
                dynamic_entity,
                "EBITDA",
                "INR",
            ),
        ]

        for pattern, entity_default, attr_default, unit_default in kpi_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                quote = m.group(0).strip()
                # Find number inside quote
                num_m = re.search(
                    r"([>~]?\d+(?:\.\d+)?\s*(?:Bn|Mn|million|billion|K)?|\d{1,3}(?:,\d{3})*)", quote
                )
                raw_val = num_m.group(0).strip() if num_m else quote
                canon_val = self._parse_canonical_number(raw_val)
                temporal_info = self._extract_temporal_hint(quote, text)

                facts.append(
                    GroundedFact(
                        entity=entity_default,
                        attribute=attr_default,
                        raw_value=raw_val,
                        canonical_value=canon_val,
                        context_box=ContextBoundingBox(
                            temporal=temporal_info,
                            scope="India",
                            canonical_unit=unit_default,
                        ),
                        provenance=Provenance(
                            source_doc_name=chunk.doc_name,
                            doc_hash=chunk.doc_hash,
                            page_number=chunk.page_number,
                            verbatim_quote=quote,
                        ),
                        confidence_score=0.92,
                    )
                )

        # Heuristic 2: Financial Metrics (Revenue, EBITDA, Net Proceeds, Income, Spend, Loss)
        fin_pattern = re.compile(
            r"([A-Z][A-Za-z0-9\s,\(\)]{3,40}?)\s*(?:of|was|reached|grew to|aggregating to|is|stood at)\s*"
            r"((?:₹|Rs\.?|INR|\$)\s*\d+(?:\.\d+)?\s*(?:Cr|crore|crores|million|billion|Mn|Bn|Lakh|Lakhs|M|B)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|Cr|crores|billion))",
            re.IGNORECASE,
        )

        for m in fin_pattern.finditer(text):
            attr_cand = m.group(1).strip()
            val_str = m.group(2).strip()
            quote = m.group(0).strip()

            # Ignore generic non-financial matches
            if any(skip in attr_cand.lower() for skip in ["table", "page", "section", "dated"]):
                continue

            canon_val = self._parse_canonical_number(val_str)
            temporal_info = self._extract_temporal_hint(quote, text)
            unit = (
                "INR"
                if ("₹" in val_str or "Rs" in val_str or "INR" in val_str or "Cr" in val_str)
                else "USD"
            )

            facts.append(
                GroundedFact(
                    entity=dynamic_entity,
                    attribute=attr_cand if len(attr_cand) < 35 else "Financial Metric",
                    raw_value=val_str,
                    canonical_value=canon_val,
                    context_box=ContextBoundingBox(
                        temporal=temporal_info,
                        scope="Consolidated",
                        canonical_unit=unit,
                    ),
                    provenance=Provenance(
                        source_doc_name=chunk.doc_name,
                        doc_hash=chunk.doc_hash,
                        page_number=chunk.page_number,
                        verbatim_quote=quote,
                    ),
                    confidence_score=0.90,
                )
            )

        # Heuristic 3: Explicit percentages (e.g., "EBITDA margin: 1.6%", "Operating Margin: 24%")
        pct_matches = re.finditer(
            r"([A-Z][A-Za-z0-9\s]{2,30}?)\s*(?:of|margin|growth|cagr|share)?\s*[:\=]?\s*(\d+(?:\.\d+)?\s*%)",
            text,
            re.IGNORECASE,
        )
        for m in pct_matches:
            attr_cand = m.group(1).strip()
            val_str = m.group(2).strip()
            quote = m.group(0).strip()

            canon_val = self._parse_canonical_number(val_str)
            temporal_info = self._extract_temporal_hint(quote, text)

            facts.append(
                GroundedFact(
                    entity=dynamic_entity,
                    attribute=attr_cand,
                    raw_value=val_str,
                    canonical_value=canon_val,
                    context_box=ContextBoundingBox(
                        temporal=temporal_info,
                        scope="GLOBAL",
                        canonical_unit="PERCENT",
                    ),
                    provenance=Provenance(
                        source_doc_name=chunk.doc_name,
                        doc_hash=chunk.doc_hash,
                        page_number=chunk.page_number,
                        verbatim_quote=quote,
                    ),
                    confidence_score=0.88,
                )
            )

        # Heuristic 4: Visual Chart & Infographic Extraction (Track B Fallback)
        chart_matches = re.finditer(
            r"(?:Chart|Graph|Figure|Plot|Infographic)\s*[\:\-]?\s*([A-Za-z0-9\s,\(\)]{3,40}?)\s*(?:stood at|reached|was|grew to)?\s*"
            r"((?:₹|Rs\.?|INR|\$)\s*\d+(?:\.\d+)?\s*(?:Cr|crore|crores|million|billion|Mn|Bn|Lakh|Lakhs|M|B)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|Cr|crores|billion)?)",
            text,
            re.IGNORECASE,
        )
        for m in chart_matches:
            attr_cand = m.group(1).strip()
            val_str = m.group(2).strip() if m.group(2) else ""
            if not val_str:
                continue
            quote = f"[Visual Chart]: {m.group(0).strip()} (X-Axis: {chunk.doc_name} Pg {chunk.page_number})"
            canon_val = self._parse_canonical_number(val_str)
            temporal_info = self._extract_temporal_hint(quote, text)
            unit = (
                "INR"
                if ("₹" in val_str or "Rs" in val_str or "INR" in val_str or "Cr" in val_str)
                else "USD"
            )

            facts.append(
                GroundedFact(
                    entity=dynamic_entity,
                    attribute=attr_cand if len(attr_cand) < 35 else "Visual Chart Metric",
                    raw_value=val_str,
                    canonical_value=canon_val,
                    context_box=ContextBoundingBox(
                        temporal=temporal_info,
                        scope="Consolidated",
                        canonical_unit=unit,
                    ),
                    provenance=Provenance(
                        source_doc_name=chunk.doc_name,
                        doc_hash=chunk.doc_hash,
                        page_number=chunk.page_number,
                        verbatim_quote=quote,
                        provenance_modality="VISUAL_CHART",
                    ),
                    confidence_score=0.92,
                )
            )

        # Deduplicate facts by verbatim_quote to keep extraction clean
        unique_facts = []
        seen_quotes = set()
        for f in facts:
            q_norm = f.provenance.verbatim_quote.strip().lower()
            if q_norm not in seen_quotes:
                seen_quotes.add(q_norm)
                unique_facts.append(f)

        return PageExtractionBatch(facts=unique_facts)

    @staticmethod
    def _parse_canonical_number(val_str: str) -> float | None:
        """
        Convert string numbers like ₹81,415Mn, ₹40,000 million, 18,793, 2.8Bn, 12.5% into canonical floats.
        Supports Crores (Cr), Lakhs (L), Millions (Mn), Billions (Bn), Thousands (K).
        """
        if not val_str:
            return None

        # Clean symbols
        clean = (
            val_str.replace("₹", "")
            .replace("$", "")
            .replace("Rs.", "")
            .replace("Rs", "")
            .replace("INR", "")
            .replace(",", "")
            .strip()
        )
        multiplier = 1.0

        if re.search(r"\bcrores?\b|\bCr\b", clean, re.IGNORECASE):
            multiplier = 10_000_000.0  # 1 Cr = 10 Million = 10,000,000
            clean = re.sub(r"\bcrores?\b|\bCr\b", "", clean, flags=re.IGNORECASE).strip()
        elif re.search(r"\blakhs?\b|\bL\b", clean, re.IGNORECASE):
            multiplier = 100_000.0  # 1 Lakh = 100,000
            clean = re.sub(r"\blakhs?\b|\bL\b", "", clean, flags=re.IGNORECASE).strip()
        elif re.search(r"\bbillion\b|\bBn\b|\bB\b", clean, re.IGNORECASE):
            multiplier = 1_000_000_000.0
            clean = re.sub(r"\bbillion\b|\bBn\b|\bB\b", "", clean, flags=re.IGNORECASE).strip()
        elif re.search(r"\bmillion\b|\bMn\b|\bM\b", clean, re.IGNORECASE):
            multiplier = 1_000_000.0
            clean = re.sub(r"\bmillion\b|\bMn\b|\bM\b", "", clean, flags=re.IGNORECASE).strip()
        elif re.search(r"\bK\b|\bthousand\b", clean, re.IGNORECASE):
            multiplier = 1_000.0
            clean = re.sub(r"\bK\b|\bthousand\b", "", clean, flags=re.IGNORECASE).strip()
        elif clean.endswith("%"):
            clean = clean.replace("%", "").strip()

        # Remove leading/trailing symbols like > or ~
        clean = re.sub(r"^[>~<]+", "", clean).strip()

        try:
            return float(clean) * multiplier
        except ValueError:
            return None

    @staticmethod
    def _extract_temporal_hint(quote: str, page_text: str) -> TemporalInterval | None:
        """
        Extract temporal horizons including Indian fiscal years (FY24, FY22, Fiscal 2020),
        dates (March 31, 2024), quarters (Q4 FY24), and perpetual bounds (since inception).
        """
        # 1. Perpetual / Inception check
        if re.search(r"\bsince inception\b", quote, re.IGNORECASE) or re.search(
            r"\bsince inception\b", page_text, re.IGNORECASE
        ):
            return TemporalInterval(
                raw_expression="since inception",
                granularity="PERPETUAL",
            )

        # 2. As of March 31, 2024 or dated May 14, 2022
        as_of_match = re.search(
            r"(?:as of|dated|ended)\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
            quote,
            re.IGNORECASE,
        ) or re.search(
            r"(?:as of|dated|ended)\s+([A-Z][a-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
            page_text,
            re.IGNORECASE,
        )

        if as_of_match:
            raw = as_of_match.group(0).strip()
            return TemporalInterval(
                raw_expression=raw,
                granularity="EXACT_DATE",
            )

        # 3. Fiscal Year (FY24, FY2024, Fiscal 2022, Fiscal 2026, FY22)
        fy_match = re.search(
            r"\b(FY\s*20\d\d|FY\s*\d\d|Fiscal\s*20\d\d|Fiscal\s*\d\d)\b", quote, re.IGNORECASE
        )
        if fy_match:
            raw = fy_match.group(1).strip()
            yr_digits = re.search(r"\d+", raw).group(0)
            full_year = int(yr_digits) if len(yr_digits) == 4 else 2000 + int(yr_digits)
            return TemporalInterval(
                raw_expression=raw,
                granularity="YEAR",
                start_date=f"{full_year - 1}-04-01",
                end_date=f"{full_year}-03-31",
            )

        # 4. Quarter match (e.g. Q4 FY24, Q3 2023)
        q_match = re.search(r"\b(Q[1-4]\s*(?:FY\s*\d\d|20\d\d)?)\b", quote, re.IGNORECASE)
        if q_match:
            return TemporalInterval(
                raw_expression=q_match.group(1).strip(),
                granularity="QUARTER",
            )

        # 5. Generic Year match in quote (e.g. "in 2023", "2020")
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", quote)
        if year_match:
            yr = year_match.group(1)
            return TemporalInterval(
                raw_expression=yr,
                granularity="YEAR",
                start_date=f"{yr}-01-01",
                end_date=f"{yr}-12-31",
            )

        # If quote itself lacks temporal hint, check page header/context
        para_fy = re.search(
            r"\b(FY\s*20\d\d|FY\s*\d\d|Fiscal\s*20\d\d)\b", page_text, re.IGNORECASE
        )
        if para_fy:
            raw = para_fy.group(1).strip()
            yr_digits = re.search(r"\d+", raw).group(0)
            full_year = int(yr_digits) if len(yr_digits) == 4 else 2000 + int(yr_digits)
            return TemporalInterval(
                raw_expression=f"{raw} (page context)",
                granularity="YEAR",
                start_date=f"{full_year - 1}-04-01",
                end_date=f"{full_year}-03-31",
            )

        return None


# Module alias
Extractor = FactExtractor
