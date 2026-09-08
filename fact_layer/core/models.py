"""
Pydantic v2 Data Contracts for Fact Knowledge Layer.
Models facts as N-dimensional context-bounded hyper-tuples with strict provenance.
"""

import hashlib
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

GranularityType = Literal["EXACT_DATE", "MONTH", "QUARTER", "YEAR", "PERPETUAL", "UNKNOWN"]


class TemporalInterval(BaseModel):
    """
    Temporal context window bounding a fact.
    """

    start_date: str | None = Field(
        default=None,
        description="ISO-8601 YYYY-MM-DD or year YYYY, if applicable.",
    )
    end_date: str | None = Field(
        default=None,
        description="ISO-8601 YYYY-MM-DD or year YYYY, if applicable.",
    )
    granularity: GranularityType = Field(
        default="UNKNOWN",
        description="Precision level of the temporal expression.",
    )
    raw_expression: str | None = Field(
        default=None,
        description="Verbatim temporal wording from source (e.g., 'Q3 2023', 'FY22').",
    )


class ContextBoundingBox(BaseModel):
    """
    N-dimensional bounding box qualifying the factual scope, unit, and time horizon.
    """

    temporal: TemporalInterval | None = Field(
        default=None,
        description="Temporal interval associated with the fact.",
    )
    scope: str = Field(
        default="GLOBAL",
        description="Geographic, organizational, or operational scope (e.g. 'North America', 'Consolidated').",
    )
    canonical_unit: str = Field(
        default="RAW",
        description="Standardized unit or metric scale (e.g. 'USD', 'EUR', 'PERCENT', 'COUNT', 'TEXT_STATUS').",
    )


class Provenance(BaseModel):
    """
    Strict lineage and ground-truth link back to source PDF document page.
    """

    source_doc_name: str = Field(
        description="Filename or path of source PDF.",
    )
    doc_hash: str = Field(
        description="SHA-256 checksum of the source PDF document.",
    )
    page_number: int = Field(
        ge=1,
        description="1-indexed page number where the fact occurs.",
    )
    verbatim_quote: str = Field(
        description="Continuous verbatim excerpt directly from source text ground truth.",
    )
    provenance_modality: str = Field(
        default="TEXT",
        description="Extraction modality: 'TEXT', 'VISUAL_CHART', or 'HYBRID'.",
    )


class GroundedFact(BaseModel):
    """
    Factual hyper-tuple representing an actionable business or semantic claim.
    """

    fact_id: str = Field(
        default="",
        description="Deterministic SHA-256 hash derived from entity, attribute, doc_hash, page, and raw_value.",
    )
    entity: str = Field(
        description="Canonical subject of the claim (e.g. 'Superjoin Technologies', 'Board of Directors').",
    )
    attribute: str = Field(
        description="Property or metric name (e.g. 'Annual Recurring Revenue', 'Managing Director').",
    )
    raw_value: str = Field(
        description="Verbatim string representation of value (e.g. '$12.5M', 'Resigned').",
    )
    canonical_value: Union[float, int, str, bool] | None = Field(
        default=None,
        description="Normalized value for deterministic comparison (e.g. 12500000.0).",
    )
    context_box: ContextBoundingBox = Field(
        default_factory=ContextBoundingBox,
        description="N-dimensional contextual qualifier (time, scope, unit).",
    )
    provenance: Provenance = Field(
        description="Verbatim quote and document location linkage.",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score assigned during extraction (0.0 to 1.0).",
    )
    is_quarantined: bool = Field(
        default=False,
        description="Circuit breaker flag set when integrity checks fail.",
    )
    quarantine_reason: str | None = Field(
        default=None,
        description="Explanation of integrity failure if quarantined.",
    )

    @model_validator(mode="after")
    def compute_fact_id_if_missing(self) -> "GroundedFact":
        """Compute a deterministic hash ID if fact_id was not populated."""
        if not self.fact_id:
            raw_key = (
                f"{self.entity.strip().lower()}|"
                f"{self.attribute.strip().lower()}|"
                f"{self.provenance.doc_hash}|"
                f"{self.provenance.page_number}|"
                f"{self.raw_value.strip()}"
            )
            self.fact_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        return self


class PageExtractionBatch(BaseModel):
    """
    Structured extraction batch payload returned per page.
    """

    facts: list[GroundedFact] = Field(
        default_factory=list,
        description="List of extracted grounded facts for a document chunk.",
    )
