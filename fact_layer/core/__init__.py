"""
Fact Knowledge Layer - Core Module
Provides data models, PDF parser, LLM fact extractor, integrity validator,
generalized normalizer, interval algebra evaluator, SQLite fact store, and reconciler.
"""

from .models import (
    TemporalInterval,
    ContextBoundingBox,
    Provenance,
    GroundedFact,
    PageExtractionBatch,
)
from .parser import DocumentParser, PageChunk
from .extractor import FactExtractor, Extractor
from .validator import FactIntegrityGate
from .normalizer import GeneralizedNormalizer, DomainNormalizer
from .intervals import evaluate_temporal_intervals, IntervalRelation
from .storage import FactStore, IncrementalFactStore
from .reconciler import FactReconciler, ReconciliationResult

__all__ = [
    "TemporalInterval",
    "ContextBoundingBox",
    "Provenance",
    "GroundedFact",
    "PageExtractionBatch",
    "DocumentParser",
    "PageChunk",
    "FactExtractor",
    "Extractor",
    "FactIntegrityGate",
    "GeneralizedNormalizer",
    "DomainNormalizer",
    "evaluate_temporal_intervals",
    "IntervalRelation",
    "FactStore",
    "IncrementalFactStore",
    "FactReconciler",
    "ReconciliationResult",
]
