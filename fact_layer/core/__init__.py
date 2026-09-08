"""
Fact Knowledge Layer - Core Module
Provides data models, PDF parser, LLM fact extractor, integrity validator,
generalized normalizer, interval algebra evaluator, SQLite fact store, and reconciler.
"""

from .extractor import Extractor, FactExtractor
from .intervals import IntervalRelation, evaluate_temporal_intervals
from .models import (
    ContextBoundingBox,
    GroundedFact,
    PageExtractionBatch,
    Provenance,
    TemporalInterval,
)
from .normalizer import DomainNormalizer, GeneralizedNormalizer
from .parser import DocumentParser, PageChunk
from .reconciler import FactReconciler, ReconciliationResult
from .storage import FactStore, IncrementalFactStore
from .validator import FactIntegrityGate

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
