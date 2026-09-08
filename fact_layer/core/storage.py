"""
SQLite-Backed Incremental Inverted Fact Store with Dynamic Entity Clustering.
Provides persistent storage, inverted indexing on attribute keys,
and O(N_new * K_matched) incremental reconciliation using token Jaccard similarity.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .models import (
    ContextBoundingBox,
    GroundedFact,
    Provenance,
    TemporalInterval,
)
from .normalizer import DomainNormalizer, GeneralizedNormalizer
from .reconciler import FactReconciler, ReconciliationResult


class FactStore:
    """
    Incremental fact store backed by SQLite with dynamic entity clustering.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize FactStore.

        Args:
            db_path: Path to SQLite database file or ':memory:' for in-memory DB.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def reset_database(self):
        """Drop and recreate all database tables and indexes."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS relationships")
        cursor.execute("DROP TABLE IF EXISTS facts")
        self.conn.commit()
        self._init_db()

    def _init_db(self):
        """Create database tables and inverted indexes."""
        cursor = self.conn.cursor()

        # Table 1: facts
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                entity_canonical TEXT NOT NULL,
                attribute_canonical TEXT NOT NULL,
                entity_raw TEXT NOT NULL,
                attribute_raw TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                canonical_value_str TEXT,
                canonical_value_num REAL,
                start_date TEXT,
                end_date TEXT,
                granularity TEXT,
                temporal_raw TEXT,
                scope TEXT NOT NULL,
                canonical_unit TEXT NOT NULL,
                doc_hash TEXT NOT NULL,
                source_doc_name TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                verbatim_quote TEXT NOT NULL,
                provenance_modality TEXT DEFAULT 'TEXT',
                confidence_score REAL NOT NULL,
                is_quarantined INTEGER NOT NULL,
                quarantine_reason TEXT
            )
            """
        )

        try:
            cursor.execute("ALTER TABLE facts ADD COLUMN provenance_modality TEXT DEFAULT 'TEXT'")
        except Exception:
            pass

        # Table 2: relationships
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id_a TEXT NOT NULL,
                fact_id_b TEXT NOT NULL,
                case_type TEXT NOT NULL,
                rationale TEXT NOT NULL,
                differing_dimensions TEXT NOT NULL,
                FOREIGN KEY (fact_id_a) REFERENCES facts(fact_id),
                FOREIGN KEY (fact_id_b) REFERENCES facts(fact_id)
            )
            """
        )

        # Inverted Indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_attr ON facts (entity_canonical, attribute_canonical)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attr ON facts (attribute_canonical)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON facts (doc_hash)")

        self.conn.commit()

    def is_document_ingested(self, doc_hash: str) -> bool:
        """Check if a document has already been ingested into the store."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM facts WHERE doc_hash = ? LIMIT 1", (doc_hash,))
        return cursor.fetchone() is not None

    def add_document_facts(
        self,
        facts: list[GroundedFact],
        doc_hash: str,
        doc_name: str,
    ) -> list[ReconciliationResult]:
        """
        Incrementally ingest new document facts into the store.
        Reconciles newly extracted facts against existing store without re-indexing prior documents.
        Uses dynamic token similarity (>= 0.70) for generalized entity resolution.

        Args:
            facts: List of GroundedFact objects extracted from the document.
            doc_hash: SHA-256 hash of the document.
            doc_name: Filename of the document.

        Returns:
            List of newly discovered ReconciliationResult objects.
        """
        # Step 1: Check duplicate document re-ingestion
        if self.is_document_ingested(doc_hash):
            print(
                f"[Info] Document '{doc_name}' ({doc_hash[:12]}...) is already in store. Skipping re-indexing."
            )
            return []

        cursor = self.conn.cursor()
        new_results: list[ReconciliationResult] = []

        # Step 2: Process each fact
        for fact in facts:
            # Canonicalize entity & attribute keys and normalizations
            ent_canon, attr_canon = GeneralizedNormalizer.canonicalize_keys(
                fact.entity, fact.attribute
            )

            # Update canonical value and unit
            canon_val, canon_unit = GeneralizedNormalizer.canonicalize_value_and_unit(
                fact.raw_value, fact.context_box.canonical_unit
            )
            if fact.canonical_value is None:
                fact.canonical_value = canon_val
            fact.context_box.canonical_unit = canon_unit

            # Normalize temporal interval
            if fact.context_box.temporal and fact.context_box.temporal.raw_expression:
                fact.context_box.temporal = GeneralizedNormalizer.canonicalize_temporal_interval(
                    fact.context_box.temporal.raw_expression
                )

            # Store the fact in DB
            self._save_fact_record(cursor, fact, ent_canon, attr_canon)

            # If fact is quarantined, do not reconcile with existing store
            if fact.is_quarantined:
                continue

            # Step 3: Query candidate facts by attribute_canonical from OTHER documents
            cursor.execute(
                """
                SELECT * FROM facts
                WHERE attribute_canonical = ?
                  AND doc_hash != ?
                  AND is_quarantined = 0
                """,
                (attr_canon, doc_hash),
            )

            matched_rows = cursor.fetchall()
            for row in matched_rows:
                existing_entity_raw = row["entity_raw"]
                existing_entity_canon = row["entity_canonical"]

                # Dynamic Entity Clustering Check
                entity_sim = GeneralizedNormalizer.compute_similarity(
                    existing_entity_raw, fact.entity
                )
                same_entity_cluster = entity_sim >= 0.70 or existing_entity_canon == ent_canon

                if same_entity_cluster:
                    existing_fact = self._row_to_grounded_fact(row)
                    res = FactReconciler.reconcile(existing_fact, fact)
                    if res:
                        new_results.append(res)
                        self._save_relationship_record(cursor, res)

        self.conn.commit()
        return new_results

    def _save_fact_record(
        self,
        cursor: sqlite3.Cursor,
        fact: GroundedFact,
        ent_canon: str,
        attr_canon: str,
    ):
        """Insert or replace a fact record into the SQLite facts table."""
        canon_num = (
            float(fact.canonical_value) if isinstance(fact.canonical_value, (int, float)) else None
        )
        canon_str = str(fact.canonical_value) if fact.canonical_value is not None else None

        temp = fact.context_box.temporal
        modality = getattr(fact.provenance, "provenance_modality", "TEXT")

        cursor.execute(
            """
            INSERT OR REPLACE INTO facts (
                fact_id, entity_canonical, attribute_canonical, entity_raw, attribute_raw,
                raw_value, canonical_value_str, canonical_value_num, start_date, end_date,
                granularity, temporal_raw, scope, canonical_unit, doc_hash, source_doc_name,
                page_number, verbatim_quote, provenance_modality, confidence_score, is_quarantined, quarantine_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.fact_id,
                ent_canon,
                attr_canon,
                fact.entity,
                fact.attribute,
                fact.raw_value,
                canon_str,
                canon_num,
                temp.start_date if temp else None,
                temp.end_date if temp else None,
                temp.granularity if temp else "UNKNOWN",
                temp.raw_expression if temp else None,
                fact.context_box.scope,
                fact.context_box.canonical_unit,
                fact.provenance.doc_hash,
                fact.provenance.source_doc_name,
                fact.provenance.page_number,
                fact.provenance.verbatim_quote,
                modality,
                fact.confidence_score,
                1 if fact.is_quarantined else 0,
                fact.quarantine_reason,
            ),
        )

    def _save_relationship_record(self, cursor: sqlite3.Cursor, res: ReconciliationResult):
        """Insert a relationship record into the SQLite relationships table."""
        cursor.execute(
            """
            INSERT INTO relationships (
                fact_id_a, fact_id_b, case_type, rationale, differing_dimensions
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                res.fact_a.fact_id,
                res.fact_b.fact_id,
                res.case_type,
                res.rationale,
                json.dumps(res.differing_dimensions),
            ),
        )

    @staticmethod
    def _row_to_grounded_fact(row: sqlite3.Row) -> GroundedFact:
        """Convert a SQLite row back into a GroundedFact Pydantic object."""
        canon_val = (
            row["canonical_value_num"]
            if row["canonical_value_num"] is not None
            else row["canonical_value_str"]
        )

        temporal = TemporalInterval(
            start_date=row["start_date"],
            end_date=row["end_date"],
            granularity=row["granularity"],
            raw_expression=row["temporal_raw"],
        )

        modality = "TEXT"
        try:
            if "provenance_modality" in row.keys() and row["provenance_modality"]:
                modality = row["provenance_modality"]
        except Exception:
            pass

        return GroundedFact(
            fact_id=row["fact_id"],
            entity=row["entity_raw"],
            attribute=row["attribute_raw"],
            raw_value=row["raw_value"],
            canonical_value=canon_val,
            context_box=ContextBoundingBox(
                temporal=temporal,
                scope=row["scope"],
                canonical_unit=row["canonical_unit"],
            ),
            provenance=Provenance(
                source_doc_name=row["source_doc_name"],
                doc_hash=row["doc_hash"],
                page_number=row["page_number"],
                verbatim_quote=row["verbatim_quote"],
                provenance_modality=modality,
            ),
            confidence_score=row["confidence_score"],
            is_quarantined=bool(row["is_quarantined"]),
            quarantine_reason=row["quarantine_reason"],
        )

    def get_all_facts(self) -> list[GroundedFact]:
        """Retrieve all stored facts."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM facts")
        return [self._row_to_grounded_fact(r) for r in cursor.fetchall()]

    def get_quarantined_facts(self) -> list[GroundedFact]:
        """Retrieve all quarantined facts."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM facts WHERE is_quarantined = 1")
        return [self._row_to_grounded_fact(r) for r in cursor.fetchall()]

    def get_all_relationships(self) -> list[tuple[str, str, str, str]]:
        """Retrieve all stored relationships."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT fact_id_a, fact_id_b, case_type, rationale FROM relationships")
        return [
            (r["fact_id_a"], r["fact_id_b"], r["case_type"], r["rationale"])
            for r in cursor.fetchall()
        ]


# Backward compatibility & alias
IncrementalFactStore = FactStore
