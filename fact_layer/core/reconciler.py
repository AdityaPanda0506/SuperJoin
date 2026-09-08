"""
Deterministic 4-Case Cross-Document Reconciliation Engine.
Evaluates N-dimensional factual coordinates (Time, Scope, Unit, Value) to categorize
corroborations, genuine contradictions, and context-reconciled discrepancies.
"""

from typing import Optional, Union

from pydantic import BaseModel, Field

from .intervals import IntervalRelation, evaluate_temporal_intervals
from .models import GroundedFact


class ReconciliationResult(BaseModel):
    """
    Structured outcome of a cross-document fact pair reconciliation.
    """

    case_type: str = Field(
        description="One of 'CASE_1_CORROBORATION', 'CASE_2_GENUINE_CONTRADICTION', or 'CASE_3_CONTEXT_RECONCILED'.",
    )
    rationale: str = Field(
        description="Explanation detailing why the pair was categorized into this case.",
    )
    fact_a: GroundedFact = Field(
        description="First GroundedFact in the comparison pair.",
    )
    fact_b: GroundedFact = Field(
        description="Second GroundedFact in the comparison pair.",
    )
    differing_dimensions: list[str] = Field(
        default_factory=list,
        description="List of coordinate dimensions that differ between the two facts.",
    )


class FactReconciler:
    """
    Pure deterministic reconciliation decision engine.
    """

    @staticmethod
    def values_match(
        v1: Union[float, int, str, bool] | None,
        v2: Union[float, int, str, bool] | None,
        float_tolerance: float = 0.005,
    ) -> bool:
        """
        Check value equivalence with a 0.5% float tolerance for accounting roundoffs.
        """
        if v1 is None or v2 is None:
            return False

        # Numeric float / int comparison with tolerance
        if isinstance(v1, int | float) and isinstance(v2, int | float):
            v1_f, v2_f = float(v1), float(v2)
            if v1_f == v2_f == 0.0:
                return True
            diff = abs(v1_f - v2_f)
            max_val = max(abs(v1_f), abs(v2_f))
            return diff <= (max_val * float_tolerance)

        # Boolean comparison
        if isinstance(v1, bool) or isinstance(v2, bool):
            return bool(v1) == bool(v2)

        # String comparison
        return str(v1).strip().lower() == str(v2).strip().lower()

    @classmethod
    def reconcile(cls, f1: GroundedFact, f2: GroundedFact) -> ReconciliationResult | None:
        """
        Reconcile two grounded facts across documents deterministically.

        Args:
            f1: First GroundedFact
            f2: Second GroundedFact

        Returns:
            ReconciliationResult object if a comparative relationship exists, or None.
        """
        # Disregard quarantined facts from primary reconciliation
        if f1.is_quarantined or f2.is_quarantined:
            return None

        # Ignore comparisons within the exact same document page
        if (
            f1.provenance.doc_hash == f2.provenance.doc_hash
            and f1.provenance.page_number == f2.provenance.page_number
        ):
            return None

        # Evaluate N-dimensional context coordinates
        temp_rel = evaluate_temporal_intervals(f1.context_box.temporal, f2.context_box.temporal)
        same_scope = f1.context_box.scope.strip().lower() == f2.context_box.scope.strip().lower()
        same_unit = f1.context_box.canonical_unit == f2.context_box.canonical_unit
        values_are_equal = cls.values_match(f1.canonical_value, f2.canonical_value)

        # 1. CASE 1: CORROBORATION
        # Matches coordinates (or both perpetual) and values agree
        if (
            temp_rel in (IntervalRelation.EQUAL, IntervalRelation.PERPETUAL)
            and same_scope
            and same_unit
            and values_are_equal
        ):
            return ReconciliationResult(
                case_type="CASE_1_CORROBORATION",
                rationale=(
                    "Corroborated across documents: Identical values asserted under the same "
                    "temporal interval and scope despite differing phrasing."
                ),
                fact_a=f1,
                fact_b=f2,
                differing_dimensions=[],
            )

        # 2. CASE 2: GENUINE CONTRADICTION
        # Same temporal window, same scope, same unit, but incompatible values
        if (
            temp_rel in (IntervalRelation.EQUAL, IntervalRelation.PERPETUAL)
            and same_scope
            and same_unit
            and not values_are_equal
        ):
            return ReconciliationResult(
                case_type="CASE_2_GENUINE_CONTRADICTION",
                rationale=(
                    "Genuine contradiction: Mutually exclusive figures reported for the exact "
                    "same reporting period and corporate scope."
                ),
                fact_a=f1,
                fact_b=f2,
                differing_dimensions=["Value Conflict"],
            )

        # 3. CASE 3: APPARENT CONTRADICTION RECONCILED BY CONTEXT
        # Values differ, but coordinate differences explain why
        diff_axes = []
        if temp_rel not in (IntervalRelation.EQUAL, IntervalRelation.PERPETUAL):
            f1_time_expr = (
                f1.context_box.temporal.raw_expression if f1.context_box.temporal else "UNKNOWN"
            )
            f2_time_expr = (
                f2.context_box.temporal.raw_expression if f2.context_box.temporal else "UNKNOWN"
            )

            if temp_rel == IntervalRelation.DISJOINT:
                diff_axes.append(
                    f"Temporal Progression/Evolution ({f1_time_expr} vs {f2_time_expr})"
                )
            elif temp_rel in (IntervalRelation.SUBSET, IntervalRelation.SUPERSET):
                diff_axes.append("Aggregation Scope (Quarterly sub-period vs Full Fiscal Year)")
            else:
                diff_axes.append(f"Different Time Windows ({f1_time_expr} vs {f2_time_expr})")

        if not same_scope:
            diff_axes.append(
                f"Organizational Scope ({f1.context_box.scope} vs {f2.context_box.scope})"
            )

        if not same_unit:
            diff_axes.append(
                f"Unit/Measurement Scale ({f1.context_box.canonical_unit} vs {f2.context_box.canonical_unit})"
            )

        if diff_axes:
            return ReconciliationResult(
                case_type="CASE_3_CONTEXT_RECONCILED",
                rationale=(
                    f"Apparent contradiction reconciled by context: Figures differ due to {', '.join(diff_axes)}."
                ),
                fact_a=f1,
                fact_b=f2,
                differing_dimensions=diff_axes,
            )

        return None
