"""
Automated NLI Reconciliation & Integrity Gate Empirical Metric Evaluator.
Computes Precision, Recall, F1, and Macro-F1 across Cases 1-4 using scikit-learn.
"""

import sys
from pathlib import Path

# Ensure project root & fact_layer are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FACT_LAYER_DIR = PROJECT_ROOT / "fact_layer"
if str(FACT_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(FACT_LAYER_DIR))

from sklearn.metrics import classification_report, precision_recall_fscore_support

try:
    from fact_layer.core import (
        GroundedFact,
        ContextBoundingBox,
        TemporalInterval,
        Provenance,
        FactReconciler,
        FactIntegrityGate,
    )
except ImportError:
    from core import (
        GroundedFact,
        ContextBoundingBox,
        TemporalInterval,
        Provenance,
        FactReconciler,
        FactIntegrityGate,
    )


def run_evaluation():
    print("=" * 75)
    print("        EMPIRICAL EVALUATION SUITE: NLI RECONCILIATION & INTEGRITY GATE")
    print("=" * 75)

    # 1. Benchmark Pairs for Cases 1, 2, 3
    test_pairs = [
        # Case 1: Corroboration (Incorporation date across filings)
        (
            GroundedFact(
                fact_id="c1_a",
                entity="Delhivery Limited",
                attribute="Incorporation Date",
                raw_value="June 22, 2011",
                canonical_value="2011-06-22",
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date=None, end_date=None, granularity="PERPETUAL", raw_expression="PERPETUAL"),
                    scope="GLOBAL",
                    canonical_unit="RAW",
                ),
                provenance=Provenance(source_doc_name="prospectus.pdf", doc_hash="h1", page_number=14, verbatim_quote="originally incorporated on June 22, 2011"),
            ),
            GroundedFact(
                fact_id="c1_b",
                entity="Delhivery",
                attribute="Incorporation Date",
                raw_value="22nd June 2011",
                canonical_value="2011-06-22",
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date=None, end_date=None, granularity="PERPETUAL", raw_expression="PERPETUAL"),
                    scope="GLOBAL",
                    canonical_unit="RAW",
                ),
                provenance=Provenance(source_doc_name="annual_report.pdf", doc_hash="h2", page_number=2, verbatim_quote="incorporated under Companies Act on 22nd June 2011"),
            ),
            "CASE_1_CORROBORATION",
        ),
        # Case 2: Contradiction (Audited workforce figures for identical period and scope)
        (
            GroundedFact(
                fact_id="c2_a",
                entity="Delhivery",
                attribute="Workforce",
                raw_value="66,000",
                canonical_value=66000,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date="2021-04-01", end_date="2022-03-31", granularity="YEAR", raw_expression="FY22"),
                    scope="CONSOLIDATED",
                    canonical_unit="COUNT",
                ),
                provenance=Provenance(source_doc_name="prospectus.pdf", doc_hash="h1", page_number=20, verbatim_quote="total employees 66,000"),
            ),
            GroundedFact(
                fact_id="c2_b",
                entity="Delhivery",
                attribute="Workforce",
                raw_value="93,000",
                canonical_value=93000,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date="2021-04-01", end_date="2022-03-31", granularity="YEAR", raw_expression="FY22"),
                    scope="CONSOLIDATED",
                    canonical_unit="COUNT",
                ),
                provenance=Provenance(source_doc_name="annual_report.pdf", doc_hash="h2", page_number=40, verbatim_quote="total workforce stood at 93,000"),
            ),
            "CASE_2_GENUINE_CONTRADICTION",
        ),
        # Case 3: Context Reconciled (Temporal progression FY22 vs FY24)
        (
            GroundedFact(
                fact_id="c3_a",
                entity="Delhivery",
                attribute="Revenue",
                raw_value="₹7,241 Cr",
                canonical_value=72410000000.0,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date="2021-04-01", end_date="2022-03-31", granularity="YEAR", raw_expression="FY22"),
                    scope="CONSOLIDATED",
                    canonical_unit="INR",
                ),
                provenance=Provenance(source_doc_name="prospectus.pdf", doc_hash="h1", page_number=35, verbatim_quote="revenue in FY22 was Rs 7,241 Crores"),
            ),
            GroundedFact(
                fact_id="c3_b",
                entity="Delhivery",
                attribute="Revenue",
                raw_value="₹8,142 Cr",
                canonical_value=81420000000.0,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date="2023-04-01", end_date="2024-03-31", granularity="YEAR", raw_expression="FY24"),
                    scope="CONSOLIDATED",
                    canonical_unit="INR",
                ),
                provenance=Provenance(source_doc_name="annual_report.pdf", doc_hash="h2", page_number=10, verbatim_quote="revenue in FY24 reached Rs 8,142 Crores"),
            ),
            "CASE_3_CONTEXT_RECONCILED",
        ),
    ]

    y_true = []
    y_pred = []
    for fa, fb, true_lbl in test_pairs:
        rel = FactReconciler.reconcile(fa, fb)
        y_true.append(true_lbl)
        y_pred.append(rel.case_type)

    target_names = ["CASE_1_CORROBORATION", "CASE_2_GENUINE_CONTRADICTION", "CASE_3_CONTEXT_RECONCILED"]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    print("\n>>> 1. MULTI-CLASS RECONCILIATION BENCHMARK (CASES 1-3):")
    for name in target_names:
        p = report[name]["precision"]
        r = report[name]["recall"]
        f1 = report[name]["f1-score"]
        print(f"  • {name:<30} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")

    macro_f1 = report["macro avg"]["f1-score"]
    print(f"\n  >> RECONCILIATION MACRO-F1: {macro_f1:.4f}")

    # 2. Benchmark for Case 4 (Integrity Gate Anomaly Isolation)
    q_tests = [
        (
            "Operating 86 active gateways.",
            GroundedFact(
                fact_id="q1",
                entity="Delhivery",
                attribute="Gateways",
                raw_value="86",
                canonical_value=86,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date=None, end_date=None, granularity="UNKNOWN", raw_expression=""),
                    scope="INDIA",
                    canonical_unit="COUNT",
                ),
                provenance=Provenance(source_doc_name="d.pdf", doc_hash="h", page_number=1, verbatim_quote="operating 86 active gateways"),
            ),
            True,  # Quarantined due to missing temporal horizon
        ),
        (
            "Revenue reached eight thousand crores.",
            GroundedFact(
                fact_id="q2",
                entity="Delhivery",
                attribute="Revenue",
                raw_value="8000 Cr",
                canonical_value=8e10,
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date="2023-04-01", end_date="2024-03-31", granularity="YEAR", raw_expression="FY24"),
                    scope="CONSOLIDATED",
                    canonical_unit="INR",
                ),
                provenance=Provenance(source_doc_name="d.pdf", doc_hash="h", page_number=1, verbatim_quote="this string does not exist on page"),
            ),
            True,  # Quarantined due to quote hallucination
        ),
        (
            "Incorporated on June 22, 2011.",
            GroundedFact(
                fact_id="q3",
                entity="Delhivery",
                attribute="Incorporation Date",
                raw_value="June 22, 2011",
                canonical_value="2011-06-22",
                context_box=ContextBoundingBox(
                    temporal=TemporalInterval(start_date=None, end_date=None, granularity="PERPETUAL", raw_expression="PERPETUAL"),
                    scope="GLOBAL",
                    canonical_unit="RAW",
                ),
                provenance=Provenance(source_doc_name="d.pdf", doc_hash="h", page_number=1, verbatim_quote="Incorporated on June 22, 2011."),
            ),
            False,  # Valid grounded non-numeric perpetual fact
        ),
    ]

    y_q_true = [t[2] for t in q_tests]
    y_q_pred = []
    for text, fact, _ in q_tests:
        v, q = FactIntegrityGate.validate_facts([fact], page_text=text)
        y_q_pred.append(len(q) > 0)

    p_q, r_q, f1_q, _ = precision_recall_fscore_support(y_q_true, y_q_pred, average="binary", zero_division=0)
    print("\n>>> 2. CIRCUIT BREAKER & QUARANTINE BENCHMARK (CASE 4):")
    print(f"  • Quarantine Precision: {p_q:.2f}")
    print(f"  • Quarantine Recall:    {r_q:.2f}")
    print(f"  • Quarantine F1-Score:  {f1_q:.2f}")

    print("\n" + "=" * 75)
    print(f"FINAL AUDIT VERDICT: Macro-F1 = {macro_f1:.2f} | Quarantine-F1 = {f1_q:.2f}")
    print("=" * 75)


if __name__ == "__main__":
    run_evaluation()
