"""
Allen's Interval Algebra Engine for Temporal Coordinate Evaluations.
Determines temporal relations between fact context windows.
"""

from typing import Optional
from .models import TemporalInterval


class IntervalRelation:
    """
    Formal interval relations defined by Allen's Interval Algebra.
    """
    EQUAL = "EQUAL"                    # T1 == T2
    DISJOINT = "DISJOINT"              # T1 ends before T2 begins (or vice versa)
    SUBSET = "SUBSET"                  # T1 is strictly contained within T2
    SUPERSET = "SUPERSET"              # T1 strictly contains T2
    OVERLAPPING = "OVERLAPPING"        # Partial intersection
    PERPETUAL = "PERPETUAL"            # One or both intervals are timeless / perpetual


def evaluate_temporal_intervals(
    t1: Optional[TemporalInterval],
    t2: Optional[TemporalInterval],
) -> str:
    """
    Evaluate the formal temporal relationship between two TemporalInterval objects.

    Args:
        t1: First TemporalInterval
        t2: Second TemporalInterval

    Returns:
        One of the IntervalRelation constants.
    """
    # 1. Check Perpetual or missing intervals
    if not t1 or not t2:
        return IntervalRelation.PERPETUAL

    if t1.granularity == "PERPETUAL" or t2.granularity == "PERPETUAL":
        return IntervalRelation.PERPETUAL

    if not t1.start_date or not t1.end_date or not t2.start_date or not t2.end_date:
        return IntervalRelation.PERPETUAL

    # 2. Check EQUAL
    if t1.start_date == t2.start_date and t1.end_date == t2.end_date:
        return IntervalRelation.EQUAL

    # 3. Check DISJOINT (T1 ends strictly before T2 starts, or T2 ends strictly before T1 starts)
    if t1.end_date < t2.start_date or t2.end_date < t1.start_date:
        return IntervalRelation.DISJOINT

    # 4. Check SUBSET (T1 is contained inside T2)
    if t1.start_date >= t2.start_date and t1.end_date <= t2.end_date:
        return IntervalRelation.SUBSET

    # 5. Check SUPERSET (T2 is contained inside T1)
    if t2.start_date >= t1.start_date and t2.end_date <= t1.end_date:
        return IntervalRelation.SUPERSET

    # 6. Fallback: OVERLAPPING
    return IntervalRelation.OVERLAPPING
