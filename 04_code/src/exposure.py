"""Pure cohort-interval utilities for the frozen exposure algorithm."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class InclusiveInterval:
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("lower must not exceed upper")

    @property
    def width(self) -> int:
        return self.upper - self.lower + 1


def outcome_cohort_interval(year: int, age_lower: int, age_upper: int) -> InclusiveInterval:
    """Map a calendar-year/age band to an inclusive birth-cohort interval."""
    if age_lower < 0 or age_upper < age_lower:
        raise ValueError("invalid age interval")
    return InclusiveInterval(year - age_upper, year - age_lower)


def target_cohort_interval(programme_year: int, target_age_lower: int,
                           target_age_upper: int) -> InclusiveInterval:
    """Map programme target ages to an inclusive birth-cohort interval."""
    if target_age_lower < 0 or target_age_upper < target_age_lower:
        raise ValueError("invalid target-age interval")
    return InclusiveInterval(
        programme_year - target_age_upper,
        programme_year - target_age_lower,
    )


def merged_union(intervals: list[InclusiveInterval]) -> list[InclusiveInterval]:
    """Return a sorted union, merging overlapping or adjacent intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.lower <= previous.upper + 1:
            merged[-1] = InclusiveInterval(previous.lower, max(previous.upper, current.upper))
        else:
            merged.append(current)
    return merged


def overlap_years(subject: InclusiveInterval,
                  eligible: list[InclusiveInterval]) -> set[int]:
    years: set[int] = set()
    for interval in merged_union(eligible):
        lower = max(subject.lower, interval.lower)
        upper = min(subject.upper, interval.upper)
        if lower <= upper:
            years.update(range(lower, upper + 1))
    return years


def classify_eligibility(
    subject: InclusiveInterval,
    eligible_intervals: list[InclusiveInterval],
    *,
    national_scope: bool,
    ambiguous: bool,
) -> dict[str, int | float]:
    """Classify a cohort band under the frozen full-overlap primary rule."""
    overlap = overlap_years(subject, eligible_intervals)
    share = len(overlap) / subject.width
    eligible_full = int(national_scope and not ambiguous and share == 1.0)
    return {
        "eligible_full": eligible_full,
        "eligible_partial": int(0.0 < share < 1.0),
        "eligibility_ambiguous": int(ambiguous or not national_scope),
        "eligible_share": share,
    }

