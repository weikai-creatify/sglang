from enum import Enum, auto
from typing import NamedTuple, Optional, Sequence


class CandidateRole(Enum):
    NONE = auto()
    PUBLISH = auto()
    CONSUME = auto()

    @classmethod
    def for_layer(cls, layer_id: int, source_layer_id: int) -> "CandidateRole":
        if layer_id == source_layer_id:
            return cls.PUBLISH
        if 0 <= source_layer_id < layer_id:
            return cls.CONSUME
        return cls.NONE


class IndexerPlan(NamedTuple):
    select_all: bool
    candidate_action: CandidateRole
    mask_topk: bool


CANDIDATE_FILTERED = "candidate_filtered"


def candidate_graph_limits(
    ratios: Sequence[int], index_topk: int, candidate_span: int
) -> list[tuple[str, int]]:
    low_ratios = set(ratios) & {1, 2}
    variants = []
    if index_topk > 0 and low_ratios:
        variants.append(("candidate_all", index_topk * min(low_ratios)))
        if low_ratios == {1, 2}:
            variants.append(("candidate_c2_all", index_topk * 2))
    variants.append(("candidate_unfiltered", candidate_span))
    limits = []
    for variant, limit in variants:
        limits.append((variant, min(limit, candidate_span)))
        if limit >= candidate_span:
            break
    return limits


def resolve_indexer_plan(
    compress_ratio: int,
    candidate_role: CandidateRole = CandidateRole.NONE,
    capture_variant: Optional[str] = None,
) -> IndexerPlan:
    select_all = capture_variant == "candidate_all" or (
        capture_variant == "candidate_c2_all" and compress_ratio == 2
    )
    bypass_candidates = capture_variant in (
        "candidate_all",
        "candidate_c2_all",
        "candidate_unfiltered",
    )
    candidate_action = CandidateRole.NONE if bypass_candidates else candidate_role
    # Consumers must reject masked top-k entries even when block filtering is bypassed.
    return IndexerPlan(
        select_all, candidate_action, candidate_role is CandidateRole.CONSUME
    )
