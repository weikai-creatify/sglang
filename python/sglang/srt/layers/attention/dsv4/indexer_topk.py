from typing import Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsv4 import (
    topk_transform_paged,
    topk_transform_paged_v2,
)
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks
from sglang.srt.layers.attention.dsv4.indexer_plan import CandidateRole, IndexerPlan


def apply_decode_candidates(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    plan: IndexerPlan,
    topk_blocks: int,
    block_size: int,
    published: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply candidate-block filtering and return logits plus an optional published mask.

    Mask columns past each sequence length to -inf before selection: the paged
    logits kernel leaves that tail uninitialized, and an all -inf block means
    unreachable. This path is graph-captured and must not synchronize with the host.
    Work scales with allocated page-table capacity, not live sequence length.
    """
    if plan.candidate_action is CandidateRole.NONE:
        return logits, None

    is_candidate_source = plan.candidate_action is CandidateRole.PUBLISH

    if (
        logits.is_cuda
        and torch.version.cuda is not None
        and logits.ndim == 2
        and logits.stride(1) == 1
        and seq_lens.device == logits.device
        and seq_lens.dtype in (torch.int32, torch.int64)
        and seq_lens.is_contiguous()
        and seq_lens.shape in ((logits.shape[0],), (logits.shape[0], 1))
        and logits.numel() > 0
        and 0 < block_size <= 1024
    ):
        from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
            candidate_block_logits,
        )

        if not is_candidate_source:
            assert (
                torch.is_tensor(published)
                and published.shape[0] == logits.shape[0]
                and published.shape[1] >= logits.shape[1]
            ), "candidate mask missing for decode"
        return candidate_block_logits(
            logits,
            seq_lens,
            topk_blocks=topk_blocks,
            block_size=block_size,
            published=None if is_candidate_source else published,
        )

    lens_col = seq_lens if seq_lens.dim() > 1 else seq_lens.unsqueeze(-1)
    reachable = torch.arange(logits.shape[-1], device=logits.device) < lens_col
    logits = logits.float().masked_fill(~reachable, -torch.inf)

    if is_candidate_source:
        # The source scores over every reachable position itself and only publishes,
        # which is what the reference does.
        return logits, select_candidate_blocks(
            logits, lens_col, topk_blocks=topk_blocks, block_size=block_size
        )

    assert torch.is_tensor(published) and published.shape[0] == logits.shape[0], (
        "candidate mask missing for decode"
    )
    return logits.masked_fill(~published[:, : logits.shape[-1]], -torch.inf), None


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


def write_paged_indexer_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    raw_indices: Optional[torch.Tensor],
    *,
    page_size: int,
    use_topk_v2: bool,
    topk_metadata,
    mask_topk: bool,
) -> None:
    selected_indices = torch.empty_like(page_indices) if mask_topk else raw_indices
    if use_topk_v2 and raw_indices is None:
        topk_transform_paged_v2(
            logits,
            seq_lens,
            None if mask_topk else page_table,
            selected_indices if mask_topk else page_indices,
            page_size,
            topk_metadata,
        )
    else:
        topk_transform_paged(
            logits,
            seq_lens,
            page_table,
            page_indices,
            page_size,
            selected_indices,
        )
    if mask_topk:
        selected_indices = mask_topk_scores(logits, selected_indices)
        columns = selected_indices.clamp_min(0).to(torch.int64)
        slots = page_table.gather(1, columns // page_size) * page_size
        slots = slots + columns % page_size
        page_indices.copy_(torch.where(selected_indices >= 0, slots, -1))
        if raw_indices is not None:
            raw_indices.copy_(selected_indices)
