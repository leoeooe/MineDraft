import torch
from vllm.spec_decode.interfaces import SpeculativeProposals


def select_proposals_no_priority(
    capacity: int, proposals: SpeculativeProposals
) -> SpeculativeProposals:
    """Select cumulative-probability frontiers using only portable Torch ops."""
    if capacity < 0:
        raise ValueError("Tetris capacity must be nonnegative")
    mask = proposals.proposal_lens > 0
    token_ids = proposals.proposal_token_ids[mask]
    if token_ids.numel() == 0:
        return proposals
    logprobs = proposals.proposal_logprobs[mask]
    positions = torch.arange(1, token_ids.size(1) + 1,
                             device=token_ids.device)
    valid = positions.unsqueeze(0) <= proposals.proposal_lens[mask].unsqueeze(1)
    # Padding can contain INVALID_TOKEN_ID (-1); it must never enter gather.
    safe_token_ids = torch.where(valid, token_ids, 0)
    values = logprobs.gather(-1, safe_token_ids.unsqueeze(-1)).squeeze(-1)
    values = values.cumsum(-1).masked_fill(~valid, float("-inf"))
    selected = torch.zeros_like(values, dtype=torch.bool)
    if capacity:
        best_indices = torch.topk(values.flatten(),
                                  min(capacity, values.numel())).indices
        selected.view(-1)[best_indices] = True
    # A dense mask avoids the torch-scatter extension and scatter_reduce
    # backend dependencies. This is the same maximum frontier per sequence.
    frontiers = ((selected & valid).to(positions.dtype)
                 * positions.unsqueeze(0)).amax(-1)
    proposals.proposal_lens[mask] = frontiers
    return proposals
