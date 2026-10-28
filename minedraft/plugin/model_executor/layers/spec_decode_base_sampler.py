from typing import List, Optional, Union

import torch
from torch import nn
from vllm.model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSampler
from vllm.platforms import current_platform

from minedraft.patching import MinePatch


class SpecDecodeBaseSamplerPatch(MinePatch[SpecDecodeBaseSampler], nn.Module):
    _orig_init = SpecDecodeBaseSampler.__init__

    def __init__(self, *args, **kwargs):
        self._orig_init(*args, **kwargs)
        self.num_good_draft_tokens: Optional[torch.Tensor] = None
        self.num_verification_tokens: Optional[torch.Tensor] = None
        self.num_req: int = 0

    def init_gpu_tensors(self, device: Union[int, str, torch.device]) -> None:
        self.init_tensors(device, current_platform.device_type)

    def init_tensors(self,
                     device: Union[int, str, torch.device],
                     device_type: Optional[Union[torch.device, str]] = None) -> None:
        assert self.num_accepted_tokens is None
        device_type = device_type or current_platform.device_type
        if isinstance(device_type, torch.device):
            device_type = device_type.type
        if isinstance(device, int):
            device = ("cpu" if device_type == "cpu"
                      else f"{device_type}:{device}")
        self.num_accepted_tokens = torch.tensor(0,
                                                dtype=torch.long,
                                                device=device)
        self.num_good_draft_tokens = torch.tensor(0,
                                                  dtype=torch.long,
                                                  device=device)
        self.num_verification_tokens = torch.tensor(0,
                                                    dtype=torch.long,
                                                    device=device)
        self.num_emitted_tokens = torch.tensor(0,
                                               dtype=torch.long,
                                               device=device)

    def _create_output(
            self,
            accepted: torch.Tensor,  # [batch_size, k]
            substitute_token_ids: torch.Tensor,  # [batch_size, k]
            draft_token_ids: torch.Tensor,  # [batch_size, k]
            bonus_token_ids: torch.Tensor,  # [batch_size]
            proposal_lens_list: Optional[Union[List[int], torch.Tensor]] = None,
            total_num_seqs: Optional[int] = None, # total number of sequences including the current batch and the non-speculated sequences, necessary to calculate emitted tokens correctly
    ) -> torch.Tensor:
        batch_size, k = substitute_token_ids.shape
        total_num_seqs = batch_size if total_num_seqs is None else total_num_seqs
        bonus_token_ids = bonus_token_ids.squeeze(-1)
        rows = torch.arange(batch_size, device=accepted.device)
        indices = torch.arange(k, device=accepted.device).unsqueeze(0)
        if proposal_lens_list is None:
            proposal_lens = torch.full((batch_size,), k, dtype=torch.long,
                                       device=accepted.device)
        else:
            proposal_lens = torch.as_tensor(proposal_lens_list, dtype=torch.long,
                                            device=accepted.device)

        # Stop at the first rejection or at the sequence's actual frontier.
        # Avoid max(bool), which is not supported by every NPU backend.
        if k:
            first_rejection = torch.where(accepted, k, indices).amin(dim=1)
            limits = torch.minimum(first_rejection, proposal_lens)
        else:
            limits = proposal_lens

        output_with_bonus_tokens = torch.full(
            (batch_size, k + self._num_bonus_tokens),
            -1,
            dtype=self.token_id_dtype,
            device=accepted.device)
        output = output_with_bonus_tokens[:, :k]
        output.copy_(torch.where(indices < limits.unsqueeze(1),
                                 draft_token_ids, -1))
        if k:
            recovered = substitute_token_ids.gather(
                1, limits.clamp(max=k - 1).unsqueeze(1)).squeeze(1)
            next_token_ids = torch.where(limits < proposal_lens,
                                         recovered, bonus_token_ids)
        else:
            next_token_ids = bonus_token_ids
        output_with_bonus_tokens[rows, limits] = next_token_ids

        self.num_verification_tokens += proposal_lens.sum()
        self.num_emitted_tokens += ((output_with_bonus_tokens != -1).sum()
                                    + total_num_seqs - batch_size)
        self.num_draft_tokens += batch_size * k
        self.num_accepted_tokens += (
            accepted & (indices < proposal_lens.unsqueeze(1))).sum()
        self.num_good_draft_tokens += limits.sum()
        self.num_req += total_num_seqs

        return output_with_bonus_tokens
