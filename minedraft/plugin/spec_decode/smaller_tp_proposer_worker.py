"""Collectively create the NPU drafter group before any rank returns."""

from vllm.distributed.parallel_state import get_tp_group, init_model_parallel_group
from vllm.platforms import current_platform
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker

from minedraft.patching import MinePatch


class SmallerTpProposerWorkerPatch(MinePatch[SmallerTpProposerWorker]):
    _orig_init_device = SmallerTpProposerWorker.init_device

    def init_device(self):
        if current_platform.device_type != "npu":
            return self._orig_init_device()

        import torch
        from vllm.distributed import parallel_state

        group = get_tp_group()
        self._is_dummy = group.rank not in self._draft_ranks
        backend = torch.distributed.get_backend(group.device_group)
        # new_group must be entered by all world ranks, including nonmembers.
        # Singleton groups for idle ranks also give every coordinator valid
        # local state without adding them to the drafter's collectives.
        rank_groups = [self._draft_ranks] + [
            [rank] for rank in range(torch.distributed.get_world_size())
            if rank not in self._draft_ranks
        ]
        self._tp_group = init_model_parallel_group(
            rank_groups, group.local_rank, backend, group_name="minedraft_draft")
        parallel_state._MINEDRAFT_DRAFT_TP = self._tp_group
        if self._is_dummy:
            return
        with self._patch_tensor_parallel_group():
            self._worker.init_device()
