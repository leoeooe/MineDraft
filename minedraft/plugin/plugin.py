import os

_PATCHES_APPLIED = False


def minedraft_plugin():
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        # Loading the installed plugin must not change ordinary Ascend/V1 runs.
        if os.environ.get("VLLM_ASCEND_ENABLE_MINEDRAFT", "0") != "1":
            return
        from vllm import envs
        if envs.VLLM_USE_V1:
            raise ValueError("MineDraft on Ascend requires VLLM_USE_V1=0")
        # Load Ascend's factory/distributed patches before capturing originals.
        from vllm_ascend.utils import adapt_patch
        adapt_patch()

    from .config import SpeculativeConfigPatch, VllmConfigPatch
    from .core.scheduler import (
        SchedulerModulePatch,
        SchedulerOutputsPatch,
        SchedulerPatch,
        SchedulerRunningOutputsPatch,
    )
    from .distributed.communication_op import CommunicationOpModulePatch
    from .distributed.parallel_state import GroupCoordinatorPatch, ParallelStateModulePatch
    from .engine.async_llm_engine import AsyncLLMEnginePatch
    from .engine.llm_engine import LLMEnginePatch
    from .model_executor.layers.rejection_sampler import RejectionSamplerPatch
    from .model_executor.layers.spec_decode_base_sampler import SpecDecodeBaseSamplerPatch
    from .sequence import HiddenStatesPatch, SequenceGroupPatch
    from .spec_decode.interfaces import SpeculativeProposalsPatch
    from .spec_decode.metrics import AsyncMetricsCollectorPatch
    from .spec_decode.mqa_scorer import MQAScorerPatch
    from .spec_decode.spec_decode_worker import (
        SpecDecodeWorkerModulePatch,
        SpecDecodeWorkerPatch,
    )
    from .spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorkerPatch
    from .spec_decode.top1_proposer import Top1ProposerPatch
    from .spec_decode.util import TimerPatch
    from .worker.worker_base import LocalOrDistributedWorkerBasePatch

    # Apply patches in distributed module
    GroupCoordinatorPatch.apply_patch()
    ParallelStateModulePatch.apply_patch()
    CommunicationOpModulePatch.apply_patch()

    # Apply patches for CLI arguments and configs
    SpeculativeConfigPatch.apply_patch()
    VllmConfigPatch.apply_patch()

    # Apply patches for worker module
    LocalOrDistributedWorkerBasePatch.apply_patch()
    if current_platform.is_cuda_alike():
        from .model_executor.models.eagle import EAGLEPatch
        from .worker.model_runner import GPUModelRunnerBasePatch, ModelRunnerPatch
        from .worker.worker import WorkerModulePatch, WorkerPatch

        WorkerModulePatch.apply_patch()
        WorkerPatch.apply_patch()
        GPUModelRunnerBasePatch.apply_patch()
        ModelRunnerPatch.apply_patch()
        EAGLEPatch.apply_patch()

    # Apply patches in model_executor module
    SpecDecodeBaseSamplerPatch.apply_patch()
    RejectionSamplerPatch.apply_patch()

    # Apply patches in sequence module
    SequenceGroupPatch.apply_patch()
    HiddenStatesPatch.apply_patch()

    # Apply patches for scheduler
    SchedulerOutputsPatch.apply_patch()
    SchedulerRunningOutputsPatch.apply_patch()
    SchedulerModulePatch.apply_patch()
    SchedulerPatch.apply_patch()

    # Apply patches for engines
    LLMEnginePatch.apply_patch()
    AsyncLLMEnginePatch.apply_patch()

    # Apply patches in spec_decode module
    TimerPatch.apply_patch()
    AsyncMetricsCollectorPatch.apply_patch()
    SpeculativeProposalsPatch.apply_patch()
    MQAScorerPatch.apply_patch()
    Top1ProposerPatch.apply_patch()
    SpecDecodeWorkerModulePatch.apply_patch()
    SpecDecodeWorkerPatch.apply_patch()
    SmallerTpProposerWorkerPatch.apply_patch()
    _PATCHES_APPLIED = True
