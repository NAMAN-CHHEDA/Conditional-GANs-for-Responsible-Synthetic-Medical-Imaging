"""Import shim: Ray >= 2.41 exposes report/Checkpoint/RunConfig under ray.tune;
older Ray 2.x has them under ray.train."""
try:  # new API (Ray >= 2.41)
    from ray.tune import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, get_checkpoint, report
except ImportError:  # pragma: no cover - older Ray
    from ray.train import Checkpoint, CheckpointConfig, FailureConfig, RunConfig, get_checkpoint, report

__all__ = ["Checkpoint", "CheckpointConfig", "FailureConfig", "RunConfig", "get_checkpoint", "report"]
