"""Native MLP with independently resettable observation-feature statistics.

Per-feature counts permit replacing a terrain sensor without discarding learned
proprioception statistics. This is deliberately independent of PPO and sensor
code; configure both native actor/critic class_name as
``legged_lab.rsl_rl.beamdojo:BeamDojoMLPModel``. For an actor-only sensor transfer,
load the actor then call ``actor.reset_observation_group('terrain')``. Recreate
critics/optimizer separately. Resetting statistics does not reset actor weights.
"""

import torch
from collections import OrderedDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.normalization import EmpiricalNormalization


class FeatureEmpiricalNormalization(EmpiricalNormalization):
    """Empirical normalization with [1, features] counts and no sample cutoff.

    Mean/variance/std names and forward behavior match native RSL-RL 5.4.1.
    Scalar-count native checkpoints are upgraded on load by broadcasting count.
    All features still observe each incoming batch; counts only differ following
    selective reset. No sensor missing-data imputation is performed here.
    """

    def __init__(self, shape: int, eps: float = 1e-2):
        if not isinstance(shape, int) or shape < 1:
            raise ValueError("Feature normalization requires a positive flat feature count")
        super().__init__(shape, eps=eps, until=None)
        self.count = torch.zeros_like(self._mean, dtype=torch.long)

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x):
        if not self.training or x.shape[0] == 0:
            return
        if x.ndim != 2 or x.shape[-1] != self._mean.shape[-1]:
            raise ValueError("Expected [batch, features] observations")
        n = x.shape[0]
        self.count.add_(n)
        rate = n / self.count.to(self._mean.dtype)
        batch_var, batch_mean = torch.var_mean(x, dim=0, correction=0, keepdim=True)
        delta = batch_mean - self._mean
        self._mean.add_(rate * delta)
        self._var.add_(rate * (batch_var - self._var + delta * (batch_mean - self._mean)))
        self._var.clamp_(min=0.)  # round-off must not produce sqrt(NaN)
        self._std.copy_(self._var.sqrt())

    @torch.jit.unused
    @torch.no_grad()
    def reset_features(self, start: int, stop: int):
        """Clear [start, stop) while retaining every other feature bit-for-bit."""
        if not 0 <= start < stop <= self._mean.shape[-1]:
            raise ValueError("Invalid feature reset interval")
        self._mean[:, start:stop] = 0.
        self._var[:, start:stop] = 1.
        self._std[:, start:stop] = 1.
        self.count[:, start:stop] = 0

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        key = prefix + "count"
        if key in state_dict and state_dict[key].numel() == 1:
            state_dict[key] = state_dict[key].reshape(1, 1).expand_as(self.count).clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class BeamDojoMLPModel(MLPModel):
    """Native MLPModel changing only the observation normalizer implementation."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        if self.obs_normalization:
            self.obs_normalizer = FeatureEmpiricalNormalization(self.obs_dim)
        if len(set(self.obs_groups)) != len(self.obs_groups):
            raise ValueError("Observation groups must be unique for selective normalization reset")
        self._observation_group_slices = {}
        start = 0
        for name in self.obs_groups:
            stop = start + obs[name].shape[-1]
            self._observation_group_slices[name] = (start, stop)
            start = stop

    @torch.jit.unused
    def reset_observation_group(self, name: str):
        """Reset one group's statistics; error on typos or disabled normalization."""
        if name not in self._observation_group_slices:
            raise KeyError(f"Unknown observation group {name!r}; expected {self.obs_groups}")
        if not self.obs_normalization:
            raise RuntimeError("Observation normalization is disabled")
        self.obs_normalizer.reset_features(*self._observation_group_slices[name])


class RecomputedMLP(torch.nn.Sequential):
    """Trade recomputation for activation memory; same layers and gradients.

    The 4096 x 100 / 4 source minibatch contains 102400 samples. Keeping
    activations of an actor and two source-sized critics at once exceeds the
    local 16 GB GPU. Checkpointing saves the inputs and recomputes the exact
    deterministic ELU MLP during backward, without changing the minibatch,
    optimizer step count, network dimensions, precision or loss coefficients.
    """
    def _dense(self, x):
        for layer in self:
            x = layer(x)
        return x

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._dense, x, use_reentrant=False)
        return self._dense(x)


class UpstreamMLPModel(BeamDojoMLPModel):
    """Source MLP with activation recomputation for the local GPU capacity."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # MLP reuses the same stateless activation object at several indices.
        # children() deduplicates those references and would drop ELU layers.
        # Retain the complete ordered layer mapping and checkpoint key names.
        self.mlp = RecomputedMLP(OrderedDict(self.mlp._modules.items()))
