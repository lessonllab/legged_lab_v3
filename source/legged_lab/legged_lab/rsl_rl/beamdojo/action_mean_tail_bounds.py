"""Explicit identity-interior mean tails; this is a Gaussian, not squashed actions.

The MLP and its parameters remain unchanged. Only the Gaussian location changes
outside an expanded execution interval. All native likelihood/KL/entropy paths
therefore use the same Normal that produced the rollout actions.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from rsl_rl.modules.distribution import GaussianDistribution


class IdentityInteriorMean(nn.Module):
    def __init__(self, lower, upper, identity_margin_halfwidth=.5, tail_width_halfwidth=.5):
        super().__init__()
        lower, upper = torch.as_tensor(lower).detach().clone(), torch.as_tensor(upper).detach().clone()
        margin, width = float(identity_margin_halfwidth), float(tail_width_halfwidth)
        if (lower.ndim != 1 or upper.shape != lower.shape or not torch.isfinite(lower).all()
                or not torch.isfinite(upper).all() or (lower >= upper).any()
                or not math.isfinite(margin) or margin < 0
                or not math.isfinite(width) or width <= 0):
            raise ValueError('Invalid identity-interior mean bounds')
        half = (upper-lower)/2
        # Not model parameters or persistent buffers: old state_dict keys and
        # all optimizer parameter identities stay exactly unchanged.
        self.register_buffer('lower', lower, persistent=False)
        self.register_buffer('upper', upper, persistent=False)
        self.register_buffer('identity_lower', lower-margin*half, persistent=False)
        self.register_buffer('identity_upper', upper+margin*half, persistent=False)
        self.register_buffer('tail_width', width*half, persistent=False)
        self.margin, self.width = margin, width

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        above = torch.relu(raw-self.identity_upper)
        below = torch.relu(self.identity_lower-raw)
        upper_tail = self.identity_upper + self.tail_width*(above/(self.tail_width+above))
        lower_tail = self.identity_lower - self.tail_width*(below/(self.tail_width+below))
        # C1 at either join: derivative is one from each side. Returning raw
        # literally in the interior preserves bitwise identity there.
        return torch.where(raw > self.identity_upper, upper_tail,
                           torch.where(raw < self.identity_lower, lower_tail, raw))

    def metadata(self):
        return {'version': 1, 'kind': 'identity_interior_rational_gaussian_mean',
                'lower': self.lower.cpu().tolist(), 'upper': self.upper.cpu().tolist(),
                'identity_margin_halfwidth': self.margin, 'tail_width_halfwidth': self.width,
                'raw_mlp_penalty': True, 'execution_clamp_unchanged': True}


class MeanTailGaussianDistribution(GaussianDistribution):
    def update(self, mlp_output):
        self.last_raw_mean = mlp_output
        super().update(self.mean_tail(mlp_output))

    def deterministic_output(self, mlp_output):
        self.last_raw_mean = mlp_output
        return self.mean_tail(mlp_output)

    def as_deterministic_output_module(self):
        return deepcopy(self.mean_tail)


def install_mean_tail(distribution, lower, upper, identity_margin_halfwidth=.5, tail_width_halfwidth=.5):
    if type(distribution) not in (GaussianDistribution, MeanTailGaussianDistribution):
        raise ValueError('Mean tails require native scalar GaussianDistribution')
    if distribution.std_type != 'scalar':
        raise ValueError('Mean tails require scalar-parameter Gaussian')
    parameter = distribution.std_param
    transform = IdentityInteriorMean(
        torch.as_tensor(lower, device=parameter.device, dtype=parameter.dtype),
        torch.as_tensor(upper, device=parameter.device, dtype=parameter.dtype),
        identity_margin_halfwidth, tail_width_halfwidth)
    if transform.lower.shape != parameter.shape:
        raise ValueError('Mean-tail bounds must match actor action dimension')
    # Preserve the native Parameter object, including Adam slots and storage.
    distribution.__class__ = MeanTailGaussianDistribution
    distribution.mean_tail = transform
    distribution.last_raw_mean = None
    return transform.metadata()


def remove_mean_tail(distribution):
    if isinstance(distribution, MeanTailGaussianDistribution):
        del distribution.mean_tail
        del distribution.last_raw_mean
        distribution.__class__ = GaussianDistribution
