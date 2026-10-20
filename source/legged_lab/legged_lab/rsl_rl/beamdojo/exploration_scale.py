"""Explicit static Gaussian-std engineering control; not a new exploration method.

Native mean/tail and std clamp run first. One final Normal owns sampling and all
PPO density operations. Parameter objects, Adam state and state_dict stay intact.
Remove scaling before installing/removing a mean tail, then reinstall scaling.
Installation/removal invalidates cached distributions: update before sampling.
"""
import math
from torch.distributions import Normal
from rsl_rl.modules.distribution import GaussianDistribution
from .action_mean_tail_bounds import MeanTailGaussianDistribution


def _scale(distribution):
    if distribution.exploration_factor != 1.:
        native=distribution._distribution
        distribution._distribution=Normal(native.mean,native.stddev*distribution.exploration_factor)


class ScaledGaussianDistribution(GaussianDistribution):
    def update(self,mlp_output):
        super().update(mlp_output)
        _scale(self)


class ScaledMeanTailGaussianDistribution(MeanTailGaussianDistribution):
    def update(self,mlp_output):
        super().update(mlp_output)
        _scale(self)


_BASE_TO_SCALED={GaussianDistribution:ScaledGaussianDistribution,
                 MeanTailGaussianDistribution:ScaledMeanTailGaussianDistribution}
_SCALED_TO_BASE={v:k for k,v in _BASE_TO_SCALED.items()}


def metadata(distribution):
    if type(distribution) not in _SCALED_TO_BASE:return None
    return {'version':1,'kind':'post_native_clamp_gaussian_std_scale',
            'factor':distribution.exploration_factor,'std_parameter_unchanged':True,
            'likelihood':'single_scaled_Normal_sampling_logprob_entropy_KL',
            'execution_clamp_unchanged':True}


def install(distribution,factor=.25):
    if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 0<factor<=1:
        raise ValueError('Engineering std factor must be finite in (0,1]')
    cls=type(distribution)
    if cls not in _BASE_TO_SCALED and cls not in _SCALED_TO_BASE:
        raise ValueError('Unknown Gaussian distribution class')
    if distribution.std_type!='scalar':raise ValueError('Only native scalar std parameterization supported')
    distribution.__class__=_BASE_TO_SCALED.get(cls,cls)
    distribution.exploration_factor=float(factor)
    distribution._distribution=None
    return metadata(distribution)


def remove(distribution):
    cls=type(distribution)
    if cls in _SCALED_TO_BASE:
        distribution.__class__=_SCALED_TO_BASE[cls]
        del distribution.exploration_factor
        distribution._distribution=None
    elif cls not in _BASE_TO_SCALED:
        raise ValueError('Unknown distribution class')


def restore(distribution,saved_metadata):
    if saved_metadata is None:
        remove(distribution);return None
    if not isinstance(saved_metadata,dict):raise ValueError('Invalid exploration metadata')
    expected={'version':1,'kind':'post_native_clamp_gaussian_std_scale',
              'factor':saved_metadata.get('factor'),'std_parameter_unchanged':True,
              'likelihood':'single_scaled_Normal_sampling_logprob_entropy_KL',
              'execution_clamp_unchanged':True}
    if saved_metadata!=expected:raise ValueError('Unsupported exploration metadata')
    return install(distribution,saved_metadata['factor'])
