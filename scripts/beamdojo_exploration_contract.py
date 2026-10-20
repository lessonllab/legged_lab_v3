"""CPU-readable declaration for the explicit fixed-std engineering control."""
import math


def declaration(factor):
    if isinstance(factor,bool) or not isinstance(factor,(int,float)) or not math.isfinite(factor) or not 0<factor<=1:
        raise ValueError('Exploration factor must be finite in (0,1]')
    return dict(version=1,kind='post_native_clamp_gaussian_std_scale',factor=float(factor),
        std_parameter_unchanged=True,likelihood='single_scaled_Normal_sampling_logprob_entropy_KL',
        execution_clamp_unchanged=True)


def validate(value):
    if value is None:return
    if not isinstance(value,dict) or value!=declaration(value.get('factor')):
        raise ValueError('Unknown or incomplete exploration scaling declaration')
