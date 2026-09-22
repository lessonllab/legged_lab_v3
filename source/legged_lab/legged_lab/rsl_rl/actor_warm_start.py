"""Transfer only a compatible actor; retain the new run's exploration settings."""
import hashlib
from pathlib import Path
import torch


def initialize_actor(actor, checkpoint):
    path = Path(checkpoint).resolve()
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    source = loaded['actor_state_dict']
    target = actor.state_dict()
    if set(source) != set(target):
        raise ValueError('Actor checkpoint schema mismatch')
    for name, value in source.items():
        if value.shape != target[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid actor tensor: {name}')
    # This restores CNN/MLP/observation normalization but does not restore old
    # action noise, critic, AMP, optimizers or learning iteration.
    copied = {name: target[name] if name.startswith('distribution.') else value for name, value in source.items()}
    actor.load_state_dict(copied, strict=True)
    return {'checkpoint': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'loaded': 'actor including observation normalizers',
            'fresh': ['action_distribution', 'critic', 'discriminator', 'optimizers', 'iteration']}
