"""Optional reward-only observation; never fed to the policy or discriminator."""
import torch


def get_style_gate(obs, dones, extras, group):
    gate = obs[group].reshape(-1).clone()
    if gate.shape != dones.shape:
        raise ValueError('Style gate must contain exactly one value per environment')
    terminal = extras.get('terminal_obs', {})
    if dones.bool().any():
        if group not in terminal:
            raise ValueError('Terminal style gate missing: refusing post-reset reward leakage')
        gate[dones.bool()] = terminal[group].reshape(-1)[dones.bool()]
    if not torch.isfinite(gate).all() or ((gate < 0.) | (gate > 1.)).any():
        raise ValueError('Style gate must be finite and between zero and one')
    return gate
