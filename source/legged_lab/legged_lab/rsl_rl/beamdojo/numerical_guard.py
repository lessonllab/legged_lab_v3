"""Opt-in fail-fast numerical diagnostics; never sanitizes training tensors."""
import json
import math
from pathlib import Path

import torch


def _json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


class NumericalGuard:
    def __init__(self, path, *, reward_abs_max=1000., action_abs_max=100.,
                 observation_abs_max=1e6, value_abs_max=1e4, returns_abs_max=1e5):
        self.path = Path(path)
        self.limits = dict(reward=float(reward_abs_max), action=float(action_abs_max),
                           observation=float(observation_abs_max), value=float(value_abs_max),
                           returns=float(returns_abs_max))
        if any(not math.isfinite(v) or v <= 0 for v in self.limits.values()):
            raise ValueError('Numerical guard bounds must be positive and finite')
        self.failed = None

    def _assert_active(self):
        if self.failed is not None:
            raise RuntimeError(self.failed)

    def check_observations(self, observations, *, stage, step, context):
        self._assert_active()
        for name, value in observations.items():
            if hasattr(value, 'items'):
                self.check_observations(value, stage=stage + '.' + str(name), step=step, context=context)
            elif isinstance(value, torch.Tensor):
                self.check('observations.' + str(name), value, 'observation', stage=stage,
                           step=step, context=context)

    def check(self, field, tensor, kind, *, stage, step, context, env_axis=0, positive=False):
        self._assert_active()
        bad = ~torch.isfinite(tensor) | (tensor.abs() > self.limits[kind])
        if positive:
            bad |= tensor <= 0
        if not bad.any():
            return
        positions = bad.nonzero()[:8].detach().cpu().tolist()
        if tensor.ndim:
            affected = bad.movedim(env_axis, 0).reshape(tensor.shape[env_axis], -1).any(-1)
            ids = affected.nonzero().flatten()[:4].detach().cpu().tolist()
            count = int(affected.sum())
        else:
            ids, count = [], 0
        values = [tensor[tuple(index)].detach().cpu().item() for index in positions]
        record = {'stage': stage, 'step': int(step), 'field': field,
                  'threshold_abs': self.limits[kind], 'requires_positive': positive,
                  'offending_env_ids': ids, 'offending_env_count': count,
                  'offending_indices': positions, 'offending_values': values,
                  'tensor_shape': list(tensor.shape),
                  'offending_samples': self._sample(tensor, ids, env_axis),
                  'context': self._sample(context, ids)}
        self._fail(record)

    def capture_exception(self, error, *, stage, step, context):
        self._assert_active()
        # Computation may fail inside the native distribution before returning
        # an action. Preserve up to four rows, clearly not claimed as culprits.
        self._fail({'stage': stage, 'step': int(step), 'field': 'native_act_exception',
                    'error': repr(error)[:2000], 'offending_env_ids': [],
                    'context_sample_env_ids': list(range(4)),
                    'context': self._sample(context, list(range(4)))})

    def check_finite(self, field, tensor, *, stage, step):
        """Check parameter/optimizer tensors without imposing a magnitude bound.

        These axes describe weights, not environments. Keep only a few bad
        coordinates rather than copying model matrices into the incident file.
        """
        self._assert_active()
        bad = ~torch.isfinite(tensor)
        if not bad.any():
            return
        positions = bad.nonzero()[:8].detach().cpu().tolist()
        self._fail({'stage': stage, 'step': int(step), 'field': field,
                    'check': 'finite_only', 'offending_env_ids': [],
                    'offending_indices': positions,
                    'offending_values': [tensor[tuple(i)].detach().cpu().item() for i in positions],
                    'tensor_shape': list(tensor.shape)})

    def _sample(self, value, ids, axis=0):
        if isinstance(value, torch.Tensor):
            if value.ndim:
                safe_ids = [i for i in ids[:4] if i < value.shape[axis]]
                value = value.index_select(axis, torch.tensor(safe_ids, device=value.device, dtype=torch.long))
            return {'shape': list(value.shape), 'values': value.detach().cpu().tolist()}
        if hasattr(value, 'items'):
            return {str(k): self._sample(v, ids) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [self._sample(v, ids) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return type(value).__name__

    def _fail(self, record):
        self.failed = f"Numerical incident at {record['stage']}: {record['field']}; record: {self.path}"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('x') as stream:
                json.dump(_json(record), stream, indent=2, allow_nan=False)
                stream.write('\n')
        except Exception as error:
            raise RuntimeError(self.failed + f'; could not write record: {error}') from error
        raise RuntimeError(self.failed)
