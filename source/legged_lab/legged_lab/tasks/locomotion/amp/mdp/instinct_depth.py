"""Instinct-style cropped depth and delayed, sparsely sampled camera history.

History is maintained here (once per control step), not by ObservationManager:
eight consecutive control frames do not represent Instinct's 37-frame buffer.
"""

import torch
import torch.nn.functional as F

from isaaclab.managers import ManagerTermBase


def preprocess_instinct_depth(raw: torch.Tensor) -> torch.Tensor:
    """64x36 optical depth -> 32x18 crop, 3x3 Gaussian blur, [0, 1].

    The reference crop has no resize_shape: remove top 18 rows and 16 columns
    on each side. Missing/far returns and positive returns below 0.1 m are
    filled with 2.5 m, following the reference camera's near clipping.
    """
    if raw.ndim != 4 or tuple(raw.shape[1:]) != (36, 64, 1):
        raise ValueError(f"Expected N x 36 x 64 x 1 depth, got {tuple(raw.shape)}")
    x = torch.nan_to_num(raw[:, 18:, 16:-16, 0], nan=2.5, posinf=2.5, neginf=0.0)
    x = torch.where((x > 0) & (x < 0.1), 2.5, x)
    x = x.clamp(0.0, 2.5).unsqueeze(1)
    axis = torch.arange(-1, 2, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * axis.square())
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel).squeeze(1) / 2.5


class InstinctDepthHistory(ManagerTermBase):
    """37 frames at 50 Hz; offsets 35,30,...,0 and per-episode 0/1 frame delay.

    Reset fills only the affected environments with their new frame. Multiple
    reads at the same simulation step must not advance time or duplicate frames.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        with torch.inference_mode(False):
            self._frames = torch.zeros(env.num_envs, 37, 18, 32, device=env.device)
            self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            self._delay = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            self._offsets = torch.arange(35, -1, -5, device=env.device)
        self._pointer = -1
        self._last_step = None
        self._needs_refill = True
        self._cached = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._pending[ids] = True
        self._delay[ids] = torch.randint(0, 2, self._delay[ids].shape, device=self.device)
        self._needs_refill = True

    def __call__(self, env, sensor_name="depth_camera"):
        step = env.common_step_counter
        if self._last_step == step and not self._needs_refill:
            return self._cached.clone()
        raw = env.scene[sensor_name].data.output["distance_to_image_plane"]
        if not isinstance(raw, torch.Tensor):
            raw = raw.torch
        frame = preprocess_instinct_depth(raw)
        if self._last_step != step:
            self._pointer = (self._pointer + 1) % 37
            self._frames[:, self._pointer] = frame
            self._last_step = step
        if self._needs_refill:
            self._frames[self._pending] = frame[self._pending, None]
            self._pending[:] = False
            self._needs_refill = False
        indices = (self._pointer - self._offsets[None, :] - self._delay[:, None]) % 37
        self._cached = self._frames[torch.arange(self.num_envs, device=self.device)[:, None], indices]
        return self._cached.clone()
