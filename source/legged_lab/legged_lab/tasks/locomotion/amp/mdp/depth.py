"""Metric depth preprocessing shared by simulation and deployment inputs."""

from __future__ import annotations

import torch


def preprocess_depth(depth: torch.Tensor, near: float = 0.1, far: float = 2.5) -> torch.Tensor:
    """Convert optical-axis depth in metres from NHW/NHW1 to NHW.

    Valid depths map to [0, 1]. Missing, non-positive, or too-near returns map
    to -1 so missing pixels are distinguishable from close obstacles. Finite
    depths beyond ``far`` saturate at 1. Never mutate the camera's buffer.
    """
    if not 0 < near < far:
        raise ValueError("Depth bounds must satisfy 0 < near < far.")
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if depth.ndim != 3:
        raise ValueError(f"Expected depth NHW or NHW1, got {tuple(depth.shape)}.")
    depth = depth.float()
    valid = torch.isfinite(depth) & (depth >= near)
    normalized = (depth.clamp(near, far) - near) / (far - near)
    return torch.where(valid, normalized, torch.full_like(normalized, -1.0))


def depth_image(env, sensor_name: str = "depth_camera", near: float = 0.1, far: float = 2.5) -> torch.Tensor:
    """Read a camera frame; ObservationManager owns history and episode reset."""
    depth = env.scene[sensor_name].data.output["distance_to_image_plane"]
    # Isaac Lab 3 sensor backends may return a ProxyArray instead of a Tensor.
    if not isinstance(depth, torch.Tensor):
        depth = depth.torch
    return preprocess_depth(depth, near=near, far=far)
