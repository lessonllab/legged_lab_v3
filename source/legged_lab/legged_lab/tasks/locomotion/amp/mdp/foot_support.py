"""Terrain-only sole support proxies; never exposed as policy observations."""
import torch
from isaaclab.utils.math import quat_apply_inverse


def sole_ray_pattern(cfg, device):
    """27 sole samples and a 2–2.5 cm perimeter, matching G1 collision anchors.

    Interior x=[-.05,.12], y=[-.025,.025]; sole z=-.035 relative
    to ankle_roll_link. The surrounding ring detects nearby height edges.
    """
    y, x = torch.meshgrid(torch.linspace(-.05, .05, 5, device=device),
                          torch.linspace(-.07125, .14125, 11, device=device), indexing="ij")
    starts = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), .3)), -1)
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1
    return starts, directions


def support_costs(local_heights, forces_z, contact_time, tolerance=.025, edge_height=.05,
                  support_target=.85, contact_ramp=0., immediate_edge=False, graded_edge=False):
    """Return per-env deficit/edge costs and per-foot supported fractions.

    local_heights: [N, feet, 5, 11], measured along the sole normal.
    Missing returns count as unsupported. Smooth slopes aligned with the sole
    have constant local height and therefore no edge cost.
    """
    valid = torch.isfinite(local_heights)
    supported = valid & ((local_heights + .035).abs() <= tolerance)
    fraction = supported[..., 1:-1, 1:-1].float().mean(dim=(-1, -2))
    safe = torch.where(valid, local_heights, torch.zeros_like(local_heights))
    dx = (safe[..., 1:] - safe[..., :-1]).abs()
    dy = (safe[..., 1:, :] - safe[..., :-1, :]).abs()
    edge = ((dx > edge_height).any(dim=(-1, -2)) |
            (dy > edge_height).any(dim=(-1, -2)) | (~valid).any(dim=(-1, -2)))
    # Retain the legacy landing grace unless a ramp is configured. The optional
    # immediate edge cost applies as soon as the foot carries load, while still
    # ignoring swing and weak contacts; support deficit keeps its landing ramp.
    if not 0. < support_target <= 1. or contact_ramp < 0.:
        raise ValueError('Invalid sole support target or contact ramp')
    age = ((contact_time / contact_ramp).clamp(0., 1.) if contact_ramp > 0.
           else (contact_time >= .08))
    contact_load = ((forces_z - 20.) / 80.).clamp(0., 1.)
    load = contact_load * age
    deficit = ((support_target - fraction) / support_target).clamp_min(0.)
    edge_load = contact_load if immediate_edge else load
    edge_cost = edge.float() * edge_load
    if graded_edge:
        # Mild cost for a fully supported foot near the lip; larger cost for
        # missing support under substantial load. Unloading naturally fades it.
        # Contact age is stance age, NOT uninterrupted time near an edge.
        severity = .15 + 2. * deficit
        graded_load = ((forces_z - 20.) / 130.).clamp(0., 2.)
        stance_factor = .5 + .5 * (contact_time / .12).clamp(0., 1.)
        edge_cost = edge.float() * severity * graded_load * stance_factor
    return (deficit * load).sum(-1), edge_cost.sum(-1), fraction


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def foot_support_penalty(env, asset_cfg, contact_cfg, sensor_names, kind,
                         support_target=.85, contact_ramp=0., immediate_edge=False, graded_edge=False):
    """Negative-weight term; no standing bonus and no extra actor/critic input."""
    robot = env.scene[asset_cfg.name]
    body_pos = _tensor(robot.data.body_pos_w)[:, asset_cfg.body_ids]
    body_quat = _tensor(robot.data.body_quat_w)[:, asset_cfg.body_ids]
    heights = []
    for i, name in enumerate(sensor_names):
        hits = _tensor(env.scene[name].data.ray_hits_w)
        q = body_quat[:, i:i+1].expand(-1, hits.shape[1], -1)
        local = quat_apply_inverse(q, hits - body_pos[:, i:i+1])
        heights.append(local[..., 2].reshape(-1, 5, 11))
    contact = env.scene[contact_cfg.name].data
    force = _tensor(contact.net_normal_forces_w)[:, contact_cfg.body_ids, 2]
    duration = _tensor(contact.current_contact_time)[:, contact_cfg.body_ids]
    deficit, edge, _ = support_costs(torch.stack(heights, 1), force, duration,
                                    support_target=support_target, contact_ramp=contact_ramp,
                                    immediate_edge=immediate_edge, graded_edge=graded_edge)
    if kind == "support":
        return deficit
    if kind == "edge":
        return edge
    raise ValueError(f"Unknown foot support cost: {kind}")
