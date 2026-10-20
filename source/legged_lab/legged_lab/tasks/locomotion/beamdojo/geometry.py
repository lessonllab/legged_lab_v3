"""G1 sole sampling and contact-conditioned support geometry (pure Torch)."""
import torch


def g1_sole_points(device='cpu', dtype=torch.float32, x_bounds=(-.055, .145),
                   y_bounds=(-.035, .035), sole_z=-.035, counts=(7, 5)):
    """35-point full-collision safety projection in each ankle_roll_link frame.

    Read-only pxr inspection of local g1_29dof_rev_1_0.usd (2026-10-02)
    found identical left/right collision geometry: a box centered at
    (.035, 0, -.016), size (.15, .06, .015), plus radius-.005 spheres at
    (-.05, +/-.025, -.03), (.12, +/-.03, -.03), and (.14, 0, -.028).
    Their full XY AABB is .20 by .07 m: x=[-.055,.145], y=[-.035,.035].
    Sampling that conservative envelope at z=-.035 checks projected support;
    it does NOT model the curved sphere contacts, actual pressure distribution,
    or the toe's higher bottom z=-.033. A finite grid is not a proof that every
    point of the continuous foot outline is supported.
    Callers may instead pass any polygon-sampled [P,3] points to the transform.
    """
    if min(counts) < 1 or x_bounds[0] > x_bounds[1] or y_bounds[0] > y_bounds[1]:
        raise ValueError('invalid sole sampling bounds/counts')
    x, y = torch.meshgrid(torch.linspace(*x_bounds, counts[0], device=device, dtype=dtype),
                          torch.linspace(*y_bounds, counts[1], device=device, dtype=dtype), indexing='ij')
    return torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), sole_z)), -1)


def transform_sole_points(foot_pos, foot_quat, local_points=None):
    """Apply full foot rotation/translation; normalized quaternion order is wxyz."""
    if local_points is None:
        local_points = g1_sole_points(foot_pos.device, foot_pos.dtype)
    p = local_points.expand(*foot_pos.shape[:-1], -1, -1)
    q = foot_quat[..., None, :]
    v = q[..., 1:].expand_as(p)
    t = 2 * torch.cross(v, p, dim=-1)
    return foot_pos[..., None, :] + p + q[..., :1]*t + torch.cross(v, t, dim=-1)


def foothold_support(sole_points, surface_heights, contact_mask, tolerance=.025, surface_valid=None):
    """Return (supported [N,F,P], unsupported-contact count [N]).

    Heights and sole z share the same world frame. Unsupported includes missing
    returns and surfaces above/below the sole beyond tolerance; swing feet do
    not incur the penalty. This avoids an absolute global-height threshold.
    """
    if tolerance < 0:
        raise ValueError('tolerance must be nonnegative')
    valid = torch.isfinite(surface_heights) & torch.isfinite(sole_points).all(-1)
    if surface_valid is not None:
        valid &= surface_valid.bool()
    supported = valid & ((sole_points[..., 2] - surface_heights).abs() <= tolerance)
    penalty = ((~supported) & contact_mask.bool()[..., None]).sum(dim=(-1, -2))
    return supported, penalty
