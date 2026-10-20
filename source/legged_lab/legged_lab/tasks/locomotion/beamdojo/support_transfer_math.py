"""Short-horizon centroidal support-transfer *proxy*, with no simulator dependency.

Only measured, actually supported sole samples may be passed as valid contacts.
Each sample supplies its own force: no convex-hull fill, gap contact, or future
contact is invented. Stages independently fit a constant centroidal wrench using
current COM lever arms (a quasi-static lever linearization). The old foot's
normal capacity fades to zero. This is not trajectory feasibility, an exact QP,
a stability certificate, or a guarantee of natural gait. The caller must gate
use on legitimate contact events.

The fixed-budget solver takes gradient steps, projects each force exactly onto
a circular Coulomb friction cone, then scales foot/total normal forces into capacity.
This composition is a feasible retraction, not the Euclidean projection onto
the joint constraint set. A positive residual may reflect limited convergence.
All computation is deterministic float32 and batched on the input device.

Objective bounds (for this fixed-lever convex proxy only): write F*=min_C
||Af-b||^2. The identity ||z||^2 >= 2*y.dot(z)-||y||^2 gives, for every y,
F* >= D(y) = -||y||^2 - 2*b.dot(y) - 2*h_C(-A.T@y). We use the returned
feasible force's residual for y. Its squared residual is a primal upper bound.
For v=-A.T@y, a contact with normal t contributes at most
(v_z + mu*||v_xy||)*t to the support function. Within each foot all normal
load can use the highest-scoring valid sample; zero force is also allowed.
The two foot maxima are sorted, and normal load is allocated to the higher
score first subject to each foot's cap and the shared total cap. This computes
h_C exactly in real arithmetic without constructing contacts between samples.
Clamping D below at zero remains a lower bound since F* is nonnegative. A
small roundoff allowance is subtracted before that clamp in float32.
Consequently positive(strict dual cost - held primal cost) certifies an
objective gap for this proxy, up to floating-point error, rather than rewarding
extra solver convergence. It is not a physical stability/trajectory certificate.
"""
from __future__ import annotations

import math
import torch


def _cone_project(force: torch.Tensor, mu: float) -> torch.Tensor:
    """Euclidean projection onto z>=0, norm([x,y])<=mu*z."""
    xy, z = force[..., :2], force[..., 2]
    if mu == 0:
        return torch.cat((torch.zeros_like(xy), z.clamp_min(0).unsqueeze(-1)), -1)
    radius = torch.linalg.vector_norm(xy, dim=-1)
    height = torch.where(radius <= mu * z, z,
                         (z + mu * radius) / (1 + mu * mu)).clamp_min(0)
    scale = (mu * height / radius.clamp_min(1e-20)).clamp_max(1.0)
    return torch.cat((xy * scale.unsqueeze(-1), height.unsqueeze(-1)), -1)


@torch.no_grad()
def support_transfer_cost(
    support_points: torch.Tensor,
    support_mask: torch.Tensor,
    com: torch.Tensor,
    com_velocity: torch.Tensor,
    angular_momentum: torch.Tensor,
    mass: torch.Tensor,
    command_xy: torch.Tensor,
    new_support_foot: torch.Tensor,
    *,
    horizon: float = 0.3,
    friction: float = 0.4,
    iterations: int = 20,
    stages: int = 3,
    max_normal_ratio: float = 2.0,
    length_scale: float = 0.5,
    gravity: float = 9.81,
    angular_momentum_band_ratio: tuple[float, float, float] = (0.04, 0.04, 0.08),
    unload_old_foot: bool = True,
    initial_force_newtons: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return bounded cost and diagnostics from actual world-frame contact data.

    Shapes: points [N,2,P,3], mask [N,2,P], COM/velocity/angular momentum
    [N,3], mass [N] (or scalar), command [N,2], new foot integer [N].
    Angular momentum is TOTAL world-frame momentum about COM, in kg*m^2/s.
    Target acceleration = ([command_xy,0]-velocity)/horizon. Momentum is
    normalized by mass*g*length_scale*horizon. Its permitted terminal ellipsoid
    has semiaxes angular_momentum_band_ratio: only the radial excess outside
    this band is requested to decay over the horizon. This is an engineering
    allowance, not measured natural-gait truth. Equal horizontal semiaxes keep
    yaw rotation invariance; anisotropic semiaxes intentionally do not.
    unload_old_foot=False holds both capacities for the same-state baseline.
    Pass transition["forces"] as initial_force_newtons [N,stages,2,P,3] when
    computing that baseline. This candidate is projected into its constraints.
    Each stage returns the best objective seen, including initialization, so
    a held baseline initialized from a feasible transition cannot have higher
    residual than that transition, up to float32 roundoff. This fixes comparison
    consistency, not exact convergence or trajectory feasibility.
    A caller can penalize max(transition_cost-held_cost,0), recording both
    absolute costs, to isolate added violation from the support transition.
    Uniform duplication of all samples leaves capacities and, up to float32
    reduction error, the fixed-iteration result unchanged. Nonuniform sample
    duplication can alter convergence, though never the physical capacity.
    Force is normalized by mass*g, torque by mass*g*length_scale.
    Residual is the norm of this six-component normalized wrench error.
    Forces are returned in newtons, with shape [N,stages,2,P,3].
    objective_primal_upper / objective_dual_lower have shape [N,stages] and
    bound the stage's exact convex wrench-fit objective (when finite=True).
    Use the monotone map 1-exp(-mean_stage(objective)) for cost bounds.
    For invalid inputs both bound diagnostics are zero, not valid certificates;
    callers must consult finite and has_support.
    Nonfinite/invalid state gives finite outputs, cost=1, and finite=False.
    Empty support gives cost=1 even if the requested wrench happens to be zero.
    """
    for name, value in (("horizon", horizon), ("length_scale", length_scale),
                        ("gravity", gravity), ("max_normal_ratio", max_normal_ratio)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(friction) or friction < 0:
        raise ValueError("friction must be finite and nonnegative")
    if not isinstance(iterations, int) or not 1 <= iterations <= 128:
        raise ValueError("iterations must be an integer in [1,128]")
    if len(angular_momentum_band_ratio) != 3 or any(
        not math.isfinite(v) or v <= 0 for v in angular_momentum_band_ratio
    ):
        raise ValueError("angular_momentum_band_ratio must have three finite positive semiaxes")
    if not isinstance(unload_old_foot, bool):
        raise ValueError("unload_old_foot must be boolean")
    if not isinstance(stages, int) or stages < 2:
        raise ValueError("stages must be at least two")
    if support_points.ndim != 4 or support_points.shape[1] != 2 or support_points.shape[-1] != 3:
        raise ValueError("support_points must have shape [N,2,P,3]")
    n, _, p, _ = support_points.shape
    if p == 0 or support_mask.shape != (n, 2, p):
        raise ValueError("mask shape must match nonempty sole sample dimensions")
    for name, value, shape in (("com", com, (n, 3)), ("com_velocity", com_velocity, (n, 3)),
                               ("angular_momentum", angular_momentum, (n, 3)),
                               ("command_xy", command_xy, (n, 2)),
                               ("new_support_foot", new_support_foot, (n,))):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if initial_force_newtons is not None and initial_force_newtons.shape != (n, stages, 2, p, 3):
        raise ValueError("initial_force_newtons must have shape [N,stages,2,P,3]")
    device = support_points.device
    points, center, velocity, momentum, command = (
        value.to(device=device, dtype=torch.float32)
        for value in (support_points, com, com_velocity, angular_momentum, command_xy)
    )
    body_mass = torch.as_tensor(mass, device=device, dtype=torch.float32).expand(n)
    mask = support_mask.to(device=device, dtype=torch.bool)
    foot = new_support_foot.to(device=device)
    valid = torch.isfinite(center).all(-1) & torch.isfinite(velocity).all(-1)
    valid &= torch.isfinite(momentum).all(-1) & torch.isfinite(command).all(-1)
    valid &= torch.isfinite(body_mass) & (body_mass > 0) & ((foot == 0) | (foot == 1))
    valid &= (torch.isfinite(points).all(-1) | ~mask).all(dim=(-1, -2))
    if initial_force_newtons is not None:
        initial_force_newtons = initial_force_newtons.to(device=device, dtype=torch.float32)
        valid &= torch.isfinite(initial_force_newtons).all(dim=(-1, -2, -3, -4))
    # Sanitization affects only invalid environments, which are charged maximal cost.
    clean = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mg = clean(body_mass).clamp_min(1e-6) * gravity
    lever = clean((points - center[:, None, None]) / length_scale)
    lever = torch.where(mask[..., None], lever, 0.0)
    desired_velocity = torch.cat((command, torch.zeros_like(command[:, :1])), -1)
    target_force = clean((desired_velocity - velocity) / (horizon * gravity))
    target_force[:, 2] += 1.0
    normalized_momentum = clean(momentum / (horizon * mg[:, None] * length_scale))
    band = torch.tensor(angular_momentum_band_ratio, device=device, dtype=torch.float32)
    band_radius = torch.linalg.vector_norm(normalized_momentum / band, dim=-1)
    excess_fraction = (1.0 - 1.0 / band_radius.clamp_min(1.0)).clamp_min(0.0)
    target_torque = -normalized_momentum * excess_fraction[:, None]
    target = torch.cat((target_force, target_torque), -1)
    # Reject unphysical magnitudes before squared norms overflow float32.
    valid &= (lever.abs() < 1e4).all(dim=(-1, -2, -3)) & (target.abs() < 1e4).all(-1)
    lever = lever.clamp(-1e4, 1e4)
    target = target.clamp(-1e4, 1e4)
    mask = mask & valid[:, None, None]
    progress = torch.linspace(0, 1, stages, device=device) if unload_old_foot else torch.zeros(stages, device=device)
    new = torch.arange(2, device=device)[None, :] == foot[:, None]
    capacity = max_normal_ratio * torch.where(new[:, None, :], 1.0, 1.0 - progress[None, :, None])
    active = mask[:, None, :, :, None]

    def project(force):
        force = _cone_project(force, friction) * active
        normal = force[..., 2].sum(-1)
        scale = (capacity / normal.clamp_min(1e-12)).clamp_max(1.0)
        force = force * scale[..., None, None]
        total = force[..., 2].sum(dim=(-1, -2))
        scale = (max_normal_ratio / total.clamp_min(1e-12)).clamp_max(1.0)
        return force * scale[..., None, None, None]

    if initial_force_newtons is None:
        force = torch.zeros((n, stages, 2, p, 3), device=device)
        force[..., 2] = mask[:, None] / mask.sum(dim=(-1, -2)).clamp_min(1)[:, None, None, None]
    else:
        force = clean(initial_force_newtons / mg[:, None, None, None, None])
    force = project(force)
    r = lever[:, None].expand(-1, stages, -1, -1, -1)
    # Frobenius norm bound for the 6-by-3K wrench operator gives a safe step.
    lipschitz = ((3.0 + 2.0 * lever.square().sum(-1)) * mask).sum(dim=(-1, -2)).clamp_min(1.0)
    step = 1.0 / lipschitz[:, None, None, None, None]

    def errors(force):
        linear = force.sum(dim=(-2, -3)) - target[:, None, :3]
        angular = torch.cross(r, force, dim=-1).sum(dim=(-2, -3)) - target[:, None, 3:]
        return linear, angular

    linear, angular = errors(force)
    best_score = linear.square().sum(-1) + angular.square().sum(-1)
    best_force = force
    for _ in range(iterations):
        gradient = linear[:, :, None, None, :] + torch.cross(angular[:, :, None, None, :].expand_as(r), r, dim=-1)
        force = project(force - step * gradient)
        linear, angular = errors(force)
        score = linear.square().sum(-1) + angular.square().sum(-1)
        improved = score < best_score
        best_force = torch.where(improved[:, :, None, None, None], force, best_force)
        best_score = torch.minimum(score, best_score)
    force = best_force
    linear, angular = errors(force)
    wrench_error = torch.cat((linear, angular), -1)
    residual = torch.linalg.vector_norm(wrench_error, dim=-1)
    primal_upper = wrench_error.square().sum(-1)
    # y=residual, v=-A^T y. Per-foot support is a normal-load allocation LP.
    dual_direction = -(linear[:, :, None, None, :] + torch.cross(
        angular[:, :, None, None, :].expand_as(r), r, dim=-1))
    point_score = dual_direction[..., 2] + friction * torch.linalg.vector_norm(dual_direction[..., :2], dim=-1)
    foot_score = torch.where(mask[:, None], point_score, 0.0).amax(-1).clamp_min(0.0)
    first_is_zero = foot_score[..., 0] >= foot_score[..., 1]
    first_score = torch.where(first_is_zero, foot_score[..., 0], foot_score[..., 1])
    second_score = torch.where(first_is_zero, foot_score[..., 1], foot_score[..., 0])
    first_cap = torch.where(first_is_zero, capacity[..., 0], capacity[..., 1])
    second_cap = torch.where(first_is_zero, capacity[..., 1], capacity[..., 0])
    first_load = first_cap.clamp_max(max_normal_ratio)
    second_load = torch.minimum(second_cap, (max_normal_ratio-first_load).clamp_min(0.0))
    support_function = first_score*first_load + second_score*second_load
    target_dot_y = (target[:, None]*wrench_error).sum(-1)
    dual_raw = -primal_upper - 2*target_dot_y - 2*support_function
    # Conservative numerical margin, scaled by absolute arithmetic operands.
    roundoff = 16*torch.finfo(torch.float32).eps*(
        1.0 + primal_upper + 2*(target[:, None]*wrench_error).abs().sum(-1) + 2*support_function)
    dual_lower = (dual_raw-roundoff).clamp_min(0.0)
    physical_force = force * mg[:, None, None, None, None]
    finite = valid & torch.isfinite(residual).all(-1) & torch.isfinite(physical_force).all(dim=(-1, -2, -3, -4))
    cost = -torch.expm1(-residual.square().mean(-1))
    has_support = mask.any(dim=(-1, -2))
    cost = torch.where(finite & has_support, cost.clamp(0, 1), 1.0)
    return {
        "cost": clean(cost), "residual": clean(residual),
        "wrench_error": clean(wrench_error), "forces": clean(physical_force),
        "finite": finite, "has_support": has_support,
        "normal_capacity": clean(capacity * mg[:, None, None]),
        "target_torque": clean(target_torque),
        "objective_primal_upper": torch.where(finite[:, None], clean(primal_upper), 0.0),
        "objective_dual_lower": torch.where(finite[:, None], clean(dual_lower), 0.0),
    }
