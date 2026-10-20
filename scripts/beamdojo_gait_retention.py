"""Opt-in, soft 11700 gait retention for the existing dual-critic PPO.

Attach immediately after loading the 11700 actor. The frozen teacher is an
independent copy, including observation normalization, mean tails and Gaussian
exploration settings. Neither the actor architecture nor its state_dict changes.

The official policy group contains termwise five-frame histories (480 values).
Commands occupy [30, 45), oldest first, so the *current raw* command is [42, 45).
Retention applies to straight forward commands only. Stand, reverse and turn
samples have zero weight; every student parameter remains trainable.

There is no auxiliary-loss callback in the project's PPO. During update only,
an actor hook captures each shuffled minibatch's mean and raw TensorDict. After
the native optimizer clears gradients, the auxiliary loss backpropagates with
retain_graph=True. Native PPO then adds its gradients, clips their sum and takes
its existing optimizer step. This also preserves the bounded-update path and
its mean-overflow loss/std projection, without copying or modifying PPO code.

Integration::

    retention = attach_algorithm(
        runner.alg, joint_names=joint_names, teacher_sha256=source_sha256,
        action_scales=(upper - lower) / 2,
    )
    # runner.alg.update() now also returns gait_retention_* loss metrics.
    retention.metrics()       # latest update metrics
    retention.metadata()      # save with the experiment contract/checkpoint infos

The caller must load the declared source checkpoint before attaching. Retention
metadata is external to the unchanged native actor/PPO checkpoint schema; on
resume, supply an independently loaded 11700 teacher_actor instead of copying
the resumed student.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GaitRetentionConfig:
    coefficient: float = .01
    min_forward_speed: float = .3
    max_forward_speed: float = 1.
    max_lateral_speed: float = .05
    max_yaw_speed: float = .05
    policy_group: str = "policy"
    policy_dimension: int = 480
    command_start: int = 42
    action_dimension: int = 29
    arm_weight: float = 1.5

    def __post_init__(self):
        values = (self.coefficient, self.min_forward_speed, self.max_forward_speed,
                  self.max_lateral_speed, self.max_yaw_speed, self.arm_weight)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Gait retention settings must be finite")
        if (self.coefficient < 0 or self.min_forward_speed <= 0
                or self.max_forward_speed < self.min_forward_speed
                or self.max_lateral_speed < 0 or self.max_yaw_speed <= 0
                or self.arm_weight <= 0 or self.action_dimension != 29
                or self.policy_dimension != 480 or self.command_start != 42):
            raise ValueError("Expected the official policy480/action29 gait retention contract")


def current_commands(observations, config: GaitRetentionConfig | None = None):
    """Read current body-frame vx/vy/yaw before either actor normalizes inputs."""
    config = config or GaitRetentionConfig()
    policy = observations[config.policy_group]
    if policy.ndim != 2 or policy.shape[-1] != config.policy_dimension:
        raise ValueError("Gait retention requires raw [batch, 480] policy history")
    command = policy[:, config.command_start:config.command_start + 3]
    if not torch.isfinite(command).all():
        raise ValueError("Nonfinite gait retention command")
    return command


def command_mask(observations, config: GaitRetentionConfig | None = None):
    """Known good 11700 forward gait; zero commands never receive an anchor."""
    config = config or GaitRetentionConfig()
    command = current_commands(observations, config)
    return ((command[:, 0] >= config.min_forward_speed)
            & (command[:, 0] <= config.max_forward_speed)
            & (command[:, 1].abs() <= config.max_lateral_speed)
            & (command[:, 2].abs() < config.max_yaw_speed))


def joint_weights(joint_names: Sequence[str], arm_weight: float = 1.5):
    """Use actual action order, with soft arms 1.5 and legs/waist 1.0 weights."""
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("Expected 29 unique joint names in native action order")
    return [float(arm_weight) if any(term in name for term in
            ("shoulder_", "elbow_", "wrist_")) else 1. for name in joint_names]


def frozen_actor_copy(actor: nn.Module) -> nn.Module:
    """Copy all model state while discarding transient autograd distribution caches.

    A cached Normal and last_raw_mean can contain non-leaf tensors which PyTorch
    refuses to deepcopy. A copy memo omits those caches without mutating the
    student's active distribution. Class-based tail/exploration behavior and
    independently copied transform buffers remain intact.
    """
    distribution = getattr(actor, "distribution", None)
    memo = {}
    if distribution is not None:
        for name in ("_distribution", "last_raw_mean"):
            cached = getattr(distribution, name, None)
            if cached is not None:
                memo[id(cached)] = None
    teacher = deepcopy(actor, memo)
    teacher.requires_grad_(False)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.grad = None
    return teacher


def _independent_model_state(student, teacher):
    if student is teacher:
        raise ValueError("Teacher must be independent of the trainable student")
    student_storage = {value.untyped_storage().data_ptr() for value in
                       (*student.parameters(), *student.buffers()) if value.numel()}
    if any(value.numel() and value.untyped_storage().data_ptr() in student_storage
           for value in (*teacher.parameters(), *teacher.buffers())):
        raise ValueError("Teacher parameters/normalizers must not share student storage")


class GaitRetention:
    """Frozen teacher plus differentiable, per-joint normalized mean loss."""

    def __init__(self, student, teacher, *, joint_names, action_scales=None,
                 config=None, teacher_sha256=None, teacher_iteration=11700):
        self.config = config or GaitRetentionConfig()
        _independent_model_state(student, teacher)
        if getattr(student, "is_recurrent", False) or getattr(teacher, "is_recurrent", False):
            raise ValueError("Gait retention supports feedforward actors only")
        if list(student.obs_groups) != list(teacher.obs_groups) or student.obs_dim != teacher.obs_dim:
            raise ValueError("Teacher and student actor observation inputs must match")
        for model in (student, teacher):
            if getattr(model.distribution, "output_dim", None) != self.config.action_dimension:
                raise ValueError("Gait retention requires matching 29-action actors")
        if teacher_iteration != 11700:
            raise ValueError("This retention experiment requires the frozen 11700 teacher")
        if teacher_sha256 is not None and (len(teacher_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in teacher_sha256.lower())):
            raise ValueError("Teacher SHA256 must be a 64-character hexadecimal digest")
        parameter = next(student.parameters())
        self.weights = torch.as_tensor(joint_weights(joint_names, self.config.arm_weight),
                                      device=parameter.device, dtype=parameter.dtype)
        self.scales = torch.as_tensor(
            torch.ones(29) if action_scales is None else action_scales,
            device=parameter.device, dtype=parameter.dtype).detach().clone()
        if (self.scales.shape != (29,) or not torch.isfinite(self.scales).all()
                or (self.scales <= 0).any()):
            raise ValueError("Action scales must be 29 finite positive values")
        self.teacher = teacher.to(device=parameter.device, dtype=parameter.dtype)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        for teacher_parameter in self.teacher.parameters():
            teacher_parameter.grad = None
        self.teacher_sha256 = teacher_sha256
        self.joint_names = list(joint_names)
        self._latest_metrics: dict[str, float] = {}
        self._algorithm = None
        self._restore_update = None

    def loss(self, observations, student_mean):
        """Use the current PPO distribution mean; never sample teacher actions.

        Reduce over the *whole minibatch*, so sparse eligibility weakens the
        anchor naturally instead of amplifying a few eligible samples. Joint
        weights are normalized by their sum. Scales use native action units.
        """
        eligible = command_mask(observations, self.config)
        if student_mean.shape != (eligible.numel(), 29):
            raise ValueError("Expected one 29-action student mean per PPO sample")
        self.teacher.eval()
        with torch.no_grad():
            teacher_mean = self.teacher(observations, stochastic_output=False)
        if teacher_mean.shape != student_mean.shape or not torch.isfinite(teacher_mean).all():
            raise ValueError("Invalid frozen teacher action mean")
        delta = student_mean - teacher_mean
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite gait retention mean deviation")
        per_sample = ((delta / self.scales).square() * self.weights).sum(-1) / self.weights.sum()
        loss = self.config.coefficient * (per_sample * eligible.to(per_sample.dtype)).mean()
        selected_delta = delta.detach().abs() * eligible[:, None]
        metrics = {
            "gait_retention_loss": float(loss.detach()),
            "gait_retention_mask_fraction": float(eligible.float().mean()),
            "gait_retention_max_mean_delta": float(selected_delta.max()),
        }
        return loss, metrics

    def metrics(self):
        return dict(self._latest_metrics)

    def metadata(self):
        return dict(version=1, kind="frozen_11700_action_mean_retention",
                    teacher_iteration=11700, teacher_sha256=self.teacher_sha256,
                    config=asdict(self.config), joint_names=list(self.joint_names),
                    action_scales=self.scales.cpu().tolist(), joint_weights=self.weights.cpu().tolist(),
                    teacher_normalization="independent frozen source statistics",
                    mean="native deterministic/distribution mean including existing mean tails",
                    normalization="joint-weighted normalized squared deviation, whole minibatch mean",
                    scope="all states with straight positive vx in [0.3, 1.0]; zero stand/reverse/turn weight",
                    student_parameters="all trainable; no body-part freezing",
                    integration="auxiliary backward after native zero_grad, before PPO backward and gradient clipping")

    def _update(self, original_update):
        algorithm = self._algorithm
        pending = []
        recorded = []
        optimizer = algorithm.optimizer
        original_zero_grad = optimizer.zero_grad
        had_instance_zero_grad = "zero_grad" in vars(optimizer)
        previous_instance_zero_grad = vars(optimizer).get("zero_grad")

        def capture(module, args, kwargs, output):
            # Native feedforward PPO passes this flag by keyword for minibatches.
            if kwargs.get("stochastic_output", False) and torch.is_grad_enabled():
                if pending:
                    raise RuntimeError("Multiple actor minibatches before optimizer zero_grad")
                observations = args[0] if args else kwargs["obs"]
                pending.append(self.loss(observations, module.output_distribution_params[0]))

        def zero_grad(*args, **kwargs):
            result = original_zero_grad(*args, **kwargs)
            if len(pending) != 1:
                raise RuntimeError("Expected exactly one sampled PPO actor batch before zero_grad")
            loss, metrics = pending.pop()
            if self.config.coefficient > 0:
                if not loss.requires_grad:
                    raise RuntimeError("Gait retention lost the student actor gradient")
                loss.backward(retain_graph=True)
            recorded.append(metrics)
            return result

        hook = algorithm.actor.register_forward_hook(capture, with_kwargs=True)
        optimizer.zero_grad = zero_grad
        try:
            result = original_update()
            expected = algorithm.num_learning_epochs * algorithm.num_mini_batches
            if pending or len(recorded) != expected:
                raise RuntimeError("Gait retention minibatch count differs from native PPO")
        finally:
            hook.remove()
            if had_instance_zero_grad:
                optimizer.zero_grad = previous_instance_zero_grad
            else:
                del optimizer.zero_grad
        self._latest_metrics = {
            "gait_retention_loss": sum(x["gait_retention_loss"] for x in recorded) / len(recorded),
            "gait_retention_mask_fraction": sum(x["gait_retention_mask_fraction"] for x in recorded) / len(recorded),
            "gait_retention_max_mean_delta": max(x["gait_retention_max_mean_delta"] for x in recorded),
        }
        return {**result, **self._latest_metrics}

    def remove(self):
        """Restore the original algorithm update; actor and optimizer are unchanged."""
        if self._algorithm is not None:
            self._restore_update()
            del self._algorithm._beamdojo_gait_retention
            self._algorithm = None


def attach_algorithm(algorithm, *, joint_names, action_scales=None, config=None,
                     teacher_actor=None, teacher_sha256=None, teacher_iteration=11700):
    """Attach after loading 11700; pass a separate source teacher on later resume."""
    if getattr(algorithm, "_beamdojo_gait_retention", None) is not None:
        raise ValueError("Gait retention is already attached")
    if (getattr(algorithm, "rnd", None) or getattr(algorithm, "symmetry", None)
            or getattr(algorithm, "is_multi_gpu", False)):
        raise ValueError("Gait retention supports single-device PPO without RND or symmetry")
    student = getattr(algorithm, "_raw_actor", algorithm.actor)
    teacher = frozen_actor_copy(student if teacher_actor is None else teacher_actor)
    retention = GaitRetention(student, teacher, joint_names=joint_names,
                              action_scales=action_scales, config=config,
                              teacher_sha256=teacher_sha256, teacher_iteration=teacher_iteration)
    original_update = algorithm.update
    had_instance_update = "update" in vars(algorithm)
    previous_instance_update = vars(algorithm).get("update")

    def restore_update():
        if had_instance_update:
            algorithm.update = previous_instance_update
        else:
            del algorithm.update

    retention._algorithm = algorithm
    retention._restore_update = restore_update
    algorithm.update = lambda: retention._update(original_update)
    algorithm._beamdojo_gait_retention = retention
    return retention


attach_gait_retention = attach_algorithm
