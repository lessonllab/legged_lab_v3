"""Original RSL-RL 5.4.1 adapter for BeamDojo's separated advantage objective.

Use the unmodified rsl_rl.runners.OnPolicyRunner with algorithm.class_name set
to ``legged_lab.rsl_rl.beamdojo:DualCriticPPO`` and native ``actor``/``critic``
model configs. Both critics have separate parameters and observation statistics.
The environment must supply dense and sparse rewards *after reward scaling* in
extras; their sum must match the scalar reward before timeout bootstrapping.

Scope: feedforward, single-device PPO; no RND, symmetry, AMP, recurrent models or
shared CNN encoders. Timeout correction follows RSL 5.4.1's current-value
convention, independently for both critics (never the reset observation).
Checkpoints resume training between rollouts, not the simulator/RNG trajectory.
"""

from copy import deepcopy

import torch
from torch import nn

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from .numerical_guard import NumericalGuard
from .action_distribution_bounds import ActionDistributionBounds, bounded_update
from .action_mean_tail_bounds import install_mean_tail, remove_mean_tail
from . import exploration_scale


def separated_gae(rewards, values, dones, last_values, gamma, lam, weights=(1.0, 0.25)):
    """Return two value targets and one weighted, independently normalized advantage.

    rewards/values: [time, env, 2]; dones: [time, env, 1]; last_values: [env, 2].
    Timeout rewards must already include gamma * V_i for the respective critic.
    Sample standard deviation matches the reference; a singleton is finite.
    """
    if rewards.shape != values.shape or values.shape[-1] != 2:
        raise ValueError("Expected two reward/value streams with matching shapes")
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_values)
    for step in range(values.shape[0] - 1, -1, -1):
        following = last_values if step == values.shape[0] - 1 else values[step + 1]
        continuing = 1.0 - dones[step].to(values.dtype)
        delta = rewards[step] + gamma * continuing * following - values[step]
        advantage = delta + gamma * lam * continuing * advantage
        returns[step] = advantage + values[step]
    raw = returns - values
    flat = raw.flatten(0, 1)
    std = flat.std(dim=0, correction=1 if flat.shape[0] > 1 else 0)
    normalized = (raw - flat.mean(dim=0)) / (std + 1e-8)
    combined = (normalized * values.new_tensor(weights)).sum(dim=-1, keepdim=True)
    return returns, combined


class IndependentCritics(nn.Module):
    """Two independently constructed native models presented as a two-value model."""

    is_recurrent = False

    def __init__(self, dense, sparse):
        super().__init__()
        if dense.is_recurrent or sparse.is_recurrent:
            raise ValueError("BeamDojo adapter supports feedforward critics only")
        if {id(p) for p in dense.parameters()} & {id(p) for p in sparse.parameters()}:
            raise ValueError("Critics must not share parameters")
        self.dense = dense
        self.sparse = sparse

    def forward(self, obs, **kwargs):
        return torch.cat((self.dense(obs, **kwargs), self.sparse(obs, **kwargs)), dim=-1)

    def update_normalization(self, obs):
        self.dense.update_normalization(obs)
        self.sparse.update_normalization(obs)

    def get_hidden_state(self):
        return None

    def reset(self, dones=None, hidden_state=None):
        self.dense.reset(dones)
        self.sparse.reset(dones)


class DualCriticStorage(RolloutStorage):
    """Retain native PPO minibatches with two values and a scalar policy advantage."""

    def __init__(self, num_envs, num_steps, obs, actions_shape, device="cpu"):
        super().__init__("rl", num_envs, num_steps, obs, actions_shape, device)
        shape = (num_steps, num_envs, 2)
        self.values = torch.zeros(shape, device=device)
        self.returns = torch.zeros_like(self.values)
        self.component_rewards = torch.zeros_like(self.values)


class DualCriticPPO(PPO):
    """Native PPO optimization with two returns and BeamDojo's policy advantage."""

    def __init__(self, actor, critic, storage, advantage_weight_dense=1.0,
                 advantage_weight_sparse=0.25, value_loss_coef=1.0, **kwargs):
        for name in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg"):
            if kwargs.get(name):
                raise ValueError(f"{name} is unsupported by the BeamDojo adapter")
        if kwargs.get("normalize_advantage_per_mini_batch", False):
            raise ValueError("BeamDojo requires independent whole-rollout advantage normalization")
        if actor.is_recurrent or not isinstance(critic, IndependentCritics):
            raise ValueError("Use a feedforward actor and IndependentCritics")
        if not isinstance(storage, DualCriticStorage):
            raise ValueError("Use DualCriticStorage")
        # Native PPO averages over the two output dimensions. Multiplication by
        # two yields L_dense + L_sparse, not half the intended value objective.
        super().__init__(actor, critic, storage, value_loss_coef=2 * value_loss_coef, **kwargs)
        self.advantage_weights = (float(advantage_weight_dense), float(advantage_weight_sparse))
        self._numerical_guard = None
        self._numerical_guard_step = 0
        self._action_distribution_bounds = None

    def set_action_distribution_bounds(self, lower, upper, mean_penalty_coef=.01):
        """Explicit PPO migration; returns the initial std/Adam projection report."""
        self._action_distribution_bounds = ActionDistributionBounds(self, lower, upper, mean_penalty_coef)
        return self._action_distribution_bounds.project_std(self)

    def set_action_mean_tail_bounds(self, lower, upper, identity_margin_halfwidth=.5, tail_width_halfwidth=.5):
        """Explicit mean-distribution migration; no parameter or Adam mutation."""
        if self._action_distribution_bounds is None:
            raise ValueError('Mean tails require existing per-axis standard deviation bounds')
        for actual, expected in ((lower, self._action_distribution_bounds.lower),
                                 (upper, self._action_distribution_bounds.upper)):
            value = torch.as_tensor(actual, device=expected.device, dtype=expected.dtype)
            if not torch.equal(value, expected):
                raise ValueError('Mean bounds must match existing distribution/execution bounds')
        distribution = self._raw_actor.distribution
        scale = self.action_exploration_scale_report()
        if scale is not None:
            exploration_scale.remove(distribution)
        try:
            return install_mean_tail(distribution, lower, upper,
                                     identity_margin_halfwidth, tail_width_halfwidth)
        finally:
            if scale is not None:
                exploration_scale.restore(distribution, scale)

    def set_action_exploration_scale(self, factor=.25):
        """Explicit on-policy std scaling; preserves parameters and Adam slots."""
        return exploration_scale.install(self._raw_actor.distribution, factor)

    def action_exploration_scale_report(self):
        return exploration_scale.metadata(self._raw_actor.distribution)

    def action_mean_tail_bounds_report(self):
        transform = getattr(self._raw_actor.distribution, 'mean_tail', None)
        return None if transform is None else transform.metadata()

    def action_raw_mean(self):
        """Latest pre-transform MLP output, for diagnostics and original mean loss."""
        return getattr(self._raw_actor.distribution, 'last_raw_mean', None)

    def action_distribution_bounds_report(self):
        bounds = self._action_distribution_bounds
        return None if bounds is None else bounds.metadata()

    def set_numerical_guard(self, path, reward_abs_max=1000., action_abs_max=100.,
                            observation_abs_max=1e6, value_abs_max=1e4, returns_abs_max=1e5):
        """Enable a runtime-only guard; no objective/checkpoint parameters change."""
        self._numerical_guard = NumericalGuard(path, reward_abs_max=reward_abs_max,
            action_abs_max=action_abs_max, observation_abs_max=observation_abs_max,
            value_abs_max=value_abs_max, returns_abs_max=returns_abs_max)

    def _guard_context(self, obs):
        return {'observations': obs, 'raw_mlp_mean': self.action_raw_mean(), 'transition': vars(self.transition),
                'storage_step': self.storage.step}

    def act(self, obs):
        guard = self._numerical_guard
        if guard is None:
            return super().act(obs)
        kw = dict(stage='act_before', step=self._numerical_guard_step, context=self._guard_context(obs))
        guard.check_observations(obs, **kw)
        try:
            actions = super().act(obs)
        except Exception as error:
            guard.capture_exception(error, stage='act_native', step=self._numerical_guard_step,
                                    context=self._guard_context(obs))
            raise
        kw.update(stage='act_after', context=self._guard_context(obs))
        raw_mean = self.action_raw_mean()
        if raw_mean is not None:
            guard.check_finite('actor.raw_mlp_mean', raw_mean, stage='act_after', step=self._numerical_guard_step)
        for field, kind in (('actions', 'action'), ('values', 'value'), ('actions_log_prob', 'value')):
            guard.check('transition.' + field, getattr(self.transition, field), kind, **kw)
        for index, parameter in enumerate(self.transition.distribution_params):
            guard.check('transition.distribution_' + ('mean' if index == 0 else 'std'),
                        parameter, 'action', positive=index == 1, **kw)
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if "rewards_dense" not in extras or "rewards_sparse" not in extras:
            raise KeyError("BeamDojo requires extras['rewards_dense'] and extras['rewards_sparse']")
        components = torch.stack([
            extras[key].to(self.device).reshape(-1)
            for key in ("rewards_dense", "rewards_sparse")
        ], dim=-1)
        if components.shape != (self.storage.num_envs, 2):
            raise ValueError("Reward components must each contain one scalar per environment")
        guard = self._numerical_guard
        if guard is not None:
            kw = dict(stage='process_env_step', step=self._numerical_guard_step,
                      context={**self._guard_context(obs), 'rewards': rewards, 'components': components})
            guard.check_observations(obs, **kw)
            guard.check('rewards', rewards, 'reward', **kw)
            guard.check('reward_components', components, 'reward', **kw)
        if not torch.allclose(components.sum(-1), rewards.reshape(-1), rtol=1e-4, atol=1e-6):
            raise ValueError("Dense plus sparse rewards must equal total reward")
        if "time_outs" in extras:
            timeout = extras["time_outs"].to(self.device).reshape(-1, 1)
            components = components + self.gamma * self.transition.values * timeout
        if guard is not None:
            guard.check('bootstrapped_components', components, 'returns', **kw)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.storage.step >= self.storage.num_transitions_per_env:
            raise RuntimeError("Rollout is full; compute returns and update before collecting more")
        self.storage.component_rewards[self.storage.step].copy_(components)
        self.transition.rewards = components.sum(dim=-1)
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
        self._numerical_guard_step += 1

    def compute_returns(self, obs):
        if self.storage.step != self.storage.num_transitions_per_env:
            raise RuntimeError("Compute returns only after collecting a complete rollout")
        with torch.no_grad():
            guard = self._numerical_guard
            if guard is not None:
                kw = dict(stage='compute_returns', step=self._numerical_guard_step, context=self._guard_context(obs))
                guard.check_observations(obs, **kw)
            last_values = self.critic(obs)
            if guard is not None:
                guard.check('last_values', last_values, 'value', **kw)
            returns, advantages = separated_gae(
                self.storage.component_rewards, self.storage.values, self.storage.dones,
                last_values, self.gamma, self.lam, self.advantage_weights,
            )
            if guard is not None:
                guard.check('returns', returns, 'returns', env_axis=1, **kw)
                guard.check('advantages', advantages, 'returns', env_axis=1, **kw)
            self.storage.returns.copy_(returns)
            self.storage.advantages.copy_(advantages)

    def _guard_model_optimizer(self, stage):
        guard = self._numerical_guard
        if guard is None:
            return
        guard._assert_active()
        kw = dict(stage=stage, step=self._numerical_guard_step)
        names = {}
        for prefix, model in (('actor', self._raw_actor), ('critic', self._raw_critic)):
            # Registered distribution parameters (including learned std) belong
            # to the raw actor, even when inference models are compiled.
            for name, parameter in model.named_parameters():
                field = prefix + '.' + name
                names[id(parameter)] = field
                guard.check_finite(field, parameter, **kw)
        for index, (parameter, state) in enumerate(self.optimizer.state.items()):
            name = names.get(id(parameter), 'parameter_' + str(index))
            for slot, value in state.items():
                if isinstance(value, torch.Tensor):
                    guard.check_finite('optimizer.' + name + '.' + str(slot), value, **kw)

    def update(self):
        if self._numerical_guard is not None:
            self._numerical_guard._assert_active()
        result = super().update() if self._action_distribution_bounds is None else bounded_update(self)
        self._guard_model_optimizer('update_after')
        return result

    def save(self):
        self._guard_model_optimizer('save_before')
        saved = super().save()
        saved["beamdojo"] = {
            "version": 1,
            "advantage_weights": self.advantage_weights,
            "learning_rate": self.learning_rate,
            # Restore objective/schedule as well as model, normalization and Adam state.
            "parameters": {name: getattr(self, name) for name in (
                "gamma", "lam", "clip_param", "value_loss_coef", "entropy_coef",
                "max_grad_norm", "desired_kl", "schedule", "num_learning_epochs", "num_mini_batches",
                "use_clipped_value_loss",
            )},
        }
        if self._action_distribution_bounds is not None:
            saved['beamdojo']['action_distribution_bounds'] = self.action_distribution_bounds_report()
        if self.action_mean_tail_bounds_report() is not None:
            saved['beamdojo']['action_mean_tail_bounds'] = self.action_mean_tail_bounds_report()
        if self.action_exploration_scale_report() is not None:
            saved['beamdojo']['action_exploration_scale'] = self.action_exploration_scale_report()
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        full_resume = load_cfg is None or load_cfg.get("optimizer", False)
        if full_resume and loaded_dict.get("beamdojo", {}).get("version") != 1:
            raise ValueError("Full resume requires a BeamDojo dual-critic checkpoint")
        # No-op for ordinary models: do not invalidate their native cached Normal.
        if self.action_exploration_scale_report() is not None:
            exploration_scale.remove(self._raw_actor.distribution)
        resume_iteration = super().load(loaded_dict, load_cfg, strict)
        if full_resume:
            state = loaded_dict["beamdojo"]
            self.advantage_weights = tuple(state["advantage_weights"])
            self.learning_rate = state["learning_rate"]
            for name, value in state["parameters"].items():
                setattr(self, name, value)
        bounds = loaded_dict.get('beamdojo', {}).get('action_distribution_bounds')
        if bounds is not None:
            if bounds.get('version') != 1:
                raise ValueError('Unsupported action distribution bounds metadata')
            self.set_action_distribution_bounds(bounds['lower'], bounds['upper'], bounds['mean_penalty_coef'])
            self._action_distribution_bounds.projection_count += bounds.get('projection_count', 0)
            if self._action_distribution_bounds.last_projection is None:
                self._action_distribution_bounds.last_projection = bounds.get('last_projection')
        mean_tail = loaded_dict.get('beamdojo', {}).get('action_mean_tail_bounds')
        if mean_tail is None:
            remove_mean_tail(self._raw_actor.distribution)
        else:
            if (mean_tail.get('version') != 1 or mean_tail.get('kind') != 'identity_interior_rational_gaussian_mean'
                    or mean_tail.get('raw_mlp_penalty') is not True
                    or mean_tail.get('execution_clamp_unchanged') is not True):
                raise ValueError('Unsupported action mean-tail metadata')
            self.set_action_mean_tail_bounds(mean_tail['lower'], mean_tail['upper'],
                mean_tail['identity_margin_halfwidth'], mean_tail['tail_width_halfwidth'])
        scale = loaded_dict.get('beamdojo', {}).get('action_exploration_scale')
        if scale is not None:
            exploration_scale.restore(self._raw_actor.distribution, scale)
        return resume_iteration

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        """Called directly by RSL-RL 5.4.1's unmodified OnPolicyRunner."""
        cfg["algorithm"].setdefault("rnd_cfg", None)  # Native runner accesses this key.
        algorithm_cfg = deepcopy(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        if algorithm_cfg.pop("share_cnn_encoders", False):
            raise ValueError("Shared encoders are unsupported; critics must remain independent")
        groups = resolve_obs_groups(obs, cfg.get("obs_groups", {}), ["actor", "critic"])
        cfg["obs_groups"] = groups

        def model(config, group, output_dim):
            settings = deepcopy(config)
            model_class = resolve_callable(settings.pop("class_name"))
            return model_class(obs, groups, group, output_dim, **settings)

        actor = model(cfg["actor"], "actor", env.num_actions)
        critic = IndependentCritics(model(cfg["critic"], "critic", 1), model(cfg["critic"], "critic", 1))
        storage = DualCriticStorage(env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        size = env.num_envs * cfg["num_steps_per_env"]
        batches = algorithm_cfg.get("num_mini_batches", 4)
        if batches < 1 or size % batches:
            raise ValueError("Rollout size must be divisible by num_mini_batches")
        algorithm = DualCriticPPO(actor, critic, storage, device=device,
                                  multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg)
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm
