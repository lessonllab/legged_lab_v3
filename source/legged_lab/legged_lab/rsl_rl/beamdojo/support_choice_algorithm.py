"""Native dual-critic PPO factory with correctly factored latent likelihoods."""
from copy import deepcopy
import math
import torch

from rsl_rl.utils import resolve_callable, resolve_obs_groups
from .dual_critic import DualCriticPPO, DualCriticStorage, IndependentCritics
from .support_choice_actor import SupportChoiceActor


class SupportChoicePPO(DualCriticPPO):
    def __init__(self, actor, *args, **kwargs):
        if not isinstance(actor, SupportChoiceActor) or actor.get_extra_state().get('version') != 2:
            raise ValueError('Support-choice PPO requires the bounded v2 actor')
        super().__init__(actor, *args, **kwargs)
        self.latent_std_projection_count = 0
        self._latent_std_post_hook = self.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: self.project_latent_std())
        self.project_latent_std()

    @torch.no_grad()
    def project_latent_std(self):
        parameter = self._raw_actor.distribution.log_std_param
        state = self.optimizer.state.get(parameter, {})
        # Validate before mutation: clamping must never hide NaN/Inf incidents.
        if not torch.isfinite(parameter).all():
            raise ValueError('Nonfinite latent log std before projection')
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite latent Adam state: '+key)
        lower, upper = (math.log(x) for x in self._raw_actor.get_extra_state()['std_range'])
        projected = (parameter < lower) | (parameter > upper)
        if projected.any():
            parameter.clamp_(lower, upper)
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if key in state:
                    if state[key].shape != parameter.shape:
                        raise ValueError('Unexpected latent Adam slot shape: '+key)
                    state[key][projected] = 0.
            self.latent_std_projection_count += int(projected.sum())
        return projected

    def load(self, loaded_dict, load_cfg=None, strict=True):
        # Reject the failed v1 before loading any model or optimizer state.
        metadata = loaded_dict.get('actor_state_dict', {}).get('_extra_state', {})
        if metadata != self._raw_actor.get_extra_state():
            raise ValueError('Support-choice v2 resume contract mismatch')
        result = super().load(loaded_dict, load_cfg, strict)
        self.project_latent_std()
        return result

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        cfg['algorithm'].setdefault('rnd_cfg', None)
        options = deepcopy(cfg['algorithm'])
        options.pop('class_name')
        if options.pop('share_cnn_encoders', False):
            raise ValueError('Shared critic encoders are unsupported')
        if env.num_actions != 4 or (env.num_envs*cfg['num_steps_per_env']) % options.get('num_mini_batches', 4):
            raise ValueError('Expected four latents and a divisible rollout')
        groups = resolve_obs_groups(obs, cfg.get('obs_groups', {}), ['actor', 'critic'])
        cfg['obs_groups'] = groups
        actor = SupportChoiceActor(env.base_teacher,
            teacher_sha256=env.source_sha256, mapping_metadata=env.mapping_metadata,
            attention=env.attention_enabled,
            predictor=env.predictor if env.attention_enabled else None)

        def critic():
            settings = deepcopy(cfg['critic'])
            cls = resolve_callable(settings.pop('class_name'))
            return cls(obs, groups, 'critic', 1, **settings)

        critics = IndependentCritics(critic(), critic())
        critics.load_state_dict(env.source_critic_state)
        storage = DualCriticStorage(env.num_envs, cfg['num_steps_per_env'], obs, [4], device)
        algorithm = SupportChoicePPO(actor, critics, storage, device=device,
                                     multi_gpu_cfg=cfg.get('multi_gpu'), **options)
        # No parent 29D bounds/tail is installed on the four-dimensional latent.
        # The teacher retains its complete tail and normalization internally.
        algorithm.compile(cfg.get('torch_compile_mode'))
        env.support_actor = algorithm._raw_actor
        return algorithm
