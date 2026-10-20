"""Four-dimensional PPO policy above an immutable deterministic teacher.

The environment adapter, not PPO, decodes latents into 29 requested actions.
No 29-dimensional likelihood is attributed to the resulting degenerate policy.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from rsl_rl.modules.distribution import GaussianDistribution


class SupportChoiceActor(nn.Module):
    is_recurrent = False
    latent_dim = 4

    def __init__(self, teacher, *, teacher_sha256, mapping_metadata,
                 hidden_dims=(128, 64), init_std=.2, attention=False, predictor=None):
        super().__init__()
        if (len(teacher_sha256) != 64 or any(c not in '0123456789abcdef' for c in teacher_sha256)
                or not math.isfinite(init_std) or not .05 <= init_std <= .5):
            raise ValueError('Invalid teacher digest or latent exploration std')
        if teacher.is_recurrent or not hasattr(teacher, 'get_latent'):
            raise ValueError('A feedforward native teacher with get_latent is required')
        self.teacher = deepcopy(teacher)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.obs_groups = list(teacher.obs_groups)
        self.obs_dim = teacher.obs_dim
        self.obs_normalization = teacher.obs_normalization
        if type(attention) is not bool:
            raise ValueError('attention must be an explicit boolean')
        self.attention = None
        self.predictor = None
        self.last_attention_weights = None
        if attention:
            from .support_choice_attention import SupportChoiceAttention
            if self.obs_dim != 708 or self.obs_groups != ['policy', 'terrain', 'course_tracking']:
                raise ValueError('Attention requires canonical 480+225+3 observation groups')
            if predictor is None or not callable(getattr(predictor, 'predict', None)):
                raise ValueError('Attention requires a frozen role predictor with predict(features)')
            predictor_sha = mapping_metadata.get('predictor_sha256', '')
            if len(predictor_sha) != 64 or any(c not in '0123456789abcdef' for c in predictor_sha):
                raise ValueError('Attention requires the predictor SHA256 in mapping metadata')
            self.predictor = deepcopy(predictor)
            self.predictor.requires_grad_(False)
            self.predictor.eval()
            self.attention = SupportChoiceAttention()
        widths = [515 if attention else self.obs_dim, *hidden_dims, 4]
        if any(not isinstance(x, int) or x < 1 for x in widths):
            raise ValueError('Invalid residual network widths')
        layers = []
        for index, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if index < len(widths)-2:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # Log std avoids negative native scalar std without 29D bound machinery.
        self.distribution = GaussianDistribution(4, init_std=init_std, std_type='log', std_range=(.05,.5))
        tail = getattr(self.teacher.distribution, 'mean_tail', None)
        self._contract = dict(version=2, teacher_sha256=teacher_sha256,
            teacher_normalization='frozen complete teacher statistics',
            teacher_mean_tail=None if tail is None else tail.metadata(),
            observation_groups=self.obs_groups, observation_dimension=self.obs_dim,
            policy_action_dimension=4, decoded_action_dimension=29,
            hidden_dims=list(hidden_dims), init_std=float(init_std),
            distribution='Gaussian log std; unsquashed latent likelihood',
            std_range=[.05,.5], mean_transform='2*tanh(raw_mlp/2)',
            optimizer_projection='post Adam log_std clamp; clear projected coordinate moments; finite required',
            mapping=deepcopy(mapping_metadata))
        # Preserve the original off-arm contract/state keys exactly. Enabled
        # children explicitly declare both the new architecture and estimator.
        if attention:
            self._contract['attention'] = self.attention.metadata()
            self._contract['residual_input_dimension'] = 515

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        if self.predictor is not None:
            self.predictor.eval()
        return self

    def update_normalization(self, obs):
        """Intentionally frozen; replay/rollout never changes teacher statistics."""

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None or hidden_state is not None:
            raise ValueError('SupportChoiceActor supports unpadded feedforward batches only')
        if self.attention is None:
            with torch.no_grad():
                features = self.teacher.get_latent(obs)
        else:
            # Predictor owns its original normalization. Never replace its raw
            # 708-vector with the teacher-normalized actor input.
            with torch.no_grad():
                raw = torch.cat([obs[k] for k in self.obs_groups], -1)
                roles = self.predictor.predict(raw)['role_class_probs'].detach()
            context, weights = self.attention(obs['policy'], obs['terrain'],
                                              obs['course_tracking'], roles)
            self.last_attention_weights = weights.detach()
            features = torch.cat((obs['policy'], obs['course_tracking'], context), -1)
        raw_mean = self.mlp(features)
        if not torch.isfinite(raw_mean).all() or not torch.isfinite(self.distribution.log_std_param).all():
            raise ValueError('Nonfinite residual mean or latent log std')
        mean = 2.*torch.tanh(raw_mean/2.)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    @torch.no_grad()
    def decode_actions(self, obs, latent, context, mapper):
        """Return requested actions before the unchanged native execution chain.

        mapper(latent, context) must return [N,29] bounded single-swing-foot
        increments. Geometry/gates and their input provenance belong to mapper.
        Zero latent rows are returned bitwise from the teacher, including -0.
        """
        if latent.ndim != 2 or latent.shape[1] != 4 or not torch.isfinite(latent).all():
            raise ValueError('Expected finite [N,4] PPO latents')
        base = self.teacher(obs, stochastic_output=False)
        delta = mapper(latent, context)
        if base.shape != (latent.shape[0], 29) or delta.shape != base.shape:
            raise ValueError('Teacher and mapped actions must both be [N,29]')
        if not torch.isfinite(base).all() or not torch.isfinite(delta).all():
            raise ValueError('Non-finite teacher or mapped action')
        zero = (latent == 0).all(-1, keepdim=True)
        if (delta.masked_select(zero.expand_as(delta)) != 0).any():
            raise ValueError('Mapper violates exact zero-residual identity')
        return torch.where(zero, base, base + delta)

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        if outputs.ndim != 2 or outputs.shape[-1] != 4:
            raise ValueError('Likelihood is defined only on four-dimensional latents')
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    def get_extra_state(self):
        return deepcopy(self._contract)

    def set_extra_state(self, state):
        if state != self._contract:
            raise ValueError('Support-choice actor contract mismatch')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Validate before copying any parameter. Teacher tail buffers are not
        # persistent, so equality of metadata is essential for exact reload.
        if state_dict.get('_extra_state') != self._contract:
            raise ValueError('Support-choice actor contract mismatch')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        if self.predictor is not None:
            self.predictor.requires_grad_(False)
            self.predictor.eval()
        self.last_attention_weights = None
        return result
