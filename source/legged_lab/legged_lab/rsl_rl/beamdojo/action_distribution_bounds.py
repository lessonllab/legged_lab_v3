"""Opt-in Gaussian stabilization using executed joint-action limits.

The bounded update is the RSL-RL 5.4.1 PPO update, with one explicit auxiliary
mean penalty and per-minibatch standard-deviation projection. Environment
rewards and PPO's raw-action likelihood remain unchanged.
"""
import math
import torch
from torch import nn


class ActionDistributionBounds:
    def __init__(self, alg, lower, upper, mean_penalty_coef=.01):
        distribution = alg._raw_actor.distribution
        if getattr(distribution, 'std_type', None) != 'scalar':
            raise ValueError('Action bounds require native scalar-parameter Gaussian')
        parameter = distribution.std_param
        self.lower = torch.as_tensor(lower, device=parameter.device, dtype=parameter.dtype).detach().clone()
        self.upper = torch.as_tensor(upper, device=parameter.device, dtype=parameter.dtype).detach().clone()
        self.coef = float(mean_penalty_coef)
        if (self.lower.shape != parameter.shape or self.upper.shape != parameter.shape
                or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all()
                or (self.lower >= self.upper).any() or not math.isfinite(self.coef) or self.coef < 0):
            raise ValueError('Invalid per-action bounds or mean penalty coefficient')
        self.half_width = (self.upper-self.lower)/2
        self.std_min = 1e-6
        if (self.half_width < self.std_min).any():
            raise ValueError('Action half-width is below native minimum standard deviation')
        # The native Gaussian uses these same tensors for sampling, likelihood,
        # KL and entropy. No independent likelihood approximation is introduced.
        distribution.std_range = [torch.full_like(parameter, self.std_min), self.half_width]
        self.last_projection = None
        self.projection_count = 0

    def mean_penalty(self, mean):
        overflow = torch.relu(self.lower-mean) + torch.relu(mean-self.upper)
        return self.coef * (overflow / self.half_width).square().mean()

    @torch.no_grad()
    def project_std(self, alg):
        parameter = alg._raw_actor.distribution.std_param
        if alg._numerical_guard is not None:
            alg._numerical_guard.check_finite('actor.distribution.std_param', parameter,
                stage='std_projection_before', step=alg._numerical_guard_step)
        elif not torch.isfinite(parameter).all():
            raise RuntimeError('Nonfinite standard deviation before projection')
        projected = torch.minimum(parameter.clamp_min(self.std_min), self.half_width)
        changed = projected != parameter
        ids = changed.nonzero().flatten()
        raw = parameter[ids].cpu().tolist()
        after = projected[ids].cpu().tolist()
        cleared = []
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            state = alg.optimizer.state.get(parameter, {}).get(name)
            if state is not None:
                if alg._numerical_guard is not None:
                    alg._numerical_guard.check_finite('optimizer.distribution.std_param.' + name, state,
                        stage='std_projection_before', step=alg._numerical_guard_step)
                elif not torch.isfinite(state).all():
                    raise RuntimeError('Nonfinite Adam standard deviation state before projection')
                state[changed] = 0.
                if ids.numel():
                    cleared.append(name)
        parameter.copy_(projected)
        report = {'projected_indices': ids.cpu().tolist(), 'raw_std': raw,
                  'projected_std': after, 'cleared_adam_slots': cleared,
                  'optimizer_step_preserved': True}
        if ids.numel():
            self.last_projection = report
            self.projection_count += int(ids.numel())
        return report

    def metadata(self):
        return {'version': 1, 'lower': self.lower.cpu().tolist(), 'upper': self.upper.cpu().tolist(),
                'mean_penalty_coef': self.coef, 'std_max': self.half_width.cpu().tolist(),
                'std_min': self.std_min, 'projection_count': self.projection_count,
                'last_projection': self.last_projection}


def bounded_update(self) -> dict[str, float]:
    """Run optimization epochs over stored batches and return mean losses."""
    mean_bounds_loss = 0.
    raw_mean_peak = bounded_mean_peak = 0.
    tail_elements = total_mean_elements = 0
    mean_value_loss = 0
    mean_surrogate_loss = 0
    mean_entropy = 0
    # RND loss
    mean_rnd_loss = 0 if self.rnd else None
    # Symmetry loss
    mean_symmetry_loss = 0 if self.symmetry else None

    # Get mini-batch generator
    if self.actor.is_recurrent or self.critic.is_recurrent:
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

    # Iterate over mini-batches
    for batch in generator:
        original_batch_size = batch.observations.batch_size[0]

        # Check if we should normalize advantages per mini-batch
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

        # Perform symmetric augmentation if enabled
        if self.symmetry:
            self.symmetry.augment_batch(batch, original_batch_size)

        # Recompute actions log prob and entropy for current batch of transitions
        # Note: We need to do this because we updated the policy with new parameters
        self.actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=True,
        )
        actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
        values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
        # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
        distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
        entropy = self.actor.output_entropy[:original_batch_size]

        # Compute KL divergence and adapt the learning rate
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                kl_mean = torch.mean(kl)

                # Reduce the KL divergence across all GPUs
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size

                # Update the learning rate only on the main process
                if self.gpu_global_rank == 0:
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                # Update the learning rate for all GPUs
                if self.is_multi_gpu:
                    lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                    torch.distributed.broadcast(lr_tensor, src=0)
                    self.learning_rate = lr_tensor.item()

                # Update the learning rate for all parameter groups
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
        surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
        surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
            value_losses = (values - batch.returns).pow(2)
            value_losses_clipped = (value_clipped - batch.returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch.returns - values).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        raw_mean = self.action_raw_mean()
        penalty_mean = distribution_params[0] if raw_mean is None else raw_mean[:original_batch_size]
        bounds_loss = self._action_distribution_bounds.mean_penalty(penalty_mean)
        if raw_mean is not None:
            raw_mean_peak = max(raw_mean_peak, float(penalty_mean.detach().abs().max()))
            bounded_mean_peak = max(bounded_mean_peak, float(distribution_params[0].detach().abs().max()))
            transform = self._raw_actor.distribution.mean_tail
            tail_elements += int(((penalty_mean < transform.identity_lower)
                                  | (penalty_mean > transform.identity_upper)).sum())
            total_mean_elements += penalty_mean.numel()
        loss = loss + bounds_loss
        mean_bounds_loss += bounds_loss.item()

        # RND loss
        rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

        # Symmetry loss
        if self.symmetry:
            symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
            if self.symmetry.use_mirror_loss:
                loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

        # Compute the gradients for PPO
        self.optimizer.zero_grad()
        loss.backward()
        # Compute the gradients for RND
        if self.rnd:
            self.rnd.optimizer.zero_grad()
            rnd_loss.backward()

        # Collect gradients from all GPUs
        if self.is_multi_gpu:
            self.reduce_parameters()

        # Apply the gradients for PPO
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._action_distribution_bounds.project_std(self)
        # Apply the gradients for RND
        if self.rnd:
            self.rnd.optimizer.step()

        # Store the losses
        mean_value_loss += value_loss.item()
        mean_surrogate_loss += surrogate_loss.item()
        mean_entropy += entropy.mean().item()
        # RND loss
        if mean_rnd_loss is not None:
            mean_rnd_loss += rnd_loss.item()
        # Symmetry loss
        if mean_symmetry_loss is not None:
            mean_symmetry_loss += symmetry_loss.item()

    # Divide the losses by the number of updates
    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
        mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
        mean_symmetry_loss /= num_updates

    # Construct the loss dictionary
    loss_dict = {
        "action_mean_bounds": mean_bounds_loss / num_updates,
        "value": mean_value_loss,
        "surrogate": mean_surrogate_loss,
        "entropy": mean_entropy,
    }
    if total_mean_elements:
        loss_dict['action_raw_mlp_mean_abs_max'] = raw_mean_peak
        loss_dict['action_bounded_mean_abs_max'] = bounded_mean_peak
        loss_dict['action_mean_tail_fraction'] = tail_elements / total_mean_elements
    if self.rnd:
        loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
        loss_dict["symmetry"] = mean_symmetry_loss

    # Clear the storage
    self.storage.clear()

    return loss_dict
