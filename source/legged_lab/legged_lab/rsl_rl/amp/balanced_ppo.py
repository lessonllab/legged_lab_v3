"""Versioned curriculum migration; model/optimizer training state is retained."""
import torch
from .scratch_ppo import ScratchPPOAMP, course_restore_indices


def clamp_loaded_exploration(actor, optimizer):
    """Keep transferred noise parameters inside bounds so clamp gradients exist."""
    distribution = actor.distribution
    if distribution.std_type == 'scalar':
        parameter, bounds = distribution.std_param, distribution.std_range
    else:
        parameter, bounds = distribution.log_std_param, distribution.log_std_range
    with torch.no_grad():
        clamped = parameter.clamp(*bounds)
        changed = not torch.equal(parameter, clamped)
        if changed:
            parameter.copy_(clamped)
            optimizer.state.pop(parameter, None)
    return changed


class BalancedPPOAMP(ScratchPPOAMP):
    def save(self):
        saved = super().save()
        c = self.course_env.command_manager.get_term('base_velocity')
        saved['scratch_course']['version'] = 2
        for name in ('flat_pool', 'flat_speed_tier', 'failure_streak', 'speed_good', 'speed_bad'):
            saved['scratch_course'][name] = getattr(c, name).detach().cpu().clone()
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        original = loaded_dict.get('scratch_course')
        if original is None or original['version'] not in (1, 2):
            raise ValueError('Expected a scratch-course v1/v2 checkpoint')
        env = self.course_env
        c, terrain = env.command_manager.get_term('base_velocity'), env.scene.terrain
        state = dict(original)
        indices = course_restore_indices(len(state['levels']), terrain.terrain_types, len(c.column_names))
        if original['version'] == 2:
            for name in ('flat_pool', 'flat_speed_tier', 'failure_streak', 'speed_good', 'speed_bad'):
                getattr(c, name)[:] = original[name][indices].to(env.device)
        else:
            # No fast-flat proficiency is inferred from the old terrain level.
            c.flat_speed_tier.zero_()
            c.failure_streak.zero_()
            c.speed_good.zero_()
            c.speed_bad.zero_()
            state['streak'] = torch.zeros_like(state['streak'])
            print('[BalancedCourse] migrating old course: preserve policy, reset promotion evidence and flat speed to .6', flush=True)
        # Parent validates layout and restores levels, then resets the physical
        # environment. Pool assignments must be reflected before that reset.
        state['levels'] = state['levels'].clone()
        if original['version'] == 1:
            c.flat_pool[:] = torch.arange(len(terrain.terrain_levels), device=env.device) % 4 == 0
            state['levels'][indices[c.flat_pool.cpu()]] = 0
        state['version'] = 1
        converted = dict(loaded_dict, scratch_course=state)
        result = super().load(converted, load_cfg, strict)
        if clamp_loaded_exploration(self.actor, self.optimizer):
            print('[BalancedCourse] clamped transferred exploration; reset only its optimizer moments', flush=True)
        return result
