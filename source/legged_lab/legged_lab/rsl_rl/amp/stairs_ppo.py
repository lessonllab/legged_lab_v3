"""Explicit v2 -> stair-course migration without reinterpreting terrain levels."""
import torch
from .ppo_amp import PPOAMP
from .scratch_ppo import ScratchPPOAMP
from .balanced_ppo import clamp_loaded_exploration
from legged_lab.tasks.locomotion.amp.mdp.stair_course import ENV_STATE, GLOBAL_STATE, STAIR_HEIGHTS
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def sample_indices(source, count):
    if not len(source):
        raise ValueError('No matching source curriculum environments')
    return source[((torch.arange(count) + .5) * len(source) / count).long()]


def restore_indices(state, groups, columns, allow_remap=False):
    # Playback loads checkpoints with map_location=cuda; layout matching is
    # deliberately CPU-only, independent of where model weights were loaded.
    state = {key: value.cpu() if isinstance(value, torch.Tensor) else value
             for key, value in state.items()}
    groups, columns = groups.cpu(), columns.cpu()
    n = len(state['levels'])
    if state['version'] in (3, 4, 5) and n == len(groups) and not allow_remap:
        if not torch.equal(state['group'], groups) or not torch.equal(state['types'], columns):
            raise ValueError('Same-size specialist checkpoint has a different environment assignment')
        return torch.arange(n)
    result = torch.empty(len(groups), dtype=torch.long)
    flat_dst = torch.where(groups == 0)[0]
    flat_src = torch.where(state['flat_pool'])[0]
    result[flat_dst] = sample_indices(flat_src, len(flat_dst))
    if state['version'] == 2:
        source_columns = (torch.arange(n) / (n / len(state['columns']))).long()
        for g in groups[groups != 0].unique():
            for column in columns[groups == g].unique():
                dst = torch.where((groups == g) & (columns == column))[0]
                name = state['columns'][column]
                matching = torch.tensor([state['columns'][i] == name for i in source_columns])
                src = torch.where(matching & ~state['flat_pool'])[0]
                result[dst] = sample_indices(src, len(dst))
    else:
        for g in groups[groups != 0].unique():
            for column in columns[groups == g].unique():
                dst = torch.where((groups == g) & (columns == column))[0]
                src = torch.where((state['group'] == g) & (state['types'] == column))[0]
                result[dst] = sample_indices(src, len(dst))
    return result


class StairsPPOAMP(ScratchPPOAMP):
    def save(self):
        saved = PPOAMP.save(self)
        c = self.course_env.command_manager.get_term('base_velocity')
        t = self.course_env.scene.terrain
        state = {'version': 4, 'rows': t.max_terrain_level, 'columns': list(c.column_names),
                 'layout_profile': getattr(c,'layout_profile','specialist_v1'),
                 'terrain_profile': getattr(c, 'terrain_profile', 'short8_v1'),
                 'tile_size': getattr(c, 'terrain_tile_size', (8., 8.)),
                 'heights': STAIR_HEIGHTS,
                 'levels': tensor(t.terrain_levels).detach().cpu().clone(),
                 'types': tensor(t.terrain_types).detach().cpu().clone()}
        for name in ENV_STATE + GLOBAL_STATE:
            state[name] = getattr(c, name).detach().cpu().clone()
        if getattr(c, 'course_profile', None) == 'speed_v1':
            state.update(version=5, profile=c.course_profile,
                         flat_speeds=c.flat_speed_schedule, stair_speeds=c.stair_speed_schedule,
                         promotion_version=getattr(c, 'promotion_version', 1))
        if hasattr(c, 'save_adaptive_state'):
            state['adaptive_height'] = c.save_adaptive_state()
        saved['scratch_course'] = state
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        state = loaded_dict.get('scratch_course', {})
        env, c = self.course_env, self.course_env.command_manager.get_term('base_velocity')
        t = env.scene.terrain
        if state.get('version') not in (2, 3, 4, 5):
            raise ValueError('Stair specialization requires a balanced v2 or specialist v3/v4/v5 checkpoint')
        speed_profile = getattr(c, 'course_profile', None) == 'speed_v1'
        source_terrain = state.get('terrain_profile', 'short8_v1')
        target_terrain = getattr(c, 'terrain_profile', 'short8_v1')
        geometry_changed = source_terrain != target_terrain
        migration = (source_terrain, target_terrain)
        if geometry_changed and migration not in (('short8_v1', 'long24_v1'), ('long24_v1', 'stairs14_v2')):
            raise ValueError('Terrain checkpoint requires its matching long/short stair task')
        expected_size = ({'short8_v1': (8., 8.), 'long24_v1': (24., 24.)}[source_terrain]
                         if geometry_changed else getattr(c, 'terrain_tile_size', (8., 8.)))
        if tuple(state.get('tile_size', (8., 8.))) != expected_size:
            raise ValueError('Incompatible saved terrain tile size')
        if state['version'] == 5:
            if (not speed_profile or state.get('profile') != c.course_profile
                    or tuple(state.get('flat_speeds', ())) != c.flat_speed_schedule
                    or tuple(state.get('stair_speeds', ())) != c.stair_speed_schedule):
                raise ValueError('Speed v5 checkpoint requires its matching Stairs-v1 schedule')
            if state.get('promotion_version', 1) > getattr(c, 'promotion_version', 1):
                raise ValueError('Checkpoint uses a newer promotion definition')
        if state['rows'] != t.max_terrain_level or state['columns'] != list(c.column_names):
            raise ValueError('Incompatible terrain geometry layout')
        saved_heights = tuple(state.get('heights', ()))
        height_expanded = saved_heights == (0., .08, .10, .12, .14, .16, .18, .20, .215, .23) and STAIR_HEIGHTS == (0., .08, .10, .12, .14, .16, .18, .20, .25, .30)
        if state['version'] in (3, 4, 5) and saved_heights != STAIR_HEIGHTS and not height_expanded:
            raise ValueError('Incompatible stair height schedule')
        source_layout = state.get('layout_profile','specialist_v1')
        target_layout = getattr(c,'layout_profile','specialist_v1')
        layout_changed = source_layout != target_layout
        if layout_changed and (source_layout,target_layout) not in (
                ('specialist_v1','agility_v10'), ('agility_v10','foothold_v12')):
            raise ValueError('Unsupported training layout migration')
        indices = restore_indices(state, c.group, tensor(t.terrain_types),allow_remap=layout_changed)
        levels = state['levels'][indices].to(env.device).clone()
        if state['version'] == 2:
            # Only stair difficulty is reset. Flat speed proficiency and the
            # non-stair rough levels survive migration to the new sampler.
            levels[c.stairs] = 2
            levels[c.flat_pool] = 0
            names = ('flat_speed_tier', 'speed_good', 'speed_bad')
        else:
            names = ENV_STATE
        if ((levels < 0) | (levels >= t.max_terrain_level)).any():
            raise ValueError('Invalid terrain levels in checkpoint')
        restored = {name: state[name][indices].to(env.device) for name in names}
        globals_ = {name: state[name].to(env.device) for name in GLOBAL_STATE} if state['version'] in (3, 4, 5) else {}
        if speed_profile:
            if ((restored['flat_speed_tier'] < 0) | (restored['flat_speed_tier'] >= len(c.flat_speed_schedule))).any():
                raise ValueError('Invalid flat speed tier')
            for values in (globals_.get('phase'), restored.get('stair_phase')):
                if values is not None and ((values < 0) | (values >= len(c.stair_speed_schedule))).any():
                    raise ValueError('Invalid stair speed phase')
        result = PPOAMP.load(self, loaded_dict, load_cfg, strict)
        for name, value in restored.items():
            getattr(c, name)[:] = value
        for name, value in globals_.items():
            getattr(c, name)[:] = value
        if height_expanded:
            # Rows above 20 cm changed geometry: retain mastered heights only.
            c.frontier.clamp_(max=7)
            levels[c.stairs] = levels[c.stairs].clamp(max=7)
            print('[StairCourse] height cap 20 -> 30 cm; preserve frontier up to 20 cm', flush=True)
        if geometry_changed and target_terrain == 'long24_v1':
            # Same rise does not imply the same proficiency over five times as
            # many treads. Preserve flat skills; begin long stairs at 8 cm/phase 0.
            levels[c.stairs] = 1
            c.frontier.fill_(1)
            c.phase.zero_()
            c.stair_phase.zero_()
            print('[StairCourse] short -> long stairs: preserve policy/flat tiers; start long stairs at 8 cm, phase 0', flush=True)
        clear_evidence = (height_expanded or layout_changed or geometry_changed or state['version'] < 4 or (speed_profile and
                          (state['version'] < 5 or state.get('promotion_version', 1)
                           != getattr(c, 'promotion_version', 1))))
        if clear_evidence:
            # The success definition changed. Preserve proficiency/stages but
            # never mix the old strict-stop failures into new evidence windows.
            c.attempts.zero_()
            c.wins.zero_()
            c.last_rate.fill_(-1.)
            c.epoch.zero_()
            c.stair_epoch.zero_()
            c.success_streak.zero_()
            c.failure_streak.zero_()
            if speed_profile:
                c.speed_good.zero_()
                c.speed_bad.zero_()
            print('[StairCourse] migrating completion criteria: preserve terrain/flat speed stages; clear old evidence', flush=True)
        tensor(t.terrain_levels)[:] = levels
        if hasattr(c, 'load_adaptive_state'):
            c.load_adaptive_state(state.get('adaptive_height'), clear_evidence=clear_evidence)
        c.sync_origins()
        c.skip_curriculum_once[:] = True
        env.reset()
        clamp_loaded_exploration(self.actor, self.optimizer)
        print(f'[StairCourse] loaded v{state["version"]}: actor/critic/AMP/optimizers retained; '
              + ('stairs reset to 10 cm, flat speed tiers retained' if state['version'] == 2
                 else 'stair stages and flat tiers restored; '
                      + ('old evidence cleared' if clear_evidence else 'evidence restored')), flush=True)
        return result
