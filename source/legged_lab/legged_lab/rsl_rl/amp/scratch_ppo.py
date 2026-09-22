"""Keep the opt-in scratch course's terrain levels across checkpoint restores."""
import torch
from .ppo_amp import PPOAMP


def course_restore_indices(source_count, destination_columns, column_count):
    """Sample within matching terrain columns when playback uses fewer robots."""
    destination_columns = destination_columns.detach().cpu().long()
    source_columns = (torch.arange(source_count) / (source_count / column_count)).long()
    indices = torch.empty(len(destination_columns), dtype=torch.long)
    for column in destination_columns.unique():
        source = torch.where(source_columns == column)[0]
        destination = torch.where(destination_columns == column)[0]
        if not len(source):
            raise ValueError('Checkpoint has no environments for a requested terrain column')
        positions = ((torch.arange(len(destination)) + .5) * len(source) / len(destination)).long()
        indices[destination] = source[positions]
    return indices


class ScratchPPOAMP(PPOAMP):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        alg = PPOAMP.construct_algorithm(obs, env, cfg, device)
        alg.course_env = env.unwrapped
        return alg

    def save(self):
        saved = super().save()
        command = self.course_env.command_manager.get_term('base_velocity')
        terrain = self.course_env.scene.terrain
        saved['scratch_course'] = {
            'version': 1, 'levels': terrain.terrain_levels.detach().cpu().clone(),
            'streak': command.success_streak.detach().cpu().clone(),
            'rows': terrain.max_terrain_level,
            'columns': list(command.column_names),
        }
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        state = loaded_dict.get('scratch_course')
        if state is None:
            raise ValueError('Scratch course requires its own checkpoint; omit --resume for a fresh run.')
        env = self.course_env
        terrain = env.scene.terrain
        command = env.command_manager.get_term('base_velocity')
        if state['version'] != 1 or state['rows'] != terrain.max_terrain_level:
            raise ValueError('Checkpoint curriculum layout differs from the current terrain layout.')
        if state.get('columns') != list(command.column_names):
            raise ValueError('Checkpoint curriculum terrain columns differ from the current layout.')
        n = len(terrain.terrain_levels)
        indices = course_restore_indices(len(state['levels']), terrain.terrain_types, len(command.column_names))
        levels = state['levels'][indices].to(env.device)
        if ((levels < 0) | (levels >= terrain.max_terrain_level)).any():
            raise ValueError('Invalid checkpoint terrain levels')
        result = super().load(loaded_dict, load_cfg, strict)
        terrain.terrain_levels[:] = levels
        terrain.env_origins[:] = terrain.terrain_origins[levels, terrain.terrain_types]
        command.skip_curriculum_once[:] = True
        env.reset()
        command.success_streak[:] = state['streak'][indices].to(env.device)
        print('[ScratchCourse] restored terrain levels and success streaks', flush=True)
        return result
