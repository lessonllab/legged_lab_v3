"""Playback-only frozen assignment: one robot on every generated terrain tile."""
import math
import torch
from .stair_course import STAIR_HEIGHTS
from .stair_speed_course import StairSpeedTargetCommand
from .style_state import tensor


def matrix_assignment(num_envs, names, device):
    columns = len(names)
    if num_envs != 10 * columns:
        raise ValueError('Stair matrix requires one robot per tile across 10 rows')
    ids = torch.arange(num_envs, device=device)
    rows, cols = ids // columns, ids % columns
    kind = torch.tensor([1 if n == 'pyramid_stairs_inv' else 2 if n == 'pyramid_stairs' else 3
                         for n in names], device=device)
    groups = kind[cols]
    groups[rows == 0] = 0
    return rows, cols, groups


class StairMatrixCommand(StairSpeedTargetCommand):
    promotion_version = 7
    expected_stair_steps = 17

    @staticmethod
    def initial_layout(num_envs, names, device):
        _, columns, groups = matrix_assignment(num_envs, names, device)
        return groups, columns

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        rows, cols, _ = matrix_assignment(self.num_envs, self.column_names, self.device)
        tensor(self.terrain.terrain_levels)[:] = rows
        tensor(self.terrain.terrain_types)[:] = cols
        self.sync_origins()
        self.matrix_speed = env.cfg.stair_matrix_speed
        self.matrix_rows = rows

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = self._ids(env_ids)
        ids = ids[~self.manual_target[ids]]
        self.speed_cap[ids] = self.matrix_speed
        self._update_selected(ids)

    def matrix_labels(self):
        labels = []
        names = {'boxes':'块状地形', 'random_rough':'起伏路面',
                 'hf_pyramid_slope':'坡面', 'hf_pyramid_slope_inv':'反向坡面'}
        for i in range(self.num_envs):
            row = int(self.matrix_rows[i]); col = i % len(self.column_names)
            group = int(self.group[i]); name = self.column_names[col]
            if group == 0:
                label = '平地'
            elif group in (1, 2):
                label = f'{"上楼" if group == 1 else "下楼"} · {STAIR_HEIGHTS[row]*100:.0f}cm · 17级'
            else:
                label = f'{names.get(name,name)} · 难度{row}/9'
            labels.append(f'{label} · 行{row}列{col}')
        return labels


def configure_stair_matrix(cfg, speed=.8):
    if not math.isfinite(speed) or not 0 < speed <= 1.5:
        raise ValueError('Stair test speed must be in (0, 1.5] m/s')
    generator = cfg.scene.terrain.terrain_generator
    if tuple(generator.size) != (14., 14.):
        raise ValueError('Stair matrix requires the 17-step, 14 m terrain')
    generator.num_rows = 10
    generator.difficulty_range = (0., 1.)
    generator.curriculum = True
    cfg.scene.num_envs = generator.num_rows * generator.num_cols
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.class_type = StairMatrixCommand
    cfg.stair_matrix_speed = speed
    cfg.observations.policy.enable_corruption = False


CONTROL_TERRAINS = {
    '上楼': 'pyramid_stairs_inv', '下楼': 'pyramid_stairs',
    '平地': None, '块状地形': 'boxes', '起伏路面': 'random_rough',
    '坡面': 'hf_pyramid_slope', '反向坡面': 'hf_pyramid_slope_inv',
}


class StairControlCommand(StairMatrixCommand):
    """One robot, manually selected prebuilt tile; no training curriculum."""
    @staticmethod
    def initial_layout(num_envs, names, device):
        return (torch.ones(num_envs, device=device, dtype=torch.long),
                torch.full((num_envs,), names.index('pyramid_stairs_inv'),
                           device=device, dtype=torch.long))

    def __init__(self, cfg, env):
        StairSpeedTargetCommand.__init__(self, cfg, env)
        self.matrix_speed = env.cfg.stair_matrix_speed
        self.select_terrain('上楼', 2, self.matrix_speed)

    def select_terrain(self, kind, level, speed):
        if kind not in CONTROL_TERRAINS or int(level) != level or not 1 <= level <= 9:
            raise ValueError('Unknown terrain or difficulty outside 1–9')
        if not math.isfinite(speed) or not 0 < speed <= 1.5:
            raise ValueError('Test speed must be in (0, 1.5] m/s')
        name = CONTROL_TERRAINS[kind]
        col = 0 if name is None else self.column_names.index(name)
        row = 0 if name is None else int(level)
        group = 0 if name is None else 1 if kind == '上楼' else 2 if kind == '下楼' else 3
        self.group[:] = group
        self.flat_pool[:] = group == 0
        tensor(self.terrain.terrain_levels)[:] = row
        tensor(self.terrain.terrain_types)[:] = col
        self.sync_origins()
        self.matrix_speed = float(speed)
        self.control_kind, self.control_level = kind, row
        self.manual_target[:] = False

    def control_label(self):
        detail = (f'{STAIR_HEIGHTS[self.control_level]*100:.0f} cm · 17 级'
                  if self.control_kind in ('上楼', '下楼') else f'难度 {self.control_level}/7')
        return f'{self.control_kind} · {detail} · 速度上限 {self.matrix_speed:g} m/s'


def configure_stair_control(cfg, speed=.8):
    configure_stair_matrix(cfg, speed)
    cfg.scene.num_envs = 1
    cfg.commands.base_velocity.class_type = StairControlCommand
