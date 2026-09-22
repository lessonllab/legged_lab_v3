"""V12: retain adaptive stairs, add slow high-step rehearsal and more boxes."""
import torch
from .agility_course import agility_layout
from .adaptive_stair_course import HEIGHT
from .style_state import tensor

SLOW = 4


def foothold_layout(n,names,device):
    groups,columns=agility_layout(n,names,device)
    ids=torch.where(groups==3)[0]
    # Within other=30%: half boxes, one third rough, one sixth slopes.
    kinds=['boxes']*3+['random_rough']*2
    slopes=[i for i,name in enumerate(names) if name in ('hf_pyramid_slope','hf_pyramid_slope_inv')]
    if not slopes:raise ValueError('V12 needs slope rehearsal columns')
    for j,kind in enumerate(kinds):
        choices=torch.tensor([i for i,name in enumerate(names) if name==kind],device=device)
        if not len(choices):raise ValueError(f'V12 requires {kind}')
        sel=ids[j::6];columns[sel]=choices[torch.arange(len(sel),device=device)%len(choices)]
    sel=ids[5::6];columns[sel]=torch.tensor(slopes,device=device)[torch.arange(len(sel),device=device)%len(slopes)]
    return groups,columns


class FootholdCourseMixin:
    promotion_version=12
    layout_profile='foothold_v12'
    initial_layout=staticmethod(foothold_layout)

    def sample_adaptive(self,ids):
        super().sample_adaptive(ids)
        current=ids[self.stair_lane[ids]==HEIGHT]
        # 30% of the 50% current-height lane -> 15% of stair episodes.
        slow=current[torch.rand(len(current),device=self.device)<.3]
        self.stair_lane[slow]=SLOW

    def _resample_command(self,env_ids):
        ids=self._ids(env_ids)
        starting = ~self.stair_started[ids].clone() if hasattr(self,'stair_started') else None
        super()._resample_command(env_ids)
        if not hasattr(self,'stair_lane'):return
        slow=ids[self.stairs[ids] & (self.stair_lane[ids]==SLOW) & ~self.manual_target[ids]]
        # Keep this cap for the entire physical traversal, including timer resamples.
        fresh=ids[starting & self.stairs[ids] & (self.stair_lane[ids]==SLOW) & ~self.manual_target[ids]]
        self.speed_cap[fresh]=.2+.3*torch.rand(len(fresh),device=self.device)
        self._update_selected(slow)

    def reset(self,env_ids=None):
        ids=self._ids(env_ids)
        # Paired numerators/denominators survive logger averaging; divide summed
        # seconds or samples when analyzing, never interpret all-env means as speeds.
        extras={}
        if hasattr(self,'turn_pool'):
            for label,pool in (('turn',self.turn_pool),('reverse',self.reverse_pool)):
                selected=ids[pool[ids] & ~self.manual_target[ids] & (self.metrics[f'{label}_seconds'][ids]>0)]
                extras[f'{label}_episode_count']=float(len(selected))
                for key in (('turn_tracking','turn_drift_mps') if label=='turn' else ('reverse_tracking','reverse_actual_vx','reverse_command_vx')):
                    extras[f'{key}_sum']=float(self.metrics[key][selected].sum())
        result=super().reset(ids)
        result.update(extras)
        return result

    def load_adaptive_state(self,state,clear_evidence=False):
        super().load_adaptive_state(state,clear_evidence)
        print('[FootholdV12] actual stair mixture: 35% current / 15% slow-current (0.2-0.5 m/s) / '
              '20% fast 8 cm / 20% review / 10% challenge; height adaptation active, cap 30 cm; '
              'environment pools: flat25/up25/down20/boxes15/rough10/slopes5%',flush=True)


def foothold_curriculum(env,env_ids):
    from .adaptive_stair_course import adaptive_stair_curriculum, crossing_success
    from .stair_speed_course import cruise_success
    c=env.command_manager.get_term('base_velocity');ids=c._ids(env_ids)
    valid=~c.skip_curriculum_once[ids] & ~c.manual_target[ids]
    failed=tensor(env.termination_manager.terminated)[ids].bool()
    complete=crossing_success((c.stair_traversed|c.exit_geometry())[ids],failed)
    m=c.metrics
    speed_ok=cruise_success(m['cruise_seconds'],m['cruise_tracking'],m['cruise_actual_mps'],m['cruise_command_mps'])[ids]
    extra={}
    for group,label in ((1,'ascent'),(2,'descent')):
        mask=(c.group[ids]==group)&(c.stair_lane[ids]==SLOW)&valid
        extra[f'{label}/slow/attempts']=mask.float().sum()
        extra[f'{label}/slow/crossings']=(mask&complete).float().sum()
        extra[f'{label}/slow/successes']=(mask&complete&speed_ok).float().sum()
    types=tensor(c.terrain.terrain_types)[ids]
    boxes=torch.tensor([name=='boxes' for name in c.column_names],device=c.device)[types] & (c.group[ids]==3)&valid
    extra['boxes/attempts']=boxes.float().sum()
    extra['boxes/failures']=(boxes&failed).float().sum()
    extra['boxes/arrivals']=(m['valid_targets_reached'][ids]*boxes).sum()
    stats=adaptive_stair_curriculum(env,ids)
    stats.update(extra)
    return stats
