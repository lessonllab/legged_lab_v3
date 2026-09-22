"""Rough visual locomotion with 10-frame, body-state AMP and mirrored references.

Keeps the v2 actor inputs, terrain and task rewards. This is an isolated AMP
alignment experiment, not a full InstinctLab/WasabiPPO reproduction.
"""
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils.configclass import configclass
from legged_lab.tasks.locomotion.amp.mdp.style_state import agent_style_state, ReferenceStyleState
from .g1_amp_depth_instinct_env_cfg import G1AmpDepthInstinctEnvCfg, G1AmpDepthInstinctEnvCfg_PLAY


@configclass
class StyleAgentObs(ObservationGroupCfg):
    state = ObservationTermCfg(func=agent_style_state)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1
        self.history_length = 10
        self.flatten_history_dim = False


@configclass
class StyleReferenceObs(ObservationGroupCfg):
    state = ObservationTermCfg(func=ReferenceStyleState, params={'animation': 'animation'})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.concatenate_dim = -1
        self.history_length = None


def configure_style(cfg):
    cfg.animation.animation.num_steps_to_use = 10
    cfg.observations.disc = StyleAgentObs()
    cfg.observations.disc_demo = StyleReferenceObs()


@configclass
class G1AmpDepthStyleEnvCfg(G1AmpDepthInstinctEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_style(self)


@configclass
class G1AmpDepthStyleEnvCfg_PLAY(G1AmpDepthInstinctEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        configure_style(self)
