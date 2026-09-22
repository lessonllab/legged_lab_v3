"""Instinct-inspired visual inputs, using Isaac Lab's native dynamic ray caster.

Reference: project-instinct/InstinctLab, parkour_env_cfg.py and
g1/g1_parkour_target_amp_cfg.py. This retains this project's locomotion task,
AMP data and terrain curriculum; it is not a reproduction of the entire paper.
"""

import math

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg
from isaaclab.utils.configclass import configclass

from legged_lab.tasks.locomotion.amp.mdp.instinct_depth import InstinctDepthHistory
from legged_lab.tasks.locomotion.amp.mdp.rewards import dont_wait
from legged_lab.tasks.locomotion.amp.mdp.foot_support import sole_ray_pattern, foot_support_penalty
from .g1_amp_depth_env_cfg import G1AmpDepthEnvCfg
from .g1_amp_depth_warmup_env_cfg import G1AmpDepthWarmupEnvCfg


def g1_depth_mesh_targets():
    """One rigid body per target, so PhysX's body-major views cannot interleave environments.

    This USD folds head_link into torso_link/visuals. The nominal optical center
    sits behind its opaque shell mesh (no lens aperture in the mesh). Exclude
    only that fixed head shell from depth ray casting, retaining torso and logo.
    Visual rendering, collision shapes, masses and joint dynamics are untouched.
    """
    bodies = ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"]
    for side in ("left", "right"):
        bodies.extend(f"{side}_{part}_link" for part in (
            "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
            "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
            "wrist_roll", "wrist_pitch", "wrist_yaw",
        ))
    targets = ["/World/ground"]
    for body in bodies:
        visual_paths = ("visuals/torso_link_rev_1_0", "visuals/logo_link") if body == "torso_link" else ("visuals",)
        for visual_path in visual_paths:
            targets.append(MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{{ENV_REGEX_NS}}/Robot/{body}/{visual_path}",
                track_mesh_transforms=True, merge_prim_meshes=True, is_shared=True,
            ))
    return targets


@configclass
class InstinctDepthObservationsCfg(ObservationGroupCfg):
    image = ObservationTermCfg(func=InstinctDepthHistory, params={"sensor_name": "depth_camera"})

    def __post_init__(self):
        self.concatenate_terms = True
        self.concatenate_dim = 1
        self.enable_corruption = False
        self.history_length = None


def configure_instinct_vision(cfg):
    old = cfg.scene.depth_camera
    # Reference is WXYZ; this installed Isaac Lab uses XYZW. Normalize the
    # slightly non-unit published quaternion, retaining its small roll component.
    xyzw = (0.004363309284746571, 0.4067366430758002, 0.0, 0.9135367613482678)
    norm = math.sqrt(sum(v * v for v in xyzw))
    cfg.scene.depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path=old.prim_path,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.0487988662332928, 0.01, 0.4378029937970051),
            rot=tuple(v / norm for v in xyzw), convention="world",
        ),
        mesh_prim_paths=g1_depth_mesh_targets(),
        pattern_cfg=old.pattern_cfg,
        update_period=cfg.sim.dt * cfg.decimation,
        data_types=["distance_to_image_plane"],
        max_distance=2.5, depth_clipping_behavior="max",
        reference_meshes=True, debug_vis=False,
    )
    cfg.observations.depth = InstinctDepthObservationsCfg()
    cfg.observations.policy.height_scan = None
    cfg.observations.critic.height_scan = None
    # The scanner remains a simulator-only termination aid (ground-relative
    # fall detection). Neither neural network receives its height samples.
    feet = [f"{side}_ankle_roll_link" for side in ("left", "right")]
    names = [f"{side}_sole_scanner" for side in ("left", "right")]
    for name, body in zip(names, feet):
        setattr(cfg.scene, name, RayCasterCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
            ray_alignment="base", pattern_cfg=patterns.PatternBaseCfg(func=sole_ray_pattern),
            mesh_prim_paths=["/World/ground"], max_distance=2.,
            update_period=cfg.sim.dt * cfg.decimation, debug_vis=False,
        ))
    for kind, weight in (("support", -.25), ("edge", -.10)):
        setattr(cfg.rewards, f"feet_{kind}", RewardTermCfg(
            func=foot_support_penalty, weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=feet, preserve_order=True),
                    "contact_cfg": SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True),
                    "sensor_names": names, "kind": kind},
        ))


@configclass
class G1AmpDepthInstinctEnvCfg(G1AmpDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_instinct_vision(self)
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.dont_wait = RewardTermCfg(
            func=dont_wait, weight=-0.5,
            params={"command_name": "base_velocity", "startup_grace_s": 0.5},
        )


@configclass
class G1AmpDepthInstinctEnvCfg_PLAY(G1AmpDepthInstinctEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.curriculum.terrain_levels = None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 6


@configclass
class G1AmpDepthInstinctWarmupEnvCfg(G1AmpDepthWarmupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_instinct_vision(self)
        # This backend triangulates an infinite Plane at +/-1e6 m, losing
        # centimetres in float32 ray intersections. Finite tiles remain flat.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=42, size=(20., 20.), num_rows=8, num_cols=8,
            border_width=20., curriculum=False,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.)},
        )
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class G1AmpDepthInstinctWarmupEnvCfg_PLAY(G1AmpDepthInstinctWarmupEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
