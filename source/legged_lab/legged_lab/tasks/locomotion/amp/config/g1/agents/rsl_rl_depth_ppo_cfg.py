from isaaclab_rl.rsl_rl import RslRlCNNModelCfg
from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_cfg import G1AmpRoughPPORunnerCfg


@configclass
class G1AmpDepthPPORunnerCfg(G1AmpRoughPPORunnerCfg):
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=0.5),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[5, 3, 3],
            stride=[2, 2, 2],
            activation="elu",
            flatten=True,
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_amp_depth"
        self.obs_groups["actor"] = ["policy", "depth"]
        # Only the critic receives the ideal terrain scan; AMP receives motion only.
        self.algorithm.symmetry_cfg = None
        self.algorithm.learning_rate = 3.0e-4
