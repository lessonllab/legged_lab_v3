"""Reproduce v4 stationary-promotion and turning/standing reward conflicts."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from legged_lab.tasks.locomotion.amp.mdp.commands.target_velocity import target_velocity,target_tracking_curriculum
from legged_lab.tasks.locomotion.amp.mdp.rewards import stand_still_joint_deviation_l1,dont_wait


def main():
    v,_,_,_=target_velocity(torch.zeros(1,3),torch.tensor([[0.,0.,0.,1.]]),
        torch.tensor([[-3.,0.,0.]]),torch.tensor([.6]),torch.tensor([False]))
    xy=torch.exp(-v[:,:2].square().sum(-1)/.25)
    yaw=torch.exp(-v[:,2].square()/.25)
    updates=[]
    terrain=NS(terrain_levels=torch.zeros(1),update_env_origins=lambda ids,up,down:updates.append((up,down)))
    command=NS(metrics={'tracking_exp_vel_xy':xy,'tracking_exp_vel_yaw':yaw})
    class Scene(dict):pass
    scene=Scene(robot=NS(data=NS(root_lin_vel_b=torch.zeros(1,3))));scene.terrain=terrain
    env=NS(scene=scene,command_manager=NS(get_term=lambda _:command,get_command=lambda _:v),
           episode_length_buf=torch.tensor([100]),step_dt=.02)
    target_tracking_curriculum(env,torch.tensor([0]))
    # Unit joint deviation isolates whether the reward is enabled, not actual pose magnitude.
    with patch('legged_lab.tasks.locomotion.amp.mdp.rewards.mdp.joint_deviation_l1',return_value=torch.ones(1)):
        standing_penalty=stand_still_joint_deviation_l1(env,'base_velocity')
    result={'command':v.tolist(),'stationary_full_horizon_xy_score':xy.item(),
            'stationary_full_horizon_yaw_score':yaw.item(),'stationary_is_promoted':bool(updates[0][0].item()),
            'turning_activates_standing_penalty':bool(standing_penalty.item()),
            'turning_does_not_activate_dont_wait':dont_wait(env,'base_velocity').item()==0.,
            'scope':'Deterministic counterexample to v4 reward and curriculum logic, not a measured rollout.'}
    assert result['stationary_is_promoted'] and result['turning_activates_standing_penalty']
    path=Path('docs/analysis/target_failure_20260912/incentives.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
