"""Check target patches, commands, curriculum relocation and optional Viser marker."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from isaaclab.app.sim_launcher import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
import legged_lab.tasks
from legged_lab.tasks.locomotion.amp.mdp.style_state import tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/analysis/target_v4/physical.json'))
    parser.add_argument('--viser_port', type=int, default=0)
    parser.add_argument('--hold_seconds', type=float, default=0.)
    add_launcher_args(parser)
    args = parser.parse_args()
    task = 'LeggedLab-Isaac-AMP-Depth-Target-G1-v0'
    cfg = load_cfg_from_registry(task,'env_cfg_entry_point')
    cfg.sim.physics = copy.deepcopy(cfg.sim.physics.default)
    cfg.scene.num_envs = 6;cfg.seed=42
    generator = cfg.scene.terrain.terrain_generator
    generator.num_cols = 6
    for sub in generator.sub_terrains.values():sub.proportion=1.
    with launch_simulation(cfg,args):
        # Import only after launch: pxr preloading crashes this Kit installation.
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rsl_rl'))
        from physx_viser import PhysxViser, _mesh, _mesh_prims
        env = gym.make(task,cfg=cfg).unwrapped
        viewer = None
        try:
            observations,_ = env.reset()
            terrain=env.scene.terrain; robot=env.scene['robot'];cmd=env.command_manager.get_term('base_velocity')
            assert cmd.column_names == list(generator.sub_terrains)
            patches = cmd.valid_targets
            local = patches - tensor(terrain.terrain_origins)[:,:,None]
            assert local[...,:2].abs().max() <= 3.701
            torch.testing.assert_close(local[:,:2,:,0],torch.full_like(local[:,:2,:,0],3.7),atol=1e-4,rtol=1e-5)
            torch.testing.assert_close(local[:,:2,:,1],torch.zeros_like(local[:,:2,:,1]),atol=1e-4,rtol=0.)
            # Raycast every patch against the terrain actually imported into the scene.
            geometry=[_mesh(p) for p in _mesh_prims(env.sim.stage.GetPrimAtPath('/World/ground'))]
            vertices,faces=[],[];offset=0
            for entry in geometry:
                if entry is None:continue
                v,f=entry;vertices.append(v);faces.append(f+offset);offset+=len(v)
            mesh=convert_to_warp_mesh(np.concatenate(vertices),np.concatenate(faces),device=env.device)
            points=patches.reshape(-1,3)
            starts=points.clone();starts[:,2]+=2.
            directions=torch.zeros_like(starts);directions[:,2]=-1.
            hits=raycast_mesh(starts,directions,mesh,max_dist=4.)[0]
            assert torch.isfinite(hits).all()
            terrain_error=(hits[:,2]-points[:,2]).abs().max().item()
            # Isaac Lab stores one ring sample's Z, not the exact center height.
            # The patch tolerance is 5 cm; commands use XY distance only.
            assert terrain_error<.051,terrain_error
            # Upgrade only env 0, downgrade env 1 from row 2; verify new goals use new rows.
            terrain.terrain_levels[1]=2
            cmd.metrics['tracking_exp_vel_xy'][:]=.4
            cmd.metrics['tracking_exp_vel_yaw'][:]=.2
            cmd.metrics['tracking_exp_vel_xy'][0]=.7
            cmd.metrics['tracking_exp_vel_xy'][1]=.2
            unchanged=cmd.pos_command_w[2:].clone()
            env._reset_idx(torch.tensor([0,1],device=env.device))
            assert tensor(terrain.terrain_levels)[:2].tolist()==[1,1]
            torch.testing.assert_close(cmd.pos_command_w[2:],unchanged)
            for i in range(6):
                candidates=patches[int(terrain.terrain_levels[i]),int(terrain.terrain_types[i])]
                assert (candidates-cmd.pos_command_w[i]).norm(dim=-1).min()<1e-5
            # At-goal commands stop even with yaw error. Moving away restores pursuit.
            pose=torch.cat((tensor(robot.data.root_pos_w).clone(),tensor(robot.data.root_quat_w).clone()),-1)
            pose[0,:2]=cmd.pos_command_w[0,:2]
            robot.write_root_pose_to_sim(pose)
            env.sim.forward();env.scene.update(env.physics_dt)
            cmd.is_standing_env[:]=False;cmd._update_command()
            assert (cmd.command[0]==0).all()
            pose[0,0]-=2.
            robot.write_root_pose_to_sim(pose)
            env.sim.forward();env.scene.update(env.physics_dt);cmd._update_command()
            assert cmd.command[0].abs().sum()>0
            for _ in range(40):
                observations,*_=env.step(torch.zeros(env.num_envs,env.action_manager.total_action_dim,device=env.device))
                assert torch.isfinite(cmd.command).all()
                assert (cmd.command[:,0]>=0).all() and (cmd.command[:,0]<=cmd.speed_cap+1e-6).all()
                assert (cmd.command[:,1]==0).all() and (cmd.command[:,2].abs()<=1.).all()
            result={'device':env.device,'patch_shape':list(patches.shape),'all_patches_on_terrain':True,
                    'max_patch_center_height_offset_m':terrain_error,'terrain_columns':cmd.column_names,
                    'partial_reset_and_curriculum_relocation':True,'arrival_stop_and_restart':True,
                    'dynamic_steps':40,'finite_bounded_commands':True}
            if args.viser_port:
                env.terrain_showcase_names=cmd.column_names
                viewer=PhysxViser(env,port=args.viser_port)
                viewer.update(observations)
                assert viewer.target_marker is not None
                result['viser_goal_marker_created']=True
                print('VISER_READY',args.viser_port,flush=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2))
            print('PASS',json.dumps(result),flush=True)
            deadline=time.monotonic()+args.hold_seconds
            while time.monotonic()<deadline:
                if viewer:viewer.update(observations)
                time.sleep(.1)
        finally:
            if viewer:viewer.close()
            env.close()


if __name__=='__main__':main()
