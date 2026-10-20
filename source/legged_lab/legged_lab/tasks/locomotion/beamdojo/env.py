"""Native environment with explicit dense/sparse transition reward channels."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from . import paper_mdp, repo_mdp, upstream_terrain, course_tracking, recovery_curriculum, action_bounds, course_schedule
from . import contact_event_risk, support_experiment_runtime, corridor_runtime, corridor_mixed_runtime


class BeamDojoEnv(ManagerBasedRLEnv):
    def load_managers(self):
        super().load_managers()
        if getattr(self.cfg,'beamdojo_repo_aligned',False) or upstream_terrain.enabled(self.cfg):
            original_compute=self.reward_manager.compute
            def compute_repo_rewards(dt):
                if getattr(self.cfg, 'beamdojo_contact_event_risk', False):
                    contact_event_risk.update(self)
                    self._beamdojo_event_log = contact_event_risk.diagnostics(self)
                if support_experiment_runtime.enabled(self):
                    support_experiment_runtime.update(self)
                    self._beamdojo_support_log = support_experiment_runtime.diagnostics(self)
                if recovery_curriculum.enabled(self.cfg):
                    recovery_curriculum.update(self)
                    self._beamdojo_recovery_log = recovery_curriculum.diagnostics(self)
                if getattr(self.cfg, 'beamdojo_course_tracking', False):
                    self._beamdojo_tracking_log = course_tracking.diagnostics(self)
                if upstream_terrain.enabled(self.cfg):
                    upstream_terrain.update_success(self)
                    if recovery_curriculum.enabled(self.cfg):
                        self.beamdojo_qualified_success &= recovery_curriculum.motion_pass(self)
                if corridor_runtime.enabled(self):
                    corridor_runtime.update(self)
                    self.beamdojo_qualified_success &= corridor_runtime.qualified_mask(self)
                    self._beamdojo_corridor_transition = corridor_runtime.transition_metrics(self)
                    self._beamdojo_corridor_log = corridor_runtime.diagnostics(self)
                if corridor_mixed_runtime.enabled(self):
                    corridor_mixed_runtime.update(self)
                if getattr(self.cfg,'beamdojo_repo_aligned',False):
                    repo_mdp.state(self)
                return original_compute(dt=dt)
            self.reward_manager.compute=compute_repo_rewards

    def step(self, action):
        if getattr(self.cfg,'beamdojo_upstream_aligned',False):
            # Source custom step clips, mixes previous processed action, adds
            # multiplicative Gaussian noise, then base step clips once more.
            action=action.clamp(-18.,18.)
            delay=torch.rand((self.num_envs,1),device=self.device)*.5
            action=(1.-delay)*action+delay*self.action_manager.action
            action=(action+.02*torch.randn_like(action)*action).clamp(-18.,18.)
        action = action_bounds.apply(self, action)
        observations, reward, terminated, truncated, extras = super().step(action)
        # Sim6 RewardManager stores weighted reward rates; reset() clears only
        # episode sums, so this includes the final pre-reset transition.
        index = self.reward_manager.active_terms.index('foothold')
        sparse = self.reward_manager._step_reward[:, index].clone() * self.step_dt
        if 'contact_event_risk' in self.reward_manager.active_terms:
            event_index = self.reward_manager.active_terms.index('contact_event_risk')
            sparse += self.reward_manager._step_reward[:, event_index] * self.step_dt
        extras['rewards_sparse'] = sparse
        extras['rewards_dense'] = reward - sparse
        extras['observation_source'] = self.cfg.observation_source
        if getattr(self.cfg,'beamdojo_paper_aligned',False) or upstream_terrain.enabled(self.cfg):
            extras.setdefault('log',{})['Curriculum/mean_level']=self.beamdojo_levels.float().mean()
            extras['log']['Curriculum/max_level']=self.beamdojo_levels.max().float()
            extras['log']['Curriculum/mean_streak']=self.beamdojo_success_streak.float().mean()
        if getattr(self.cfg, 'beamdojo_course_integrity', False):
            # Diagnostic buffers have already been updated before autoreset.
            extras.setdefault('log', {}).update(upstream_terrain.course_diagnostics(self))
        if getattr(self.cfg, 'beamdojo_course_tracking', False):
            extras.setdefault('log', {}).update(getattr(self, '_beamdojo_tracking_log', {}))
        if recovery_curriculum.enabled(self.cfg):
            extras.setdefault('log', {}).update(getattr(self, '_beamdojo_recovery_log', {}))
        if getattr(self.cfg, 'beamdojo_bound_joint_actions', False):
            extras.setdefault('log', {}).update(action_bounds.diagnostics(self))
        if getattr(self.cfg, 'beamdojo_contact_event_risk', False):
            extras.setdefault('log', {}).update(getattr(self, '_beamdojo_event_log', {}))
        if support_experiment_runtime.enabled(self):
            extras.setdefault('log', {}).update(getattr(self, '_beamdojo_support_log', {}))
        if corridor_runtime.enabled(self):
            extras['corridor_transition'] = getattr(self, '_beamdojo_corridor_transition', {})
            extras.setdefault('log', {}).update(getattr(self, '_beamdojo_corridor_log', {}))
        return observations, reward, terminated, truncated, extras

    def _reset_idx(self, env_ids):
        course_log={}
        if upstream_terrain.enabled(self.cfg):
            course_log=upstream_terrain.before_reset(self,env_ids)
        elif getattr(self.cfg,'beamdojo_paper_aligned',False):
            course_log=paper_mdp.before_reset(self,env_ids)
        super()._reset_idx(env_ids)
        if getattr(self.cfg, 'beamdojo_contact_event_risk', False):
            contact_event_risk.reset_state(self, env_ids)
        recovery_curriculum.reset_state(self, env_ids)
        if corridor_runtime.enabled(self):
            corridor_runtime.reset(self, env_ids)
        if getattr(self.cfg, 'beamdojo_course_tracking', False):
            # Command reset may relocate a Stage-2 spawn, so capture only after
            # the native physics/events/managers reset sequence has completed.
            course_tracking.reset_state(self, env_ids)
        if upstream_terrain.enabled(self.cfg):
            self._upstream_heightmap_cache = None
        if getattr(self.cfg, 'beamdojo_upstream_aligned', False):
            from .upstream_mdp import reset_observation_history
            reset_observation_history(self, env_ids)
        if getattr(self.cfg,'beamdojo_repo_aligned',False):
            repo_mdp.reset_state(self,env_ids)
        if hasattr(self,'extras'):
            log=self.extras.setdefault('log',{})
            if 'Metrics/success_rate' in log:
                log['Metrics/velocity_tracking_pass_rate']=log.pop('Metrics/success_rate')
            log.update(course_log)
        if hasattr(self, '_beamdojo_gait_cycle'):
            self._beamdojo_gait_cycle[env_ids] = 0.
        if hasattr(self, '_beamdojo_arrived'):
            self._beamdojo_arrived[env_ids] = False
        if hasattr(self, '_beamdojo_finish_time'):
            self._beamdojo_finish_time[env_ids] = 0.
        if hasattr(self, '_beamdojo_map'):
            self._beamdojo_map.reset(env_ids)
            self._beamdojo_map_noise.reset(env_ids)
            self._beamdojo_map_tick = -1
        if hasattr(self, '_beamdojo_lidar_cache'):
            self._beamdojo_lidar_cache.reset(env_ids)
            self._beamdojo_lidar_hold.reset(env_ids)
            self._beamdojo_lidar_tick = -1

    def snapshot_curriculum(self):
        upstream=upstream_terrain.enabled(self.cfg)
        if not (getattr(self.cfg,'beamdojo_paper_aligned',False) or upstream):
            return None
        (upstream_terrain if upstream else paper_mdp).ensure_state(self)
        integrity = getattr(self.cfg, 'beamdojo_course_integrity', False)
        result = {'version':2 if integrity else 1,'stage':self.cfg.beamdojo_stage,'num_envs':self.num_envs,
                'course_schedule':course_schedule.profile_name(self.cfg),
                'levels':self.beamdojo_levels.cpu().tolist(),
                'success_streak':self.beamdojo_success_streak.cpu().tolist()}
        if integrity:
            result['course_integrity_version'] = 1
        if getattr(self.cfg, 'beamdojo_unitree_motion_aligned', False):
            term = self.command_manager.get_term('base_velocity')
            result['command_ranges'] = {
                name: list(getattr(term.cfg.ranges, name))
                for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')}
        if recovery_curriculum.enabled(self.cfg):
            result['recovery'] = recovery_curriculum.snapshot(self)
        if getattr(self.cfg, 'beamdojo_bound_joint_actions', False):
            result['action_bounds'] = getattr(self, 'beamdojo_action_bounds_summary', None)
        if support_experiment_runtime.enabled(self):
            result['support_experiment'] = support_experiment_runtime.snapshot(self)
        return result

    def restore_command_curriculum(self, state):
        """Also restore across Stage 1 -> 2, without copying terrain levels."""
        if not getattr(self.cfg, 'beamdojo_unitree_motion_aligned', False):
            return
        if state is None or 'command_ranges' not in state:
            raise ValueError('Unitree-motion continuation requires saved command ranges')
        term = self.command_manager.get_term('base_velocity')
        for name in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
            values = state['command_ranges'][name]
            limits = getattr(term.cfg.limit_ranges, name)
            if (len(values) != 2 or not all(torch.isfinite(torch.tensor(v)) for v in values)
                    or not limits[0] - 1e-6 <= values[0] <= values[1] <= limits[1] + 1e-6):
                raise ValueError(f'Invalid saved command range: {name}')
            setattr(term.cfg.ranges, name, tuple(values))

    def restore_curriculum(self,state, *, migrate_course=False, source_schedule=None, demote_levels=False):
        if state is None:
            return
        target_schedule = course_schedule.profile_name(self.cfg)
        saved_schedule = state.get('course_schedule', course_schedule.profile_name(None))
        if source_schedule is not None:
            state = recovery_curriculum.remap_curriculum_schedule(state, source_schedule, target_schedule)
        elif course_schedule.profile_name(saved_schedule) != target_schedule:
            raise ValueError('Schedule migration requires explicit source_schedule')
        upstream=upstream_terrain.enabled(self.cfg)
        if not (getattr(self.cfg,'beamdojo_paper_aligned',False) or upstream):
            raise ValueError('Cannot restore paper curriculum into pilot profile')
        integrity = getattr(self.cfg, 'beamdojo_course_integrity', False)
        expected_version = 2 if integrity else 1
        migration = integrity and migrate_course and state.get('version') == 1
        if ((state.get('version') != expected_version and not migration)
                or state.get('stage')!=self.cfg.beamdojo_stage or state.get('num_envs')!=self.num_envs):
            raise ValueError('Curriculum stage/version/environment count mismatch')
        if integrity and not migration and state.get('course_integrity_version') != 1:
            raise ValueError('Course-integrity curriculum version mismatch')
        self.restore_command_curriculum(state)
        (upstream_terrain if upstream else paper_mdp).ensure_state(self)
        levels=torch.as_tensor(state['levels'],device=self.device,dtype=torch.long)
        streak=torch.as_tensor(state['success_streak'],device=self.device,dtype=torch.long)
        if levels.shape!=(self.num_envs,) or streak.shape!=(self.num_envs,) or ((levels<0)|(levels>course_schedule.max_level(self.cfg))).any() or ((streak<0)|(streak>3)).any():
            raise ValueError('Invalid curriculum levels/streak')
        self.beamdojo_levels[:]=levels
        self.beamdojo_success_streak[:]=0 if migration else streak
        recovery_curriculum.restore(self, state.get('recovery'))
        if support_experiment_runtime.enabled(self):
            support_experiment_runtime.restore(self, state.get('support_experiment'))
            support_experiment_runtime.assign_levels(self)
            self.cfg.recovery_easy_replay_fraction = 0.
            self.cfg.recovery_challenge_fraction = 0.
        demotion = recovery_curriculum.demote_once(self) if demote_levels else None
        if upstream:
            self._upstream_skip_curriculum_once=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
            upstream_terrain.refresh_geometry(self)
            return demotion
        self._paper_skip_curriculum_once=torch.ones(self.num_envs,device=self.device,dtype=torch.bool)
        paper_mdp.refresh_geometry(self)
        return demotion
