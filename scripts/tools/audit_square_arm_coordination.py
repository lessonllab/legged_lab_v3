"""Offline arm/leg coordination and momentum audit of native Isaac traces.

No policy updates. Momentum cancellation is a decomposition of an observed
trajectory, NOT an experiment demonstrating causal benefit from using arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import periodogram, butter, sosfiltfilt


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 20 or min(a.std(), b.std()) < .005:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def measure(trace, row, body_names, section):
    cid = row['case_id']
    time = trace['time'][:, cid]
    mask = trace['active'][:, cid] & (time >= .5)
    command = trace['command'][:, cid]
    if section == 'first3':
        mask &= time <= 3.
    elif section == 'stone_crossing':
        # Purely spatial gates; do not select only the successful/stable frames.
        entry = 4. if row['family'] == 0 else 4.*np.sqrt(2.)-.35355339059
        mask &= (trace['root'][:, cid, 0] >= .9) & (trace['root'][:, cid, 0] < entry)
        mask &= np.abs(command[:, 0]) >= .25
    else:
        mask &= time <= 8.
    idx = np.flatnonzero(mask)
    if len(idx) < 30:
        return None
    # Correlation/differentiation requires one contiguous first-episode segment.
    if not np.all(np.diff(idx) == 1):
        raise ValueError('Non-contiguous audit interval')
    q = trace['root_quat'][idx, cid]
    rotation = Rotation.from_quat(q).as_matrix()
    loc = trace['link_local'][idx, cid]
    root = trace['root_local'][idx, cid]
    rel = np.einsum('tji,tbj->tbi', rotation, loc-root[:, None, :])
    def body(name):
        return body_names.index(name)
    hand = np.stack([rel[:, body(s+'_wrist_yaw_link'), 0]
                     - rel[:, body(s+'_shoulder_pitch_link'), 0] for s in ('left', 'right')], -1)
    foot = np.stack([rel[:, body(s+'_ankle_roll_link'), 0] for s in ('left', 'right')], -1)
    arm_bodies = [i for i, name in enumerate(body_names)
                  if any(part in name for part in ('shoulder', 'elbow', 'wrist', 'hand'))]
    leg_bodies = [i for i, name in enumerate(body_names)
                  if any(part in name for part in ('hip', 'knee', 'ankle'))]
    H = trace['body_momentum'][idx, cid]
    armH = H[:, arm_bodies].sum(1)
    legH = H[:, leg_bodies].sum(1)
    totalH = H.sum(1)
    restH = totalH-armH
    names = list(trace['joint_names'])
    arm_joints = [i for i, n in enumerate(names) if any(k in n for k in ('shoulder', 'elbow', 'wrist'))]
    shoulder = [i for i, n in enumerate(names) if 'shoulder' in n]
    velocity = trace['joint_vel'][idx, cid][:, arm_joints]
    dt = float(np.median(np.diff(time[idx])))
    filtered_hand = sosfiltfilt(butter(2, [.4, 2.5], btype='bandpass', fs=1./dt, output='sos'), hand, axis=0)
    filtered_foot = sosfiltfilt(butter(2, [.4, 2.5], btype='bandpass', fs=1./dt, output='sos'), foot, axis=0)
    frequency, power = periodogram(velocity, fs=1./dt, axis=0)
    high = power[frequency >= 3.].sum()/max(power.sum(), 1e-12)
    pose = trace['joint_pos'][idx, cid]
    torque = trace['applied_torque'][idx, cid][:, arm_joints]
    action = trace['processed_action'][idx, cid][:, arm_joints]
    contacts = trace['contact'][idx, cid]
    grav = trace['gravity'][idx, cid]
    roll = np.arctan2(-grav[:, 1], -grav[:, 2])
    pitch = np.arctan2(grav[:, 0], np.sqrt(grav[:, 1]**2+grav[:, 2]**2))
    values = dict(frames=len(idx), duration_s=float(len(idx)*dt),
        left_arm_right_foot_corr=correlation(hand[:, 0], foot[:, 1]),
        right_arm_left_foot_corr=correlation(hand[:, 1], foot[:, 0]),
        left_right_arm_corr=correlation(hand[:, 0], hand[:, 1]),
        left_right_foot_corr=correlation(foot[:, 0], foot[:, 1]),
        left_arm_right_foot_stride_corr=correlation(filtered_hand[:, 0], filtered_foot[:, 1]),
        right_arm_left_foot_stride_corr=correlation(filtered_hand[:, 1], filtered_foot[:, 0]),
        left_right_arm_stride_corr=correlation(filtered_hand[:, 0], filtered_hand[:, 1]),
        hand_sagittal_excursion_m=np.ptp(hand, axis=0).tolist(),
        arm_speed_rms_rads=rms(velocity),
        arm_acceleration_rms_rads2=rms(np.diff(velocity, axis=0)/dt),
        arm_velocity_power_above_3hz_fraction=float(high),
        arm_mechanical_abs_power_w=float(np.abs(torque*velocity).sum(-1).mean()),
        arm_raw_action_increment_rms=rms(np.diff(action, axis=0)),
        shoulder_excursion_deg=float(np.ptp(pose[:, shoulder], axis=0).mean()*180/np.pi),
        arm_yaw_momentum_rms_kgm2s=rms(armH[:, 2]),
        legs_yaw_momentum_rms_kgm2s=rms(legH[:, 2]),
        total_yaw_momentum_rms_kgm2s=rms(totalH[:, 2]),
        nonarm_yaw_momentum_rms_kgm2s=rms(restH[:, 2]),
        yaw_momentum_ratio_total_to_nonarm=rms(totalH[:, 2])/max(rms(restH[:, 2]), 1e-9),
        yaw_momentum_cross_term=float(np.mean(2*restH[:, 2]*armH[:, 2])),
        arm_legs_yaw_momentum_corr=correlation(armH[:, 2], legH[:, 2]),
        arm_nonarm_yaw_momentum_corr=correlation(armH[:, 2], restH[:, 2]),
        torso_sway_deg=float(np.sqrt(roll.var()+pitch.var())*180/np.pi),
        tilt_rms_deg=float(np.sqrt(np.mean(np.arccos(np.clip(-grav[:, 2], -1, 1))**2))*180/np.pi),
        double_support_fraction=float(contacts.all(-1).mean()),
        flight_fraction=float((contacts.sum(-1)==0).mean()),
        velocity_error_mps=float(np.linalg.norm(trace['body_velocity'][idx, cid, :2]-command[idx, :2], axis=-1).mean()),
        arm_joint_mean_deg={name:float(pose[:, i].mean()*180/np.pi) for i, name in enumerate(names) if i in arm_joints})
    return values


def summarize(rows):
    keys = list(rows[0]) if rows else []
    out = dict(cases=len(rows))
    for key in keys:
        valid = [r[key] for r in rows if isinstance(r[key], (float, int)) and r[key] is not None]
        if valid:
            out[key] = float(np.mean(valid))
            if key.endswith('_corr'):
                out[key+'_valid_cases'] = len(valid)
    return out


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    args = p.parse_args()
    root = args.input
    result = json.loads((root/'results.json').read_text())
    with np.load(root/'body_properties.npz') as f:
        bodies = list(f['body_names'])
    output = dict(protocol=result['protocol'],
        momentum='Exact native COM, link velocities and COM inertias rotated from link to world; per-link H about total instantaneous COM. Ratio total/nonarm <1 means instantaneous cancellation, NOT an arm immobilization experiment.',
        coordination='Pearson correlation between wrist-minus-shoulder fore-aft displacement and contralateral foot-minus-pelvis fore-aft displacement, in instantaneous root frame; +1 is coordinated opposition during forward walking. No score assigned to near-static signals.',
        conditions='Paired deterministic rollout, no observation/map noise or pushes. Map scope and seed are recorded in protocol.',
        models={})
    for label, metadata in result['results'].items():
        with np.load(root/f'traces_{label}.npz') as f:
            trace = {k:f[k] for k in f.files}
        rows=[]
        for row in metadata['episodes']:
            sections = ['first3', 'stone_crossing'] if row['kind']=='stone' else ['first3', 'walking8']
            for section in sections:
                measurements=measure(trace, row, bodies, section)
                if measurements:
                    rows.append(dict(case_id=row['case_id'], kind=row['kind'], mode=row['mode'],
                        family=row['family'], variant=row['variant'], level=row['level'],
                        section=section, metrics=measurements))
        summaries={}
        for mode in ('forward_0.5', 'forward_1.0', 'backward', 'course', 'stand'):
            for section in ('first3', 'walking8', 'stone_crossing'):
                selected=[r['metrics'] for r in rows if r['mode']==mode and r['section']==section]
                if selected:summaries[mode+'_'+section]=summarize(selected)
        output['models'][label]=dict(summary=summaries, cases=rows)
    (root/'arm_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    for label, data in output['models'].items():
        print(label)
        for key in ('forward_1.0_walking8', 'course_stone_crossing', 'backward_walking8'):
            s=data['summary'][key]
            print(key, {k:round(s[k], 4) for k in (
                'left_arm_right_foot_corr', 'right_arm_left_foot_corr', 'left_right_arm_corr',
                'yaw_momentum_ratio_total_to_nonarm', 'arm_nonarm_yaw_momentum_corr',
                'arm_speed_rms_rads', 'arm_acceleration_rms_rads2', 'arm_mechanical_abs_power_w',
                'torso_sway_deg') if k in s})


if __name__ == '__main__':
    main()
