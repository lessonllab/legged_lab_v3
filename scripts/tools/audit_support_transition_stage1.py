"""Native rollout audit; predicts observed next contacts, never claims viability."""
import argparse
import json
from pathlib import Path
import numpy as np


def mean(values):
    return float(np.mean(values)) if len(values) else None


def events(trace, cid):
    contact=trace['contact'][:,cid]
    active=trace['active'][:,cid]
    air=np.zeros(2,dtype=int);last=np.zeros(2,dtype=int);found=[]
    for i in np.flatnonzero(active):
        for foot in (0,1):
            if contact[i,foot] and air[foot]>=3:
                found.append(dict(frame=int(i),foot=foot,previous_tick=int(last[foot])))
                last[foot]=i+1
        air=np.where(contact[i],0,air+1)
    return found


def analyze(trace, rows, body_names):
    feet=[body_names.index(s+'_ankle_roll_link') for s in ('left','right')]
    names=list(trace['joint_names'])
    knees=[i for i,n in enumerate(names) if 'knee_' in n]
    summaries=[];prediction=[]
    for row in rows:
        cid=row['case_id'];time=trace['time'][:,cid]
        # Complete first episodes only; no successful-frame filtering.
        active=trace['active'][:,cid];idx=np.flatnonzero(active)
        entry=4. if row['family']==0 else 4.*np.sqrt(2.)-.35355339059
        crossing=active & (time>=.5)
        if row['kind']=='stone':
            crossing &= (trace['root'][:,cid,0]>=.9)&(trace['root'][:,cid,0]<entry)
        else:crossing &= time<=3.
        event_list=events(trace,cid)
        cross_events=[e for e in event_list if crossing[e['frame']]]
        contacts=trace['contact'][crossing,cid]
        supported=trace['full_foot_support'][crossing,cid]
        values=dict(case_id=cid,kind=row['kind'],mode=row['mode'],level=row['level'],variant=row['variant'],
            crossing_frames=int(crossing.sum()),touchdowns=len(cross_events),
            alternation_fraction=mean([float(a['foot']!=b['foot'] and a['frame']!=b['frame'])
                for a,b in zip(cross_events,cross_events[1:])]),
            touchdown_full_support_fraction=mean([float(trace['full_foot_support'][e['frame'],cid,e['foot']]) for e in cross_events]),
            contacting_full_support_fraction=float((contacts&supported).sum()/max(1,contacts.sum())),
            knee_excursion_deg=float(np.ptp(trace['joint_pos'][crossing,cid][:,knees],axis=0).mean()*180/np.pi) if crossing.any() else None)
        if 'heading_error_deg' in trace:
            values['crossing_heading_abs_deg']=float(np.abs(trace['heading_error_deg'][crossing,cid]).mean()) if crossing.any() else None
            values['final_heading_abs_deg']=float(abs(trace['heading_error_deg'][idx[-1],cid])) if len(idx) else None
            values['crossing_lateral_abs_m']=float(np.abs(trace['root'][crossing,cid,1]).mean()) if crossing.any() else None
        if 'foot_centers' in trace:
            angle=np.deg2rad(((0,90,180,-90),(45,135,-135,-45))[row['family']][row.get('route',0)])
            direction=np.array([np.cos(angle),np.sin(angle)])
            strides=[]
            for foot in (0,1):
                own=[e['frame'] for e in cross_events if e['foot']==foot]
                strides.extend([float((trace['foot_centers'][b,cid,foot,:2]-trace['foot_centers'][a,cid,foot,:2])@direction) for a,b in zip(own,own[1:])])
            values['same_foot_stride_forward_m']=mean(strides)
            values['touchdown_interval_s']=mean([float(time[b['frame']]-time[a['frame']]) for a,b in zip(cross_events,cross_events[1:])])
        if 'transition_delta_rad' in trace and crossing.any():
            delta=trace['transition_delta_rad'][crossing,cid]
            values.update(residual_rms_rad=float(np.sqrt(np.mean(delta**2))),
                residual_abs_max_rad=float(np.abs(delta).max()),
                active_fraction=float(trace['transition_active'][crossing,cid].mean()))
        summaries.append(values)
        if 'transition_handoff' not in trace:continue
        for e in event_list:
            i,f=e['frame'],e['foot'];future=i+6
            if future>=len(active) or not active[future]:continue
            if any(v['foot']==f and i<v['frame']<future for v in event_list):continue
            for lag in (4,8,12,16,20,24):
                a=i+1-lag
                if a<0 or a<=e['previous_tick'] or not active[a]:continue
                q=trace['transition_prefix_quat'][a,cid]
                yaw=np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
                xy=trace['link_local'][i,cid,feet[f],:2]-trace['transition_prefix_root'][a,cid,:2]
                xy=np.array([np.cos(yaw)*xy[0]+np.sin(yaw)*xy[1],-np.sin(yaw)*xy[0]+np.cos(yaw)*xy[1]])
                visible=bool(np.isfinite(xy).all() and (np.abs(xy)<=.7).all())
                grid=np.clip(np.rint((xy+.7)/.1),0,14).astype(int);cell=int(grid[0]*15+grid[1])
                pred_cell=int(trace['transition_region_cell'][a,cid,f]);pred_xy=np.array([pred_cell//15,pred_cell%15])*.1-.7
                target=np.r_[lag*.02/.5,trace['body_velocity'][future,cid,:2],
                    trace['gravity'][future,cid,:2]/.3,trace['body_omega'][future,cid,2],
                    max(0,trace['contact_fz'][future,cid,f])/max(1,np.maximum(trace['contact_fz'][future,cid],0).sum())]
                pred=trace['transition_handoff'][a,cid,f]
                prediction.append(dict(case_id=cid,kind=row['kind'],gated=bool(trace['transition_active'][a,cid]),
                    visible=visible,cell_correct=float(cell==pred_cell),location_error_m=float(np.linalg.norm(pred_xy-xy)),
                    time_absolute_error_s=float(abs(pred[0]-target[0])*.5),
                    handoff_velocity_error_mps=float(np.linalg.norm(pred[1:3]-target[1:3])),
                    handoff_tilt_component_error=float(np.linalg.norm(pred[3:5]-target[3:5])*.3),
                    handoff_yaw_rate_error_rads=float(abs(pred[5]-target[5])),
                    handoff_load_share_error=float(abs(pred[6]-target[6]))))
    result=dict(cases=summaries,prediction_label_count=len(prediction),groups={})
    for name,selected in [('stones',[r for r in summaries if r['kind']=='stone']),('flat',[r for r in summaries if r['kind']=='flat'])]:
        result['groups'][name]={k:mean([r[k] for r in selected if r.get(k) is not None]) for k in
            ('alternation_fraction','touchdown_full_support_fraction','contacting_full_support_fraction',
             'knee_excursion_deg','residual_rms_rad','residual_abs_max_rad','active_fraction')}
        for key in ('crossing_heading_abs_deg','final_heading_abs_deg','crossing_lateral_abs_m','same_foot_stride_forward_m','touchdown_interval_s'):
            result['groups'][name][key]=mean([r[key] for r in selected if r.get(key) is not None])
    if prediction:
        result['prediction']={}
        for name,selected in [('all',prediction),('stones_gated',[r for r in prediction if r['kind']=='stone' and r['gated']])]:
            record=dict(labels=len(selected),visible_fraction=mean([r['visible'] for r in selected]))
            for key in ('cell_correct','location_error_m','time_absolute_error_s','handoff_velocity_error_mps',
                        'handoff_tilt_component_error','handoff_yaw_rate_error_rads','handoff_load_share_error'):
                use=[r for r in selected if r['visible']] if key in ('cell_correct','location_error_m') else selected
                record[key]=mean([r[key] for r in use])
            result['prediction'][name]=record
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.input/'results.json').read_text())
    with np.load(a.input/'body_properties.npz',allow_pickle=False) as body:bodies=list(body['body_names'])
    output=dict(protocol=result['protocol'],
        support_measure='All sampled sole points within3.5cm of collision top; conservative diagnostic, not an added reward/termination',
        prediction_scope='First next foot contact after3air frames; observed120ms later in same active episode; no post-reset labels; no viability claims',
        units='Next-contact location is a15x15 cell in prefix root-yaw frame at10cm resolution; time and state are direct predictions',models={})
    for label,meta in result['results'].items():
        with np.load(a.input/f'traces_{label}.npz',allow_pickle=False) as f:trace={k:f[k] for k in f.files}
        output['models'][label]=analyze(trace,meta['episodes'],bodies)
    path=a.input/'support_transition_audit.json';path.write_text(json.dumps(output,indent=2,allow_nan=False))
    print(path)


if __name__=='__main__':main()
