"""Case-weighted completed-pair summaries; no best-case model selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(folder):
    def read(name):return json.loads((folder/name).read_text())
    result=read('results.json');arm=read('arm_audit.json');support=read('support_transition_audit.json')
    models={}
    for label,value in result['results'].items():
        a=[row['metrics'] for row in arm['models'][label]['cases'] if row['kind']=='stone' and row['section']=='stone_crossing']
        s=[row for row in support['models'][label]['cases'] if row['kind']=='stone']
        metrics={}
        for records,keys in [(a,['torso_sway_deg','left_arm_right_foot_stride_corr','right_arm_left_foot_stride_corr',
            'arm_speed_rms_rads','flight_fraction','yaw_momentum_ratio_total_to_nonarm']),
            (s,['crossing_heading_abs_deg','final_heading_abs_deg','crossing_lateral_abs_m',
                'same_foot_stride_forward_m','touchdown_interval_s','knee_excursion_deg','alternation_fraction'])]:
            for key in keys:
                numbers=[row[key] for row in records if row.get(key) is not None]
                if not all(math.isfinite(x) for x in numbers):raise ValueError('Nonfinite '+key)
                metrics[key]=dict(sum=sum(numbers),observed_cases=len(numbers))
        counts={key:value['stone_overall'][key] for key in ['cases','arrivals','stable_parks','failed']}
        models[label]=dict(counts=counts,metrics=metrics)
    return dict(geometry_seed=result['geometry_seed'],support_raster_sha256=result['actual_support_raster_sha256'],models=models)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    cfg=json.loads((root/'pipeline_config.json').read_text());groups={}
    for reference in ['14700','candidate15700']:
        scopes={scope:summarize(root/f'{reference}_vs_corrected_{scope}') for scope in ['saved','fresh']}
        assert scopes['fresh']['geometry_seed']==cfg['fresh_geometry_seed']
        assert scopes['saved']['support_raster_sha256']!=scopes['fresh']['support_raster_sha256']
        aggregate={}
        for label in [reference,'corrected16700']:
            values=[scopes[scope]['models'][label] for scope in ['saved','fresh']]
            counts={key:sum(v['counts'][key] for v in values) for key in values[0]['counts']}
            metrics={}
            for key in values[0]['metrics']:
                total=sum(v['metrics'][key]['sum'] for v in values)
                n=sum(v['metrics'][key]['observed_cases'] for v in values)
                metrics[key]=dict(mean=total/n if n else None,observed_cases=n)
            aggregate[label]=dict(counts=counts,metrics=metrics)
        groups[reference]=dict(scopes=scopes,aggregate=aggregate)
    status=json.loads(Path(cfg['training_status']).read_text())
    if status['state']!='budget_complete' or status['completed_updates']!=1000:raise ValueError('Training incomplete')
    checkpoint=Path(status['run_dir'])/'model_16700.pt'
    record=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        comparisons=groups,
        scope='Each comparison has64saved+128fresh stone cases; fresh terrain generator seed20261016. Same candidate rerun for each reference, not384independent maps. Missing metrics excluded with explicit observed-case counts; failure counts retain every episode.')
    (root/'aggregate_results.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    for reference,group in groups.items():
        for label,v in group['aggregate'].items():
            print(reference,label,v['counts'],{key:round(m['mean'],4) if m['mean'] is not None else None for key,m in v['metrics'].items()})


if __name__=='__main__':main()
