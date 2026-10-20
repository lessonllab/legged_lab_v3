"""Render matched native Isaac traces, including the actual held-out geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageFont
import render_square_effects_replay as replay


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--case-ids',type=int,nargs='+');p.add_argument('--duration',type=float,default=14.)
    a=p.parse_args();root=a.input.resolve(strict=True);out=(a.output or root/'replays').resolve();out.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/'results.json').read_text());replay.LABELS=tuple(data['results'])
    if len(replay.LABELS)!=2:raise ValueError('Expected two paired models')
    reference=Path(data['common_environment_checkpoint']);contract=replay.read_json(reference.parent/'contract.json')
    profile=replay.read_json(replay.PROFILE)
    if profile['sha256']!=contract['asset_sha256']:raise ValueError('Visual robot mismatch')
    modules=replay.archived_geometry(reference.parent/'params')
    rows=list(data['results'][replay.LABELS[0]]['episodes'])
    if a.case_ids is None:
        selected=[]
        for mode in ('forward_1.0','backward','turn_left'):
            selected.append(next(r for r in rows if r['kind']=='flat' and r['mode']==mode and r['replica']==0))
        for family in (0,1):
            selected.append(next(r for r in rows if r['kind']=='stone' and r['family']==family and r['route']==0 and r['level']==10 and r['variant']==7 and r['replica']==0))
        regular=[r for r in rows if r['kind']=='stone' and r['family']==0 and r['route']==0 and r['level']==10 and r['variant']==0 and r['replica']==0]
        selected+=regular
        candidate_rows={r['case_id']:r for r in data['results'][replay.LABELS[1]]['episodes']}
        def regression_order(row):
            other=candidate_rows[row['case_id']];old=row['metrics_0p5_to_3s'];new=other['metrics_0p5_to_3s']
            movement_delta=(new['torso_tilt_rms_deg']-old['torso_tilt_rms_deg']+.1*(new['mean_abs_heading_error_deg']-old['mean_abs_heading_error_deg'])) if old and new else 0.
            return (int(row['reached'] and not other['reached']),int(row['stable_park'] and not other['stable_park']),movement_delta)
        worst=max((r for r in rows if r['kind']=='stone'),key=regression_order)
        if worst['case_id'] not in [r['case_id'] for r in selected]:selected.append(worst)
    else:selected=[next(r for r in rows if r['case_id']==cid) for cid in a.case_ids]
    traces={};episodes={}
    fields=('active','time','root_local','root_quat','joint_pos','heading_error_deg','body_velocity','command','joint_names')
    for label,meta in data['results'].items():
        with np.load(root/f'traces_{label}.npz',allow_pickle=False) as f:traces[label]={k:f[k] for k in fields}
        episodes[label]={r['case_id']:r for r in meta['episodes']}
    fonts={name:ImageFont.truetype(replay.FONT_PATH,size) for name,size in [('title',21),('body',16),('small',15)]}
    reports=[];previews=[]
    specs=data.get('actual_geometry_specs',contract['square_mixed']['specs'])
    for row in selected:
        cid=row['case_id'];provenance=None
        for label in replay.LABELS:
            other=episodes[label][cid]
            if any(row[k]!=other[k] for k in ('kind','family','route','level','variant','replica','column')):
                raise ValueError('Mismatched video cases')
        if row['kind']=='flat':boxes=None;title='Flat '+row['mode'];seconds=min(a.duration,6.)
        else:
            seed,family,variant=specs[row['column']]
            if (family,variant)!=(row['family'],row['variant']):raise ValueError('Geometry column mismatch')
            geometry=modules['beamdojo_square_extended_geometry'];name=contract['square_mixed']['families'][family]
            course=geometry.mixed_layout(seed,row['level']/10.,name,variant);raster=geometry.rasterize(course)
            grid=raster['collision_height_m'];meshes,origin=geometry.grid_mesh(grid)
            boxes=[mesh.bounds-origin for mesh in meshes]
            provenance=dict(seed=seed,family=name,variant=variant,level=row['level'],
                raster_sha256=hashlib.sha256(grid.tobytes()).hexdigest())
            title=f"{name} R{row['route']} L{row['level']} V{variant}";seconds=a.duration
        model=replay.rendering_model(boxes,out/f'case_{cid}.scene.xml',profile)
        report,preview=replay.render_video(model,traces,episodes,(f'case_{cid}',title,row),out,seconds,fonts)
        report['geometry']=provenance;reports.append(report);previews.append(preview)
    sheet=Image.new('RGB',(replay.PANEL_WIDTH*2,replay.PANEL_HEIGHT*len(previews)))
    for i,preview in enumerate(previews):sheet.paste(preview,(0,i*replay.PANEL_HEIGHT))
    sheet.save(out/'preview_contactsheet.jpg',quality=94)
    (out/'replay_manifest.json').write_text(json.dumps(dict(source='Actual recorded native Isaac rollout poses',
        physics_steps=0,renderer='MuJoCo forward kinematics and rasterization only; not sim2sim evidence',
        protocol=data['protocol'],videos=reports),indent=2,allow_nan=False))


if __name__=='__main__':main()
