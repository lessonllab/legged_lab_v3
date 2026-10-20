"""Extend the existing local gait diagnostic report with the corrective run."""
import argparse
import html
import json
from pathlib import Path


NAMES={'14700':'原始14700','candidate15700':'第一阶段15700','corrected16700':'修正后16700',
       'no_arms':'取消新增手臂修正','leg_pitch_roll':'只保留腿部俯仰/侧倾修正'}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    def read(p):return json.loads(p.read_text())
    manifest=read(root/'experiment_manifest.json');cfg=read(root/'pipeline_config.json')
    training=read(Path(cfg['training_status'])) if Path(cfg['training_status']).exists() else {}
    conclusion=manifest.get('conclusion_zh','训练与对照尚在进行，下面只展示已完成的数据。')
    def number(value):return '—' if value is None else f'{value:.3f}'
    panels=[];video_choices=[];aggregates={}
    folders=[('diagnosis_no_arms','冻结模型干预：取消新增手臂修正'),
             ('diagnosis_leg_pitch_roll','冻结模型干预：只保留腿部俯仰/侧倾修正')]
    for scope in ('saved','fresh'):
        for label in ('14700','candidate15700'):
            folders.append((f'{label}_vs_corrected_{scope}',
                ('已有地图' if scope=='saved' else f"新地图种子{cfg['fresh_geometry_seed']}")+'：'+NAMES[label]+'与修正版'))
    metric_names=['到达外环','稳定停车1秒','失败回合','整段过桩平均绝对偏航（°）','终止时绝对偏航（°）',
        '身体晃动（°）','左臂与右脚步频范围相关性','右臂与左脚步频范围相关性',
        '同一只脚两次落地前进距离（m）','左右相邻触地间隔（s）','膝关节幅度（°）',
        '手臂关节速度RMS（rad/s）','左右交替触地比例','双脚腾空帧占比']
    for folder_name,title in folders:
        folder=root/folder_name
        if not all((folder/n).exists() for n in ['results.json','arm_audit.json','support_transition_audit.json']):continue
        result=read(folder/'results.json');arms=read(folder/'arm_audit.json');support=read(folder/'support_transition_audit.json')
        labels=list(result['results']);columns=[];numbers={}
        for label in labels:
            counts=result['results'][label]['stone_overall']
            a=arms['models'][label]['summary']['course_stone_crossing'];s=support['models'][label]['groups']['stones']
            values=[s.get('crossing_heading_abs_deg'),s.get('final_heading_abs_deg'),a.get('torso_sway_deg'),
                a.get('left_arm_right_foot_stride_corr'),a.get('right_arm_left_foot_stride_corr'),
                s.get('same_foot_stride_forward_m'),s.get('touchdown_interval_s'),s.get('knee_excursion_deg'),
                a.get('arm_speed_rms_rads'),s.get('alternation_fraction'),a.get('flight_fraction')]
            columns.append([f"{counts['arrivals']}/{counts['cases']}",f"{counts['stable_parks']}/{counts['cases']}",
                str(counts['failed'])]+[number(value) for value in values])
            numbers[label]=dict(cases=counts['cases'],arrivals=counts['arrivals'],stable_parks=counts['stable_parks'],
                failed=counts['failed'],**dict(zip(metric_names[3:],values)))
        if not folder_name.startswith('diagnosis'):
            reference=labels[0];aggregates.setdefault(reference,{})[folder_name.rsplit('_',1)[1]]=numbers
        rows=''.join('<tr><td>'+name+'</td>'+''.join('<td>'+column[i]+'</td>' for column in columns)+'</tr>'
                     for i,name in enumerate(metric_names))
        note='冻结同一15700网络，只改变附加修正的关节范围；没有训练更新。只保留腿部修正时偏航明显变差，因此未采用该方案续训。' if folder_name.startswith('diagnosis') else '同一初始物理状态、共享质量与摩擦、无观测噪声/推力；所有桩顶等高。每回合等权，不能用摆臂变小单独判断自然程度。'
        panels.append('<section><h2>'+title+'</h2><table><tr><th>指标</th>'+''.join('<th>'+NAMES.get(label,label)+'</th>' for label in labels)+'</tr>'+rows+'</table><p class="note">'+note+'</p>'+
            f'<p><a href="{folder_name}/results.json">通行与平地结果</a> · <a href="{folder_name}/arm_audit.json">摆臂数据</a> · <a href="{folder_name}/support_transition_audit.json">落脚数据</a></p></section>')
        if not folder_name.startswith('diagnosis'):
            flat_rows=[]
            for mode,title in [('forward_1.0','前进1m/s'),('backward','后退0.5m/s'),('left','左移0.3m/s'),
                ('right','右移0.3m/s'),('turn_left','左转0.2rad/s'),('turn_right','右转0.2rad/s'),('stand','站立')]:
                cells=[]
                for label in labels:
                    item=result['results'][label]['by_flat_mode'][mode];m=item['metrics']
                    cells.append(f"失败{item['failed']}/{item['cases']}；平移误差{number(m.get('xy_velocity_error_mps'))}m/s；转速误差{number(m.get('yaw_rate_error_rads'))}rad/s")
                flat_rows.append('<tr><td>'+title+'</td>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
            panels.append('<details><summary>查看该组平地指令对照</summary><table>'+''.join(flat_rows)+'</table></details>')
        for replay_folder in ('replays','replays_fullcross_regressions'):
            if not (folder/replay_folder/'replay_manifest.json').exists():continue
            videos=read(folder/replay_folder/'replay_manifest.json')['videos']
            for item in videos:
                case=item['case'];name=case['mode'] if case['kind']=='flat' else f"{'正向' if case['family']==0 else '斜向'} 路线{case['route']} L{case['level']} 变体{case['variant']}"
                if replay_folder!='replays':name+=' · 整段退步样本'
                video_choices.append(dict(label=title+' / '+name,src=f"{folder_name}/{replay_folder}/{Path(item['video']).name}",
                    poster=f"{folder_name}/{replay_folder}/{Path(item['preview']).name}"))
    summary=dict(manifest=manifest,training_status=training,paired_results=aggregates)
    if (root/'aggregate_results.json').exists():summary['overall_results']=read(root/'aggregate_results.json')
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    progress=f"{training.get('completed_updates',0)}/1000轮"
    if training.get('state')=='budget_complete':progress='1000轮已完成，终点16700轮'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>支撑转换第二阶段：偏航与摆臂修正</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#232a30;background:#fafafa;max-width:1120px;margin:32px auto;padding:0 20px}h1{font-size:25px}h2{font-size:20px;margin-top:28px}.note{color:#58636b;font-size:14px}video{width:100%;display:block}select{font:inherit;padding:10px;max-width:100%;margin:12px 0}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf1f3}a{color:#245f82}section{margin-top:30px}
</style><nav><a href="../support_transition_stage1/">第一阶段原始对照</a></nav>
<h1>支撑转换第二阶段：偏航与摆臂修正</h1>'''
    page+='<p><strong>'+html.escape(progress)+'</strong> · 4096环境，一半平地、一半桩路。</p><p>'+html.escape(conclusion)+'</p>'
    if manifest.get('live_play',{}).get('verified'):
        page+='<p><a href="'+html.escape(manifest['live_play']['url'])+'">实时IsaacLab播放与指令调节</a></p>'
    page+='<p>从15700继续：原策略动态控制全身，取消新增手臂修正，保留腰髋平衡能力；分开预测与控制梯度，轻度抑制持续修正偏置。运动与过桩奖励沿用原设置。</p>'
    calibration=manifest.get('repeat_calibration',{}).get('metrics',{}).get('crossing_heading_abs_deg')
    if calibration:
        page+='<p class="note">重复测量参考：同一14700的整段偏航均值为'+number(calibration['mean_first'])+'° / '+number(calibration['mean_repeat'])+'°，单回合绝对差平均'+number(calibration['mean_abs_case_difference'])+'°。仅一个重复对，用于了解复位/内部状态/数值波动，不是训练种子方差或置信区间。<a href="repeat_calibration_summary.json">查看复测数据</a>。</p>'
    if video_choices:
        initial=next((i for i,v in enumerate(video_choices) if '修正版' in v['label'] and '正向 路线0 L10 变体7' in v['label']),0)
        options=''.join(f'<option value="{i}"'+(' selected' if i==initial else '')+'>'+html.escape(item['label'])+'</option>' for i,item in enumerate(video_choices))
        first=video_choices[initial]
        page+=f'<h2>同场景动作回放</h2><select id="scene">{options}</select><video id="video" controls muted playsinline loop preload="metadata" src="{first["src"]}" poster="{first["poster"]}"></video>'
        page+='<p class="note">左右模型名称标在画面中。录制的真实Isaac轨迹，MuJoCo只渲染姿态；包括固定场景及相对退步明显的样本。</p>'
        page+='<script>const items='+json.dumps(video_choices,ensure_ascii=False)+';const v=document.getElementById("video");document.getElementById("scene").onchange=e=>{const x=items[Number(e.target.value)];v.src=x.src;v.poster=x.poster;v.load();v.play().catch(()=>{});};</script>'
    page+=''.join(panels)+'<p class="note">旧1729地图已用于上一阶段诊断；本阶段新测试种子为20261016，沿用相同地形生成类型。仅一个训练种子；不代表任意地形、高低混合桩、真实传感器或实机表现。首次测试的预测干扰仍是结构性假设，尚无单模块消融结论。</p><p><a href="experiment_manifest.json">实验定义</a> · <a href="pretrain_action_diagnosis.json">训练前关节修正统计</a></p></html>'
    (root/'index.html').write_text(page)
    print(root/'index.html')


if __name__=='__main__':main()
