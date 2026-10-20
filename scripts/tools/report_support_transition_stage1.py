"""Extend the local policy diagnostic artifact with the completed stage-one run."""
import argparse
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--partial',action='store_true',help='Show completed paired evaluations while the remaining jobs run.')
    a=p.parse_args()
    root=a.root.resolve(strict=True);panels=[];video_choices=[]
    manifest=json.loads((root/'experiment_manifest.json').read_text())
    def read(path):return json.loads(path.read_text())
    def number(x):return '—' if x is None else f'{x:.3f}'
    names={'14700':'原始14700','ordinary15700':'普通续训15700','candidate15700':'新接口15700'}
    for scope in ('saved','holdout'):
        for reference in ('14700','ordinary15700'):
            folder=root/f'{reference}_vs_candidate_{scope}'
            required=('results.json','arm_audit.json','support_transition_audit.json','replays/replay_manifest.json')
            if a.partial and not all((folder/name).is_file() for name in required):continue
            result=read(folder/'results.json');arms=read(folder/'arm_audit.json')
            support=read(folder/'support_transition_audit.json');replays=read(folder/'replays/replay_manifest.json')
            labels=list(result['results']);cols=[]
            for label in labels:
                row=result['results'][label]['stone_overall'];m=row['metrics']
                arm=arms['models'][label]['summary']['course_stone_crossing'];sup=support['models'][label]['groups']['stones']
                cols.append([f"{row['arrivals']}/{row['cases']}",f"{row['stable_parks']}/{row['cases']}",str(row['failed']),
                    number(m['mean_abs_heading_error_deg']),number(m['xy_velocity_error_mps']),
                    number(sup.get('crossing_heading_abs_deg')),number(sup.get('final_heading_abs_deg')),
                    number(sup.get('same_foot_stride_forward_m')),number(sup.get('touchdown_interval_s')),
                    number(m['knee_excursion_deg']),number(arm['torso_sway_deg']),
                    number(arm['arm_speed_rms_rads']),number(arm['arm_acceleration_rms_rads2']),
                    number(arm.get('left_arm_right_foot_stride_corr')),number(arm.get('right_arm_left_foot_stride_corr')),
                    number(sup['alternation_fraction']),number(sup['contacting_full_support_fraction'])])
            row_names=['到达外环','连续稳停1秒','失败回合','平均绝对偏航（°，0.5–3秒）','速度误差（m/s，0.5–3秒）',
                '整段过桩平均绝对偏航（°）','回合终止时绝对偏航（°）',
                '同一只脚两次落地前进距离（m）','左右相邻触地时间间隔（s）',
                '膝关节幅度（°，0.5–3秒）','过桩身体晃动（°）','过桩手臂速度RMS（rad/s）','过桩手臂加速度RMS（rad/s²）',
                '左臂与右脚步频范围内相关性','右臂与左脚步频范围内相关性',
                '过桩左右交替触地比例','承重期间完整脚底采样支持比例']
            body=''.join('<tr><td>'+html.escape(title)+'</td>'+''.join('<td>'+c[i]+'</td>' for c in cols)+'</tr>' for i,title in enumerate(row_names))
            scope_title='已有地图' if scope=='saved' else '未见地图种子1729'
            title=f'{scope_title}：{names[reference]} 与新接口'
            panels.append('<section><h2>'+title+'</h2><table><thead><tr><th>指标</th>'+''.join('<th>'+names[x]+'</th>' for x in labels)+'</tr></thead><tbody>'+body+'</tbody></table>'+
                '<p class="note">每项按回合等权平均。膝关节幅度、摆臂幅度或支持比例不能单独判定步态优劣；需结合速度、失败及视频。图中所有桩高相同。</p>'+
                f'<p><a href="{folder.name}/results.json">通行与停车原始数据</a> · <a href="{folder.name}/arm_audit.json">摆臂与晃动</a> · <a href="{folder.name}/support_transition_audit.json">落脚与预测</a></p></section>')
            modes={'forward_0.5':'前进0.5m/s','forward_1.0':'前进1m/s','backward':'后退0.5m/s',
                   'left':'左移0.3m/s','right':'右移0.3m/s','turn_left':'左转0.2rad/s',
                   'turn_right':'右转0.2rad/s','stand':'站立'}
            flat_rows=[]
            for mode,title in modes.items():
                cells=[]
                for label in labels:
                    flat=result['results'][label]['by_flat_mode'][mode];metric=flat['metrics']
                    cells.append(f"失败{flat['failed']}/{flat['cases']}；平移误差{number(metric.get('xy_velocity_error_mps'))}m/s；转速误差{number(metric.get('yaw_rate_error_rads'))}rad/s")
                flat_rows.append('<tr><td>'+title+'</td>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
            panels.append('<details><summary>平地能力逐项对照</summary><table><thead><tr><th>指令</th>'+''.join('<th>'+names[x]+'</th>' for x in labels)+'</tr></thead><tbody>'+''.join(flat_rows)+'</tbody></table><p class="note">每项4回合。新接口仅在前进且观察到间隙时启用，因此本轮不预期修复原模型已有的后退或转向不足。</p></details>')
            for item in replays['videos']:
                case=item['case'];label=f"{scope_title} / {names[reference]}对照 / "+(case['mode'] if case['kind']=='flat' else f"{'正向' if case['family']==0 else '斜向'} 路线{case['route']} L{case['level']} 变体{case['variant']}")
                video_choices.append(dict(label=label,src=f"{folder.name}/replays/{Path(item['video']).name}",poster=f"{folder.name}/replays/{Path(item['preview']).name}"))
    if not video_choices:raise ValueError('Completed replay evidence is required')
    initial=next((i for i,v in enumerate(video_choices) if '正向 路线0 L10 变体7' in v['label']),0)
    options=''.join(f'<option value="{i}"'+(' selected' if i==initial else '')+f'>{html.escape(v["label"])}</option>' for i,v in enumerate(video_choices))
    first=video_choices[initial]
    pending='<p class="note">阶段性结果：此页仅展示已完成的评估，其余对照仍在进行，暂不据此选定最终模型。</p>' if a.partial else ''
    if manifest.get('conclusion_zh'):pending+='<p><strong>本轮判断：</strong>'+html.escape(manifest['conclusion_zh'])+'</p>'
    page=('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>14700之后：支撑转换接口1000轮实验</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#232a30;background:#fafafa;max-width:1120px;margin:32px auto;padding:0 20px}h1{font-size:25px}h2{font-size:20px;margin-top:28px}.note{color:#58636b;font-size:14px}video{width:100%;display:block}select{font:inherit;padding:10px;max-width:100%;margin:12px 0}table{width:100%;border-collapse:collapse;background:white}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}th{background:#edf1f3}a{color:#245f82}section{margin-top:30px}
</style><h1>14700之后：支撑转换接口1000轮实验</h1>
<p>从同一14700轮出发。新接口冻结原策略，预测下一次落脚、步时和触地后120毫秒的身体及承重状态，驱动有幅度限制的小修正；普通续训保留原网络训练。两组各累计1000次更新，运动奖励保持一致。</p>
<p class="note">这是第一阶段接口原型，尚未实现多步候选评分或可行性规划。当前是理想局部高程图输入。两组冻结方式、网络和辅助监督均有差别，因此不能把结果归因于某个单一部件。普通组因主机内存不足在15100存档处恢复，补齐原1000轮预算。</p>
<h2>同场景动作回放</h2>'''+pending+f'<select id="scene">{options}</select><video id="video" controls muted playsinline loop preload="metadata" src="{first["src"]}" poster="{first["poster"]}"></video>'+
        '<p class="note">左为所选对照策略，右为新接口15700。真实Isaac Sim状态记录，MuJoCo仅渲染姿态；不是MuJoCo物理复验。包括固定样本及相对退步最明显的桩路样本。终止后冻结最后一帧并标注。</p>'+''.join(panels)+
        '<p class="note">评估设计：每组已有地图96回合、未见地图160回合，均含32个平地例。相同初始物理状态、共享质量与摩擦、无观测噪声或推力。未见地图沿用相同随机化类型，不代表任意地形或实机泛化；本轮仅一个训练种子。</p>'+\
        '<p><a href="experiment_manifest.json">实验定义</a></p><script>const items='+json.dumps(video_choices,ensure_ascii=False)+';const v=document.getElementById("video");document.getElementById("scene").onchange=e=>{const x=items[Number(e.target.value)];v.src=x.src;v.poster=x.poster;v.load();v.play().catch(()=>{});};</script></html>')
    if a.partial and not manifest.get('ordinary_completed'):
        page=page.replace('两组各累计1000次更新，运动奖励保持一致。','新接口已完成1000次更新，普通对照仍在补齐相同预算，运动奖励保持一致。')
    (root/'index.html').write_text(page)
    print(root/'index.html')


if __name__=='__main__':main()
