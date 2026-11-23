"""Publish experiment-one records locally, preserving raw data and provenance."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

from rigid_coupling.runner import write_json
from validation.ch04_impact import verify_sources


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(out,thesis,replace_report=False):
    protocol=verify_sources(out)
    asset_name='ch04-impact-'+protocol['date'].replace('-','')
    if protocol.get('variant'):asset_name+='-'+protocol['variant']
    destination=thesis/'assets'/asset_name
    public=thesis/'Ch04实验1_水流冲击刚体.md'
    if destination.exists():raise FileExistsError(destination)
    if public.exists() and not replace_report:raise FileExistsError(public)
    result=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    display=json.loads((out/'figures/particle_display.json').read_text(encoding='utf-8'))
    displayed=display['selected_counts']['center_two_way']
    audit=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    assert audit['passed']
    rows={r['case']:r for r in result['cases']}
    center,offset,mirror=[rows[n] for n in ['center_two_way','offset_two_way','mirror_two_way']]
    assert center['final_vx']>0 and offset['final_vx']>0
    assert abs(center['final_omega_z'])<1e-3
    assert offset['final_omega_z']<0<mirror['final_omega_z']
    assert all(rows[n]['displacement_x']==0 and rows[n]['peak_angular_speed']==0
               for n in ['center_one_way','offset_one_way'])
    max_penetration=max(r['max_penetration_h']/32 for r in result['cases'])
    assert max_penetration<1e-5
    copy_scripts=out/'source_snapshot'/'validation'
    for filename in ['ch04_impact_analyze.py','ch04_impact_report.py']:
        shutil.copy2(Path(__file__).with_name(filename),copy_scripts/filename)
    max_impulse_v=max(r['max_v_error'] for r in result['analytical_checks']['checks'])
    max_impulse_w=max(r['max_omega_error'] for r in result['analytical_checks']['checks'])
    max_drift=max(r['max_linear_momentum_drift_percent'] for r in result['cases']
                  if 'two_way' in r['case'])
    claims={
        'C4101':('采用当前 rigid_coupling.Solver，水/刚体分开传递、完整 P2G/G2P、刚体体积占据及几何接触开启。重力、黏性和浮力辅助关闭；无容器移动或刚体外加运动。此次采用 raw/clamp 的局部界面配置，没有调用 reflected 容器墙面路径。', ['E4101','E4102']),
        'C4102':(f'4 项解析冲量检查全部通过，包括居中分布冲量、偏心单采样冲量，以及刚体旋转 0.63 rad 后的对应检查。线速度最大绝对误差为 {max_impulse_v:.3g}，角速度最大绝对误差为 {max_impulse_w:.3g}。这验证已知粒子冲量到刚体速度的反馈计算，不代表整场景的流体压力具有同等精度。', ['E4103']),
        'C4103':(f'在 0.5 s 时，对心双向组的刚体沿 x 方向移动 {center["displacement_x"]:.6f}，水平速度为 {center["final_vx"]:.6f}，绕 z 轴角速度为 {center["final_omega_z"]:.3g} rad/s；偏心双向组的水平速度为 {offset["final_vx"]:.6f}，绕 z 轴角速度为 {offset["final_omega_z"]:.6f} rad/s，绕 z 轴的有符号姿态角为 {offset["signed_yaw_deg"]:.3f}°。两个单向组的刚体均保持初始位姿和零速度。', ['E4104','E4105']),
        'C4104':(f'对心与偏心双向组的刚体分别获得初始水流 x 向动量的 {center["body_px_received_percent"]:.3f}% 和 {offset["body_px_received_percent"]:.3f}%。包括两项数值抽查在内，所有双向组的最大总线动量漂移不超过初始水流动量的 {max_drift:.5f}%。这里的动量漂移只反映该隔离场景的离散收支，不能作为轨迹物理精度的替代指标。', ['E4104','E4105']),
        'C4105':(f'将水团从刚体中心上方移至等距离下方后，末时刻角速度由 {offset["final_omega_z"]:.6f} 变为 {mirror["final_omega_z"]:.6f} rad/s。对反向偏心轨迹作 y 镜像后，完整时间窗内刚体质心的最大差为 {result["symmetry"]["max_center_difference"]:.3g}，角速度最大差为 {result["symmetry"]["max_omega_difference"]:.3g} rad/s。', ['E4104','E4105']),
        'C4106':(f'8 条轨迹均完成预定 0.5 s，状态有限，刚体无世界边界接触。全部物理步末的最大水粒子采样穿透为 {max_penetration:.3g}，未超过代码中统计内部粒子所用的 1e-5 容差；粒子传递支撑未触及执行截断的外边界节点。该记录不是连续时间、子迭代内部或连续水体几何的无穿透保证。', ['E4104','E4105']),
    }
    def claim(cid):
        text,ids=claims[cid]
        return text+' <!-- [claim:'+cid+'] [evidence:'+','.join(ids)+'] -->'
    main_table='\n'.join(f'| {rows[n]["label"]} | {rows[n]["displacement_x"]:.6f} | {rows[n]["final_vx"]:.6f} | {rows[n]["final_omega_z"]:.6f} | {rows[n]["signed_yaw_deg"]:.3f} |' for n in ['center_one_way','center_two_way','offset_one_way','offset_two_way','mirror_two_way'])
    quality_table='\n'.join(f'| {r["label"]} | {r["max_linear_momentum_drift_percent"]:.5f} | {r["max_angular_momentum_drift_percent"]:.5f} | {r["max_volume_change_percent"]:.4f} | {r["max_penetration_h"]:.3g} |' for r in result['cases'])
    sensitive='\n'.join(f'| {rows[name]["label"]} | {v["position_rms_body_length"]*100:.4f} | {v["velocity_rms_incoming_speed"]*100:.4f} | {v["orientation_rms_deg"]:.4f} |' for name,v in result['sensitivity'].items())
    preliminary_note = ''
    if (out/'preliminary/README.md').exists():
        preliminary_note = ('初次加密运行中，单向偏心组在末段触及原外边界截断层，因此将截断层宽从 3 格改为 2 格，再运行全部对照。'
                            '初次还将解析检查的粒子数提高至 12288，两个居中冲量检查的线速度最大绝对差为 4.74×10⁻⁷，超过原 192 粒子检查的容差。'
                            '正式解析检查沿用原 192 粒子设置；高粒子数检查的结果单独保留。过程及原始结果见 [初次运行记录](preliminary/README.md)。')
    report=f'''# Ch04 实验 1：水流冲击刚体的双向反馈验证

日期：{protocol['date']}。水体采用球形水团，侧视投影呈圆形；数据及图表对应本次重新运行的实验。

## 1 实验配置

本实验检查材料能否推动刚体，以及偏心作用能否产生相应转动。比较相同初态下的单向与双向耦合，再以反向偏心和无刚体水流作为补充检查。协议在首次数值运行前保存；实验过程中未改变生产求解器源码。[E4101,E4102,E4105]

{claim('C4101')}

| 项目 | 设置 |
|---|---|
| 计算域 / 网格 | [0,1]³，32³ |
| 时间步 / 每步迭代 | 0.002 s / 5 |
| 外边界截断层宽 | {protocol['baseline']['boundary']} 个网格单元 |
| 正式观察窗 | 0–0.5 s，250 步；含初态共 251 个记录 |
| 水团中心 / 形状 | (0.328125,0.5,0.5) / 球形 |
| 水团半径 / 体积 | {protocol['water_radius']:.6f} / {protocol['water_volume']:.11f} |
| 水流初速度 | (0.75,0,0) |
| 刚体中心 / 尺寸 | (0.625,0.5,0.5) / (0.125,0.25,0.1875) |
| 水 / 刚体密度 | 均为 1 |
| 水 / 刚体采样数 | {protocol['water_particle_count']} / {protocol['body_particle_count']} |
| 采样方式 | 边长 1/128 的体积单元内随机取点，按中心平面对称生成 |
| 固定随机种子 | 水体 {protocol['sampling']['water_seed']}；刚体 {protocol['sampling']['body_seed']} |
| 水质量 / 刚体质量 | {protocol['water_mass']:.11f} / {protocol['body_mass']:.9f} |
| 偏心量 | 水团中心沿 y 移动 +0.0625；反向组为 −0.0625 |
| 初始几何间隙 | 水团前表面距刚体左表面 {protocol['initial_surface_gap']:.6f} |
| 后端 / 精度 | Taichi 1.7.4，CUDA；求解为 float32，统计为 NumPy float64 |

以上长度和质量为一致的计算单位，不将场景尺寸解释为经过物理标定的米制实验。水体材料仍执行体积修正，未为了得到平移或转动而关闭材料求解。[E4101,E4102]

单向组调用已有 `coupling="one_way"`：刚体不接收反馈，接触有效质量不含刚体逆质量与逆惯量。在本实验零初速度、零重力条件下，刚体保持静止。这是规定运动边界对照，不是准确解；单向与双向组的差异包含这一完整开关的行为。两对主实验的初始存档逐字段一致，配置仅 coupling 不同。[E4101,E4102,E4105]

共运行 4 条主要轨迹、反向偏心与无刚体控制各 1 条，另有两条偏心双向数值抽查。每种配置记录一次完整轨迹；预热后恢复初态，才开始记录。此实验不作性能基准，日志耗时包含采样与写盘。[E4101,E4104]

## 2 解析冲量检查

解析检查沿用原来的 192 粒子简化刚体。将已知冲量写入采样粒子的位移增量，调用生产代码的反馈收集与投影，并与独立 NumPy 计算比较：

\\[
\\Delta\\mathbf v=\\mathbf J/M,\\qquad
\\Delta\\boldsymbol\\omega=\\mathbf I_{{world}}^{{-1}}(\\mathbf r\\times\\mathbf J).
\\]

居中冲量按质量比例分配到全部采样，理论合力矩为零；偏心冲量施加于一个已知位置的采样粒子。惯量由同一采样质量分布计算，角速度采用世界坐标。[E4102,E4103]

{claim('C4102')}

## 3 冲击响应

{claim('C4103')}

| 配置 | 末时刻水平位移 | 末时刻水平速度 | 末时刻 ωz / rad·s⁻¹ | 末时刻有符号姿态角 / ° |
|---|---:|---:|---:|---:|
{main_table}

表中末时刻统一为 0.5 s，位移从初始质心计算；姿态角为世界 XY 投影中的绕 z 轴角。数据来自各组完整轨迹。[E4104,E4105]

![刚体运动曲线](figures/response.png)

图 1：实线为双向耦合，虚线为单向耦合，使用相同物理时间。对心组主要产生平移，偏心组同时产生转动。[E4104,E4106]

![四个主对照的同时间水流冲击画面](figures/impact_frames.png)

图 2：蓝色为水粒子，橙色为刚体采样粒子。两种材料均跟踪初始满足 |z−0.5|＜1/64 的中央薄层内的同一批粒子，分别显示 {displayed['0']}、{displayed['1']} 个粒子的 XY 投影。各面板采用相同的视角、坐标范围和物理时刻，统计使用全部三维粒子。[E4104,E4106]

[查看四组对照动画](figures/impact_comparison.gif)。动画每隔 0.01 s 保存一帧，按 50 帧/s 播放，即约 0.5 倍物理速度。[E4101,E4106]

## 4 反馈检查

{claim('C4104')}

{claim('C4105')}

![动量收支及反向偏心响应](figures/momentum_and_mirror.png)

图 3：左图比较水流损失与刚体获得的 x 向动量；中图为总线动量漂移；右图为上、下偏心冲击的转动方向。左图两条近乎重合的曲线表示相应动量收支接近。[E4104,E4105,E4106]

总线动量由水粒子质量乘速度与刚体质量乘质心速度相加。角动量以初始刚体质心为固定参考点，水体包括轨道项与 APIC 仿射项，刚体包括轨道项与世界惯量乘角速度。线动量漂移除以初始水流动量，角动量漂移除以该动量乘刚体 y 向尺寸 0.25。[E4101,E4102,E4105]

| 配置 | 最大线动量漂移 / % | 最大角动量漂移 / % | 最大水体体积变化 / % | 最大步末采样穿透 / h |
|---|---:|---:|---:|---:|
{quality_table}

单向组丢弃刚体反作用，其水—刚体状态不构成封闭动量系统，表中较大的漂移反映未计入的外部反力，不能用来宣称双向算法比单向算法具有相应百分比的物理精度提升。水体体积由各粒子的 V0×J 求和。[E4101,E4102,E4104,E4105]

{claim('C4106')}

## 5 参数敏感性

保持几何、质量、采样和初态不变，分别把时间步改为 0.001 s（仍 5 轮），或将迭代改为 10 轮（仍 dt=0.002 s）。两项对照分别改变一个参数。下表对齐共同的 250 个非初始物理时刻，不平移时间轴。[E4101,E4104,E4105]

| 相对基准配置 | 质心位置 RMS 差 / 体长 % | 速度 RMS 差 / 来流速度 % | 姿态 RMS 差 / ° |
|---|---:|---:|---:|
{sensitive}

体长为 0.25、来流速度为 0.75；姿态差采用归一化四元数之间的最小旋转角。这些是配置差异，不是相对物理真值的误差，也不足以估计收敛阶。[E4101,E4105]

![偏心双向冲击的参数敏感性](figures/sensitivity.png)

## 6 本实验支持的结论

在本次无重力、无浮力辅助的水流冲击场景中，当前分开传递与几何接触的组合能将材料作用反馈为刚体平移和转动；偏心方向变化对应转动方向变化。单向开关对照、解析冲量检查和动量记录共同支持这一有限范围的结论。[E4101–E4105]

该实验没有独立的整场景物理参考，未检查自由漂浮、墙面镜像、多刚体接触或长时间稳定性。刚体形状由共同位姿重建，形状误差只作为实现一致性检查。此处结果属于当前参数下的机制验证，不推广为任意场景的精确流固耦合。[E4101,E4102,E4105]

## 7 复现命令

实验入口调用当前主求解器。输出目录必须尚不存在；以下命令运行本实验全部 8 条轨迹并生成图表。随机种子写在实验脚本中，重复运行使用相同的初始采样。

```powershell
Set-Location -LiteralPath 'D:\\mine\\learningMarkdown\\科研知识库\\code\\PBMPM_taichi'
$impactOut = 'results/ch04_impact_sphere_repeat'
.\\.venv\\Scripts\\python.exe -m validation.ch04_impact --stage prepare --output $impactOut
.\\.venv\\Scripts\\python.exe -m validation.ch04_impact --stage impulse --output $impactOut
if ($LASTEXITCODE -ne 0) {{ throw '解析检查失败' }}

$impactCases = @('center_one_way', 'center_two_way', 'offset_one_way', 'offset_two_way', 'mirror_two_way', 'fluid_control')
foreach ($impactCase in $impactCases) {{
    .\\.venv\\Scripts\\python.exe -m validation.ch04_impact --stage run --case $impactCase --output $impactOut
    if ($LASTEXITCODE -ne 0) {{ throw "实验失败：$impactCase" }}
}}
.\\.venv\\Scripts\\python.exe -m validation.ch04_impact --stage run --case offset_two_way --dt 0.001 --output $impactOut
.\\.venv\\Scripts\\python.exe -m validation.ch04_impact --stage run --case offset_two_way --iterations 10 --output $impactOut
.\\.venv\\Scripts\\python.exe -m validation.ch04_impact_analyze --output $impactOut
```

默认后端为 CUDA。已生成结果可直接查看 `figures/impact_comparison.gif`；无需重新运行。若未来主代码变化，新的 prepare 会记录新的源码哈希；旧输出不能混入新版本结果。[E4101,E4102]

## 8 实验记录

- [执行前协议](protocol.json)：配置、观察窗、指标和触发项。
- [结果汇总](analysis.json)、[对照表 CSV](comparison.csv)、[一致性核对](audit.json)。
- 各配置文件夹：每步 `metrics.csv`，5 个完整状态存档，51 帧动画状态及 `summary.json`。
- [解析冲量检查](impulse_checks/results.json)。
- [证据索引](source_manifest.json)、[论断登记](claims.csv)、`source_snapshot/`：生产源码及实验/分析脚本快照。

机器核对包括主对照初态逐字段一致、配置差异、全部输出有限性，以及从 40 个完整状态存档独立重算总线动量。人工科学核验记录为 pending。[E4105]

## 9 采样调整记录

水体从长方体改为等体积球形，总质量及来流速度保持不变。先在小体积单元内随机取点，再保留球内粒子，并按球体体积除以粒子数分配单粒子体积。采样关于三个中心平面对称，随机种子固定。刚体沿用 12288 个随机采样粒子。图中使用保存的粒子坐标，蓝色表示水体，橙色表示刚体。

{preliminary_note}
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    with (out/'claims.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['claim_id','text_sha256','evidence_ids','human_verification'])
        writer.writeheader()
        for cid,(text,ids) in claims.items():
            writer.writerow(dict(claim_id=cid,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                 evidence_ids=','.join(ids),human_verification='pending'))
    groups={
        'E4101':['protocol.json','environment_cuda.json'] + [str(p.relative_to(out)).replace('\\','/') for p in sorted((out/'preliminary').glob('*')) if p.is_file()],
        'E4102':[str(p.relative_to(out)).replace('\\','/') for p in sorted((out/'source_snapshot').rglob('*.py'))],
        'E4103':['impulse_checks/results.json'],
        'E4104':[str(p.relative_to(out)).replace('\\','/') for p in sorted(out.glob('*/*')) if p.suffix in ['.csv','.npz','.json'] and p.parent.name in rows],
        'E4105':['analysis.json','audit.json','comparison.csv','sampling_audit.json'],
        'E4106':[str(p.relative_to(out)).replace('\\','/') for p in sorted((out/'figures').iterdir())]}
    manifest=dict(human_verification='pending',generated_from_local_records=True,
                  sources=[dict(id=eid,verification='unverified_pending_human',
                                files=[dict(path=p,sha256=sha(out/p)) for p in paths])
                           for eid,paths in groups.items()])
    write_json(out/'source_manifest.json',manifest)
    shutil.copytree(out,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    if public.exists():shutil.copy2(public,destination/'previous_public_report.md')
    # Rewrite local figure/data links to the self-contained chapter asset folder.
    public_text=re.sub(r'(!?\[[^\]]*\])\(([^)]+)\)',lambda m:f'{m[1]}(assets/{asset_name}/{m[2]})',report)
    public.write_text(public_text,encoding='utf-8')
    check=[]
    for source in manifest['sources']:
        for file in source['files']:
            assert sha(destination/file['path'])==file['sha256']
            check.append(file['path'])
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',public_text):
        assert (thesis/link).exists(),link
    write_json(destination/'publication_audit.json',dict(passed=True,verified_copied_files=len(check),
                                                      document_sha256=sha(public),human_review='pending'))
    print(public)
    print(f'Copied and verified {len(check)} evidence files')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--thesis',type=Path,required=True)
    p.add_argument('--replace-report',action='store_true',help='Archive and replace the existing companion report')
    a=p.parse_args();publish(a.output,a.thesis,a.replace_report)


if __name__=='__main__':main()
