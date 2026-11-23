"""Summarize completed rendered-frame measurements for the chapter table."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    all_results=[]
    for resolution in protocol['resolutions']:
        folder=out/f'resolution_{resolution}'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        trials=json.loads((folder/'trials.json').read_text(encoding='utf-8'))
        assert summary['completed'] and summary['source_unchanged'] and len(trials)==3
        for trial in trials:
            with (folder/f'frames_trial{trial["trial"]}.csv').open(encoding='utf-8') as file:
                frames=list(csv.DictReader(file))
            assert len(frames)==trial['frames']
            assert sum(int(r['physical_steps']) for r in frames)==trial['physical_steps']
            assert all(0<=int(row['physical_steps'])<=2 for row in frames)
            assert trial['wall_seconds']>=20 and trial['finite']
            np.testing.assert_allclose(trial['fps'],len(frames)/trial['wall_seconds'],atol=1e-12)
            np.testing.assert_allclose(trial['simulation_speed'],trial['physical_steps']*.002/trial['wall_seconds'],atol=1e-12)
            assert float(frames[-1]['wall_end_s'])<=trial['wall_seconds']
        row=dict(resolution=resolution,water_particles=summary['counts']['water'],
                 rigid_samples=summary['counts']['rigid_samples'],total_particles=summary['counts']['total'],
                 mean_fps=float(np.mean([t['fps'] for t in trials])),
                 min_fps=min(t['fps'] for t in trials),max_fps=max(t['fps'] for t in trials),
                 mean_speed=float(np.mean([t['simulation_speed'] for t in trials])),
                 min_speed=min(t['simulation_speed'] for t in trials),max_speed=max(t['simulation_speed'] for t in trials),
                 realtime=all(t['fps']>=30 for t in trials),trials=trials)
        all_results.append(row)
    source_match=all(digest(out/'source_snapshot/rigid_coupling'/key)==value for key,value in protocol['production_hashes'].items())
    assert source_match and digest(out/'source_snapshot/validation/ch04_realtime_gui.py')==protocol['runner_sha256']
    result=dict(passed=True,rows=all_results,completed_trials=12,
                classification='Every trial has displayed FPS >= 30; simulated/real time is reported separately.',
                source_snapshot_matches=True,human_scientific_review='pending')
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['| 水粒子数 | 平均帧率 / FPS | 实际播放速度 | 是否达到实时显示标准 |',
           '| ---: | ---: | ---: | :---: |']
    for row in all_results:
        lines.append(f"| {row['water_particles']:,} | {row['mean_fps']:.2f} | {row['mean_speed']:.3f} 倍 | {'是' if row['realtime'] else '否'} |")
    (out/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,out/'source_snapshot/validation'/Path(__file__).name)
    print('\n'.join(lines))
    print('All 12 measured trials and source hashes checked.')


if __name__=='__main__':main()
