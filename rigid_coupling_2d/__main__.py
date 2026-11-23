"""Interactive planar impact or water entry: Space pause, R restart, Esc close."""
import argparse
from pathlib import Path
import numpy as np
import taichi as ti
from . import Config, Solver, build_scene
from .scene import CASES, RIGID, MATERIALS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['impact','drop','falling','rigid_drop'],default='falling')
    p.add_argument('--case',choices=list(CASES),default='offset_two_way')
    p.add_argument('--material',choices=list(MATERIALS),default='water')
    p.add_argument('--compliance',type=float,default=Config().compliance,help='elastic XPBMPM compliance; larger values are softer')
    p.add_argument('--arch',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--duration',type=float,default=None)
    p.add_argument('--steps-per-frame',type=int,default=4)
    p.add_argument('--view-width',type=float,default=None,help='display width centered at (0.5, 0.5)')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--screenshot',type=Path)
    a=p.parse_args()
    if a.scene=='drop' and a.material!='water':p.error('drop scene uses water')
    if a.scene in ('falling','rigid_drop') and a.case!='offset_two_way':p.error('--case applies to --scene impact')
    if a.duration is None:a.duration=.8 if a.scene=='drop' else (.9 if a.scene in ('falling','rigid_drop') else .5)
    if a.view_width is None:a.view_width=1.04 if a.scene in ('drop','falling','rigid_drop') else 1.65
    if a.duration<=0 or a.steps_per_frame<1 or not np.isfinite(a.view_width) or a.view_width<=0:
        p.error('duration, steps-per-frame and view-width must be positive')
    ti.init(arch=ti.cuda if a.arch=='cuda' else ti.cpu,offline_cache=False,enable_fallback=False)
    if a.scene=='drop':
        from .drop import DropSpec,drop_config,drop_scene,settled_pool
        c=drop_config();spec=DropSpec(duration=a.duration)
        print('Settling water before release...',flush=True)
        water=settled_pool(c,spec,lambda i,n:print(f'Settling {i}/{n}',flush=True))
        arrays,body=drop_scene(c,spec,water)
    elif a.scene=='falling':
        from .falling import falling_config,falling_scene
        c=falling_config(compliance=a.compliance)
        arrays,body=falling_scene(c,a.material)
    elif a.scene=='rigid_drop':
        from .rigid_drop import rigid_drop_config,rigid_drop_scene
        c=rigid_drop_config(compliance=a.compliance)
        arrays,body=rigid_drop_scene(c,a.material)
    else:
        c=Config(coupling=CASES[a.case][1],compliance=a.compliance)
        arrays,body=build_scene(c,a.case,a.material)
    s=Solver(c,arrays,body)
    gui=ti.GUI('2D material impact | Space: pause | R: restart | Esc: close',res=(900,900),background_color=0xF6F8FC,show_gui=not a.headless)
    continuum=arrays['material']!=RIGID
    color={'water':0x2185CE,'elastic':0x289776,'sand':0xA77829}[a.material]
    end=round(a.duration/c.dt)
    paused=False
    while gui.running:
        for event in gui.get_events(ti.GUI.PRESS):
            if event.key in (ti.GUI.ESCAPE,ti.GUI.EXIT): gui.running=False
            elif event.key==ti.GUI.SPACE: paused=not paused
            elif event.key in ('r','R'): s.reset();paused=False
        if not gui.running: break
        if not paused:
            for _ in range(min(a.steps_per_frame,end-s.step_index)): s.step()
            s.validate()
        points=(s.x.to_numpy()-.5)/a.view_width+.5
        if a.scene in ('drop','falling','rigid_drop'):
            lo=(c.origin+c.boundary*c.dx-.5)/a.view_width+.5
            hi=(c.origin+(c.grid-c.boundary)*c.dx-.5)/a.view_width+.5
            gui.lines(np.array([[lo,hi],[lo,lo],[hi,lo]]),np.array([[lo,lo],[hi,lo],[hi,hi]]),radius=2,color=0x536274)
            if a.scene in ('falling','rigid_drop'):gui.line((lo,hi),(hi,hi),radius=2,color=0x536274)
        gui.circles(points[continuum],radius=.95,color=color)
        gui.circles(points[~continuum],radius=.95,color=0xD77C31)
        label='Rigid water entry' if a.scene=='drop' else a.material+' | '+(a.case if a.scene=='impact' else a.scene)
        gui.text(f'{label}    t={s.step_index*c.dt:.3f} s',pos=(.04,.96),color=0x233347,font_size=24)
        gui.text('Space: pause   R: restart   Esc: close',pos=(.04,.92),color=0x536274,font_size=18)
        finished=s.step_index>=end
        if finished: paused=True
        if a.headless and finished:
            if a.screenshot:
                a.screenshot.parent.mkdir(parents=True,exist_ok=True)
                gui.show(str(a.screenshot))
            break
        gui.show()


if __name__=='__main__': main()
