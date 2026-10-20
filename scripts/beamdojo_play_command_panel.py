"""One authoritative command state for the mixed-course playback GUI."""
from threading import RLock

MANUAL='手动速度'
AUTO='自动过桩停车'
PRESETS={'自定义':None,'站立':(0.,0.,0.),'前进':(.75,0.,0.),'后退':(-.5,0.,0.),
         '左横移':(0.,.2,0.),'右横移':(0.,-.2,0.),'左原地转向':(0.,0.,.2),'右原地转向':(0.,0.,-.2)}

class CommandPanel:
    def __init__(self,gui,interactive=True,flat=False,cruise=.75):
        if flat and not interactive:raise ValueError('Flat playback requires manual commands')
        if not .5<=cruise<=1.:raise ValueError('Cruise outside trained range')
        self._lock=RLock();self.interactive=interactive;self.flat=flat
        self._state={'manual':bool(interactive),'command':(.5,0.,0.),'cruise':float(cruise)}
        with gui.add_folder('运动指令',order=-95):
            self.mode=gui.add_dropdown('指令来源',options=[MANUAL] if flat and interactive else ([MANUAL,AUTO] if interactive else [AUTO]),initial_value=MANUAL if interactive else AUTO)
            with gui.add_folder('手动速度',visible=interactive) as self.manual_group:
                self.preset=gui.add_dropdown('快捷填写',options=list(PRESETS),initial_value='自定义')
                self.axes=[gui.add_slider(label,min=lo,max=hi,step=step,initial_value=value) for label,lo,hi,step,value in [
                    ('前后 vx (m/s)',-.5,1.,.05,.5),('左右 vy (m/s)',-.3,.3,.05,0.),('转向 wz (rad/s)',-.2,.2,.02,0.)]]
                gui.add_markdown('本体坐标；正值分别为前进、左移、左转。拖动后立即生效。站立是三轴速度均为零。')
            with gui.add_folder('自动过桩',visible=not interactive) as self.auto_group:
                self.cruise=gui.add_slider('巡航 vx (m/s)',min=.5,max=1.,step=.05,initial_value=cruise)
                gui.add_markdown('出生面朝桩路，横移和转速为零；到所选终点后自动停车。')
            self.status=gui.add_markdown('')
        self.mode.on_update(lambda _:self.set_mode(self.mode.value))
        self.preset.on_update(lambda _:self.set_preset(self.preset.value))
        for index,handle in enumerate(self.axes):
            handle.on_update(lambda _,i=index:self.set_axis(i,float(self.axes[i].value)))
        self.cruise.on_update(lambda _:self.set_cruise(float(self.cruise.value)))
        self._sync_visibility()

    def snapshot(self):
        with self._lock:return {**self._state,'command':list(self._state['command'])}

    def _sync_visibility(self):
        manual=self._state['manual'];self.manual_group.visible=manual;self.auto_group.visible=not manual

    def set_mode(self,mode):
        with self._lock:
            if mode==AUTO and self.flat:
                raise ValueError('Automatic course mode is unavailable on flat terrain')
            if mode not in (MANUAL,AUTO):raise ValueError('Unknown command mode')
            self._state['manual']=mode==MANUAL
            if self.mode.value!=mode:self.mode.value=mode
            self._sync_visibility()

    def set_axis(self,index,value):
        with self._lock:
            current=list(self._state['command'])
            if current[index]==value:return
            bounds=[(-.5,1.),(-.3,.3),(-.2,.2)][index]
            if not bounds[0]<=value<=bounds[1]:raise ValueError('Command outside trained range')
            current[index]=value;self._state['command']=tuple(current)
            if self.axes[index].value!=value:self.axes[index].value=value
            if self.preset.value!='自定义':self.preset.value='自定义'

    def set_preset(self,name):
        with self._lock:
            values=PRESETS[name]
            if values is None:return
            # Publish all axes first. Delayed slider callbacks then see the same
            # values and cannot transiently combine old and new command axes.
            self._state['command']=values
            if self.preset.value!=name:self.preset.value=name
            for handle,value in zip(self.axes,values):
                if handle.value!=value:handle.value=value

    def set_cruise(self,value):
        if not .5<=value<=1.:raise ValueError('Cruise outside trained range')
        with self._lock:
            self._state['cruise']=value
            if self.cruise.value!=value:self.cruise.value=value

    def set_terrain(self,flat):
        with self._lock:
            if flat and not self.interactive:raise ValueError('Flat playback requires manual commands')
            self.flat=bool(flat)
            if not self.interactive:return
            # Viser options setters can synchronously switch mode.value and run
            # callbacks. Commit the safe command before updating these options.
            if flat and not self._state['manual']:
                self._state['manual']=True;self._state['command']=(0.,0.,0.)
                self.set_preset('站立')
            self.mode.options=[MANUAL] if flat else [MANUAL,AUTO]
            self._sync_visibility()

    def report(self,effective,velocity,omega):
        state=self.snapshot();mode=MANUAL if state['manual'] else AUTO
        line=f'**当前生效：{mode}**\n\n目标 vx {effective[0]:.2f} · vy {effective[1]:.2f} m/s · wz {effective[2]:.2f} rad/s\n\n实际 vx {velocity[0]:.2f} · vy {velocity[1]:.2f} m/s · wz {omega[2]:.2f} rad/s'
        if state['manual'] and not self.flat:line+='\n\n手动桩路：自动终点停车关闭。'
        self.status.content=line
