import vxpy.core.protocol as vxprotocol
from controls import fish_kebab_controls


class Protocol01Static(vxprotocol.StaticProtocol):

    def create(self):
        for i in range(10):

            phase = vxprotocol.Phase(duration=10)
            phase.set_control(fish_kebab_controls.Control01, {'velocity': i+1, 'direction': 1})
            self.add_phase(phase)


class Protocol01Triggered(vxprotocol.TriggeredProtocol):

    def create(self):
        for i in range(10):

            phase = vxprotocol.Phase(duration=10)
            phase.set_control(fish_kebab_controls.Control01, {'velocity': i+1})
            self.add_phase(phase)
