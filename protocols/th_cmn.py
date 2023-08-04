
import vxpy.core.protocol as vxprotocol
from visuals.spherical_global_motion import CMN_100_000f_20fps_10tp_0p1sp


class CMN20230804(vxprotocol.StaticProtocol):

    def create(self):

        phase = vxprotocol.Phase(duration=60*80)
        phase.set_visual(CMN_100_000f_20fps_10tp_0p1sp)

        self.add_phase(phase)
