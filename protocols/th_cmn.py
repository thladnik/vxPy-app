import vxpy.core.protocol as vxprotocol
from visuals.spherical_global_motion import CMN_100_000f_20fps_10tp_0p1sp
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class CMN20230804(vxprotocol.StaticProtocol):

    def create(self):

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        phase = vxprotocol.Phase(duration=60*60)
        phase.set_visual(CMN_100_000f_20fps_10tp_0p1sp)

        self.add_phase(phase)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)
