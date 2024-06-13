import vxpy.core.protocol as vxprotocol
# from visuals.yunyi_formal.tr_0430_moving import Triangle
from visuals.yunyi_pre.foreground_moving_forward import ForegroundMovingForward
# from visuals.yunyi_formal.foregournd_back import Triangle
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class CMN_binaryforeground(vxprotocol.StaticProtocol):

    def create(self):
        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [1.0, 1.0, 1.0]})
        self.add_phase(p)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 10)
        phase.set_visual(ForegroundMovingForward)
        self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)
