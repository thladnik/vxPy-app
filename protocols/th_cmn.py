import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.spherical_global_motion import (CMN_100_000f_20fps_10tp_0p1sp,
                                             CMN_15_000f_15fps_10tp_0p1sp_0p03ns,
                                             CMN_15_000f_15fps_10tp_0p1sp_0p03ns_inv)
from visuals.spherical_global_motion import TranslationGrating
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


class CMN20230808(vxprotocol.StaticProtocol):

    def create(self):

        # p = vxprotocol.Phase(15)
        # p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        # self.add_phase(p)

        for i in range(3):
            for azim in np.arange(-180, 180, 45):

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 0})
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 30})
                self.add_phase(phase)


        phase = vxprotocol.Phase(duration=15*60)
        phase.set_visual(CMN_15_000f_15fps_10tp_0p1sp_0p03ns)
        self.add_phase(phase)

        phase = vxprotocol.Phase(duration=15*60)
        phase.set_visual(CMN_15_000f_15fps_10tp_0p1sp_0p03ns_inv)
        self.add_phase(phase)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)
