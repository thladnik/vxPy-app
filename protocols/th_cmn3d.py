import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.cmn_redesign import ContiguousMotionNoise3D, CMN3D20240410, CMN3D20240411
from visuals.spherical_global_motion import TranslationGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class CMN3DBaseProtocol(vxprotocol.StaticProtocol):
    cmn_version: ContiguousMotionNoise3D

    def create(self):

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)

        # Grey
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        # CMN
        phase = vxprotocol.Phase(duration=5 * 6)
        phase.set_visual(CMN3D20240411)
        self.add_phase(phase)

        # Grey
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        # Translation characterization
        for i in range(3):
            for azim in np.arange(-180, 180, 30):
                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 0})
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=5)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 30})
                self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


class CMN3DProtocol20240411(CMN3DBaseProtocol):
    cmn_version = CMN3D20240411
