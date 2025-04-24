import numpy as np

import vxpy.core.protocol as vxprotocol

from controls.control_tests import TestControl01
from visuals.spherical_grating import SphericalBlackWhiteGrating
from visuals.closed_loop_global_motion.gratings.rotation_grating import ClosedLoopRotationGrating

import vxpy.core.container as vxcontainer


class RotationGainAdaptationGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)




        nTrials = 20
        for i in range(nTrials):


            # closed loop high or low gain
            external_speed = -0.5 if i % 2 == 0 else 0.5
            high_gain = 2
            low_gain = 0.5

            # high gain
            p = vxprotocol.Phase(30)
            p.set_visual(ClosedLoopRotationGrating,
                         {ClosedLoopRotationGrating.external_angular_velocity: external_speed,
                          ClosedLoopRotationGrating.angular_period: 45.0,
                          ClosedLoopRotationGrating.azimuth: 0.0,
                          ClosedLoopRotationGrating.elevation: 0.0,
                          ClosedLoopRotationGrating.fish_vel_gain: high_gain,
                          })
            self.add_phase(p)

            # relax
            p = vxprotocol.Phase(15)
            p.set_visual(ClosedLoopRotationGrating,
                         {ClosedLoopRotationGrating.external_angular_velocity: 0,
                          ClosedLoopRotationGrating.angular_period: 45.0,
                          ClosedLoopRotationGrating.azimuth: 0.0,
                          ClosedLoopRotationGrating.elevation: 0.0,
                          ClosedLoopRotationGrating.fish_vel_gain: 1,
                          })
            self.add_phase(p)


            # low gain
            p = vxprotocol.Phase(30)
            p.set_visual(ClosedLoopRotationGrating,
                         {ClosedLoopRotationGrating.external_angular_velocity: external_speed,
                          ClosedLoopRotationGrating.angular_period: 45.0,
                          ClosedLoopRotationGrating.azimuth: 0.0,
                          ClosedLoopRotationGrating.elevation: 0.0,
                          ClosedLoopRotationGrating.fish_vel_gain: low_gain,
                          })
            self.add_phase(p)

            # relax
            p = vxprotocol.Phase(15)
            p.set_visual(ClosedLoopRotationGrating,
                         {ClosedLoopRotationGrating.external_angular_velocity: 0,
                          ClosedLoopRotationGrating.angular_period: 45.0,
                          ClosedLoopRotationGrating.azimuth: 0.0,
                          ClosedLoopRotationGrating.elevation: 0.0,
                          ClosedLoopRotationGrating.fish_vel_gain: 0.0,
                          })
            self.add_phase(p)





