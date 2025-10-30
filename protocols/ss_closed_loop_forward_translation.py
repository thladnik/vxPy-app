import numpy as np

import vxpy.core.protocol as vxprotocol

from visuals.closed_loop_global_motion.gratings.translation_grating import ClosedLoopTranslationGrating

import vxpy.core.container as vxcontainer


class ForwardGainAdaptationGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)


        # vxcontainer.create_dataset('blaaa', (1,), np.int64)

        nTrials = 10
        for i in range(nTrials):


            # closed loop high or low gain
            external_speed_work = -1
            external_speed_relax = 0
            high_gain = 2.25
            low_gain = 0.5

            p = vxprotocol.Phase(60)
            p.set_visual(ClosedLoopTranslationGrating,
                         {ClosedLoopTranslationGrating.external_forward_velocity: external_speed_work,
                          ClosedLoopTranslationGrating.angular_period: 45.0,
                          ClosedLoopTranslationGrating.azimuth: 0.0,
                          ClosedLoopTranslationGrating.elevation: 0.0,
                          ClosedLoopTranslationGrating.fish_vel_gain: high_gain,
                          })
            self.add_phase(p)

            # relax
            p = vxprotocol.Phase(30)
            p.set_visual(ClosedLoopTranslationGrating,
                         {ClosedLoopTranslationGrating.external_forward_velocity: external_speed_relax,
                          ClosedLoopTranslationGrating.angular_period: 45.0,
                          ClosedLoopTranslationGrating.azimuth: 0.0,
                          ClosedLoopTranslationGrating.elevation: 0.0,
                          ClosedLoopTranslationGrating.fish_vel_gain: 1.0,
                          })
            self.add_phase(p)

            # low gain
            p = vxprotocol.Phase(60)
            p.set_visual(ClosedLoopTranslationGrating,
                         {ClosedLoopTranslationGrating.external_forward_velocity: external_speed_work,
                          ClosedLoopTranslationGrating.angular_period: 45.0,
                          ClosedLoopTranslationGrating.azimuth: 0.0,
                          ClosedLoopTranslationGrating.elevation: 0.0,
                          ClosedLoopTranslationGrating.fish_vel_gain: low_gain,
                          })
            self.add_phase(p)

            # relax
            p = vxprotocol.Phase(30)
            p.set_visual(ClosedLoopTranslationGrating,
                         {ClosedLoopTranslationGrating.external_forward_velocity: external_speed_relax,
                          ClosedLoopTranslationGrating.angular_period: 45.0,
                          ClosedLoopTranslationGrating.azimuth: 0.0,
                          ClosedLoopTranslationGrating.elevation: 0.0,
                          ClosedLoopTranslationGrating.fish_vel_gain: 1.0,
                          })
            self.add_phase(p)

