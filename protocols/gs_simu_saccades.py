import numpy as np

from vxpy.core.protocol import Phase, StaticPhasicProtocol
from vxpy.visuals import pause

from visuals.sphere_simu_saccade import IcoGaussianConvolvedNoiseSphereWithSimulatedHorizontalSaccade as VisualClass


class Protocol01(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        # # Blank at start of protocol
        p = Phase(duration=5)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)

        # Static texture
        p = Phase(duration=4)
        p.set_visual(VisualClass,
                     **{VisualClass.p_sacc_duration: 100,
                        VisualClass.p_sacc_azim_target: 15,
                        VisualClass.p_sacc_direction: 1,
                        VisualClass.p_flash_polarity: 1,
                        VisualClass.p_sacc_start_time: -1.,
                        VisualClass.p_flash_start_time: -1,
                        VisualClass.p_flash_duration: 20.,
                        })
        self.add_phase(p)

        for i in range(10):
            p = Phase(duration=4)
            sacc_start = 1. + np.random.rand()
            flash_start = sacc_start + np.random.rand() / 2
            p.set_visual(VisualClass,
                         **{VisualClass.p_sacc_duration: 100,
                            VisualClass.p_sacc_azim_target: 15,
                            VisualClass.p_sacc_direction: [-1, 1][np.random.randint(2)],
                            VisualClass.p_flash_polarity: [-1, 1][np.random.randint(2)],
                            VisualClass.p_sacc_start_time: sacc_start,
                            VisualClass.p_flash_start_time: flash_start,
                            VisualClass.p_flash_duration: 20.,
                            })
            self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=30)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)
