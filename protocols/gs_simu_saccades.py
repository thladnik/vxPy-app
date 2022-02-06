import numpy as np

from vxpy.core.protocol import Phase, StaticPhasicProtocol
from vxpy.visuals import pause

from visuals.sphere_simu_saccade import IcoGaussianConvolvedNoiseSphereWithSimulatedHorizontalSaccade as VisualClass
from visuals.sphere_visual_field_mapping import BinaryNoiseVisualFieldMapping16deg
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class Protocol01(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        np.random.seed(1)

        # # Blank at start of protocol
        # p = Phase(duration=5)
        # p.set_visual(pause.ClearBlack)
        # self.add_phase(p)

        # for luminance_lvl in np.random.permutation(np.linspace(0., 1., 11)):
        #     p = Phase(duration=2)
        #     p.set_visual(SphereUniformBackground,
        #                  **{SphereUniformBackground.u_color: (luminance_lvl,) * 3})
        #     self.add_phase(p)

        p = Phase(duration=10)
        p.set_visual(BinaryNoiseVisualFieldMapping16deg,
                     **{BinaryNoiseVisualFieldMapping16deg.p_interval: 1000,
                        BinaryNoiseVisualFieldMapping16deg.p_bias: .5,
                        BinaryNoiseVisualFieldMapping16deg.p_inverted: False})
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
