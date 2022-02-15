import numpy as np

from vxpy.core.protocol import Phase, StaticPhasicProtocol
from vxpy.visuals import pause

from visuals.sphere_simu_saccade import GaussianConvNoiseSphereSimuSaccade as VisualClass
from visuals.sphere_visual_field_mapping import BinaryNoiseVisualFieldMapping16deg, BinaryNoiseVisualFieldMapping8deg
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.spherical_grating import SphericalBlackWhiteGrating


class Protocol01(StaticPhasicProtocol):
    saccade_params = [(1, -1, -150), (1, -1, 17), (1, -1, 50), (1, -1, 100), (1, -1, 250), (1, -1, 500), (1, -1, 2000),
                      (1, 1, -150), (1, 1, 17), (1, 1, 50), (1, 1, 100), (1, 1, 250), (1, 1, 500), (1, 1, 2000),
                      (-1, -1, -150), (-1, -1, 17), (-1, -1, 50), (-1, -1, 100), (-1, -1, 250), (-1, -1, 500),
                      (-1, -1, 2000), (-1, 1, -150), (-1, 1, 17), (-1, 1, 50), (-1, 1, 100), (-1, 1, 250), (-1, 1, 500),
                      (-1, 1, 2000), (0, -1, 2000), (0, 1, 2000)]

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        # Fix seed
        np.random.seed(1)

        # Blank at start of protocol for baseline
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)

        # Global ON/OFF response
        for i in range(4):
            p = Phase(duration=2)
            p.set_visual(SphereUniformBackground,
                         **{SphereUniformBackground.u_color: (0.5,) * 3})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(SphereUniformBackground,
                         **{SphereUniformBackground.u_color: (0.0,) * 3})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(SphereUniformBackground,
                         **{SphereUniformBackground.u_color: (0.5,) * 3})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(SphereUniformBackground,
                         **{SphereUniformBackground.u_color: (1.0,) * 3})
            self.add_phase(p)

        # Global rotations
        # In yaw
        for direction in [-1, 1]:
            p = Phase(duration=6)
            p.set_visual(SphericalBlackWhiteGrating,
                         **{SphericalBlackWhiteGrating.u_spat_period: 60,
                            SphericalBlackWhiteGrating.u_ang_velocity: direction * 30,
                            SphericalBlackWhiteGrating.p_type: 'rotation',
                            SphericalBlackWhiteGrating.p_shape: 'rectangular'})
            self.add_phase(p)

        # In roll
        # for direction in [-1, 1]:
        #     p = Phase(duration=6)
        #     p.set_visual(GratingRotAlongRoll,
        #                  **{GratingRotAlongRoll.u_spat_period: 60,
        #                     GratingRotAlongRoll.u_ang_velocity: direction * 30,
        #                     GratingRotAlongRoll.p_type: 'rotation',
        #                     GratingRotAlongRoll.p_shape: 'rectangular'})
        #     self.add_phase(p)

        # Spatial RF mapping for pos/neg polarities
        for inv in [True, False]:
            # 16 deg
            p = Phase(duration=120)
            p.set_visual(BinaryNoiseVisualFieldMapping16deg,
                         **{BinaryNoiseVisualFieldMapping16deg.p_interval: 1000,
                            BinaryNoiseVisualFieldMapping16deg.p_bias: .1,
                            BinaryNoiseVisualFieldMapping16deg.p_inverted: inv})
            self.add_phase(p)

            # 8 deg
            p = Phase(duration=180)
            p.set_visual(BinaryNoiseVisualFieldMapping8deg,
                         **{BinaryNoiseVisualFieldMapping8deg.p_interval: 1000,
                            BinaryNoiseVisualFieldMapping8deg.p_bias: .1,
                            BinaryNoiseVisualFieldMapping8deg.p_inverted: inv})
            self.add_phase(p)

        # Static texture
        p = Phase(duration=5)
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

        # Simulated saccades + flashes
        for _ in range(3):
            for sacc_direction, flash_polarity, flash_delay in np.random.permutation(self.saccade_params):
                p = Phase(duration=5)
                sacc_start = 1.5  # 1. + np.random.rand()
                flash_start = sacc_start + flash_delay / 1000

                p.set_visual(VisualClass,
                             **{VisualClass.p_sacc_duration: 100,
                                VisualClass.p_sacc_azim_target: 15,
                                VisualClass.p_sacc_direction: sacc_direction,
                                VisualClass.p_flash_polarity: flash_polarity,
                                VisualClass.p_sacc_start_time: sacc_start,
                                VisualClass.p_flash_start_time: flash_start,
                                VisualClass.p_flash_duration: 20.,
                                })
                self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)
