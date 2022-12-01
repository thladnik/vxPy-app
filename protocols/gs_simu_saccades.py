import numpy as np

from vxpy.core.protocol import Phase, StaticProtocol
from vxpy.visuals import pause

from visuals.sphere_simu_saccade import GaussianConvNoiseSphereSimuSaccade
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.spherical_grating import SphericalBlackWhiteGrating
from visuals.gs_saccadic_suppression import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression import SimuSaccadeWithSineFlash4000


'''class SimuSaccadeWithSinesZeroFlicker(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(5):

            p = Phase(duration=5)
            p.set_visual(SimuSaccadeWithSineFlash2000,
                         {SimuSaccadeWithSineFlash2000.saccade_duration: 180,
                          SimuSaccadeWithSineFlash2000.saccade_start_time: 1000,
                          SimuSaccadeWithSineFlash2000.saccade_target_angle: 20.,
                          SimuSaccadeWithSineFlash2000.sine_start_time: 2000,
                          SimuSaccadeWithSineFlash2000.sine_duration: 1000,
                          SimuSaccadeWithSineFlash2000.sine_amp: 0.0,
                          SimuSaccadeWithSineFlash2000.sine_freq: 2.0,
                          SimuSaccadeWithSineFlash2000.baseline_lum: 0.5,
                          SimuSaccadeWithSineFlash2000.contrast: 0.5})
            self.add_phase(p)'''


'''class SimuSaccadeWithSinesZeroSaccade(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(5):

            p = Phase(duration=5)
            p.set_visual(SimuSaccadeWithSineFlash,
                         {SimuSaccadeWithSineFlash.saccade_duration: 100,
                          SimuSaccadeWithSineFlash.saccade_start_time: 1000,
                          SimuSaccadeWithSineFlash.saccade_target_angle: 0.0,
                          SimuSaccadeWithSineFlash.sine_start_time: 2000,
                          SimuSaccadeWithSineFlash.sine_duration: 1000,
                          SimuSaccadeWithSineFlash.sine_amp: 0.25,
                          SimuSaccadeWithSineFlash.sine_freq: 2.0,
                          SimuSaccadeWithSineFlash.baseline_lum: 0.5,
                          SimuSaccadeWithSineFlash.contrast: 0.5})
            self.add_phase(p)'''


class SimuSaccadeWithSines(StaticProtocol):

    # Define parameters saccade size in degrees (0, -20, 20), sine flicker delay in ms (-500, 20, 100, 250, 500, 1000,
    # 2000, no flicker), flicker amplitude (35% oder 0%)
    saccade_params = [(-20, -500, 0.35), (-20, 20, 0.35),(-20, 100, 0.35), (-20, 250, 0.35), (-20, 500, 0.35),
                      (-20, 1000, 0.35), (-20, 2000, 0.35), (20, -500, 0.35), (20, 20, 0.35),(20, 100, 0.35),
                      (20, 250, 0.35), (20, 500, 0.35), (20, 1000, 0.35), (20, 2000, 0.35), (0, 0, 0.35), (0, 0, 0.35),
                      (-20, 500, 0), (-20, 500, 0), (20, 500, 0), (20, 500, 0)]


    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        # Fix seed
        np.random.seed(1)

        # # Blank at start of protocol for baseline
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)

        # Repeat paradigm twice for coarse and fine texture
        for i in range(2):

            # Texture displacement saccade protocol for fine texture (4000 blobs)
            for sacc_target, sine_delay, sine_amp in np.random.permutation(self.saccade_params):
                p = Phase(duration=8)
                sine_start = sine_delay + 1500
                p.set_visual(SimuSaccadeWithSineFlash4000,
                         {SimuSaccadeWithSineFlash4000.saccade_duration: 100,
                          SimuSaccadeWithSineFlash4000.saccade_start_time: 1500,
                          SimuSaccadeWithSineFlash4000.saccade_target_angle: sacc_target,
                          SimuSaccadeWithSineFlash4000.sine_start_time: sine_start,
                          SimuSaccadeWithSineFlash4000.sine_duration: 500,
                          SimuSaccadeWithSineFlash4000.sine_amp: sine_amp,
                          SimuSaccadeWithSineFlash4000.sine_freq: 15.0,
                          SimuSaccadeWithSineFlash4000.baseline_lum: 0.5,
                          SimuSaccadeWithSineFlash4000.contrast: 0.3})
                self.add_phase(p)

            # Texture displacement saccade protocol for coarse texture (2000 blobs)
            for sacc_target, sine_delay, sine_amp in np.random.permutation(self.saccade_params):

                p = Phase(duration=8)
                sine_start = sine_delay + 1500
                p.set_visual(SimuSaccadeWithSineFlash2000,
                         {SimuSaccadeWithSineFlash2000.saccade_duration: 100,
                          SimuSaccadeWithSineFlash2000.saccade_start_time: 1500,
                          SimuSaccadeWithSineFlash2000.saccade_target_angle: sacc_target,
                          SimuSaccadeWithSineFlash2000.sine_start_time: sine_start,
                          SimuSaccadeWithSineFlash2000.sine_duration: 500,
                          SimuSaccadeWithSineFlash2000.sine_amp: sine_amp,
                          SimuSaccadeWithSineFlash2000.sine_freq: 15.0,
                          SimuSaccadeWithSineFlash2000.baseline_lum: 0.5,
                          SimuSaccadeWithSineFlash2000.contrast: 0.3})
                self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)


'''class Protocol01(StaticPhasicProtocol):
    # Pre-2022-03-15 parameters:
    # saccade_params = [(1, -1, -150), (1, -1, 17), (1, -1, 50), (1, -1, 100), (1, -1, 250), (1, -1, 500), (1, -1, 2000),
    #                   (1, 1, -150), (1, 1, 17), (1, 1, 50), (1, 1, 100), (1, 1, 250), (1, 1, 500), (1, 1, 2000),
    #                   (-1, -1, -150), (-1, -1, 17), (-1, -1, 50), (-1, -1, 100), (-1, -1, 250), (-1, -1, 500),
    #                   (-1, -1, 2000), (-1, 1, -150), (-1, 1, 17), (-1, 1, 50), (-1, 1, 100), (-1, 1, 250), (-1, 1, 500),
    #                   (-1, 1, 2000), (0, -1, 2000), (0, 1, 2000)]

    saccade_params = [(1, -1, -150), (1, -1, 17), (1, -1, 50), (1, -1, 100), (1, -1, 250), (1, -1, 500), (1, -1, 2000),
                      (1, 1, -150), (1, 1, 17), (1, 1, 50), (1, 1, 100), (1, 1, 250), (1, 1, 500), (1, 1, 2000),
                      (-1, -1, -150), (-1, -1, 17), (-1, -1, 50), (-1, -1, 100), (-1, -1, 250), (-1, -1, 500),
                      (-1, -1, 2000), (-1, 1, -150), (-1, 1, 17), (-1, 1, 50), (-1, 1, 100), (-1, 1, 250), (-1, 1, 500),
                      (-1, 1, 2000), (0, -1, 2000), (0, 1, 2000), (1, 0, 0), (-1, 0, 0)]

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        # Fix seed
        np.random.seed(1)

        # # Blank at start of protocol for baseline
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)

        # Global ON/OFF response
        for i in range(3):
            for c in [0.25, 0.50, 0.75, 1.0, 0.75, 0.50, 0.25, 0.0]:
                p = Phase(duration=4)
                p.set_visual(SphereUniformBackground,
                             {SphereUniformBackground.u_color: (c,) * 3})
                self.add_phase(p)

        # Global rotations
        for i in range(3):
            # In yaw
            for direction in [-1, 1]:
                p = Phase(duration=3)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: 60,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

                p = Phase(duration=6)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: 60,
                              SphericalBlackWhiteGrating.angular_velocity: direction * 30})
                self.add_phase(p)

        for i in range(3):
            # In roll
            for direction in [-1, 1]:
                p = Phase(duration=3)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'forward',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: 60,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

                p = Phase(duration=6)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'forward',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: 60,
                              SphericalBlackWhiteGrating.angular_velocity: direction * 30})
                self.add_phase(p)

        # Static texture
        p = Phase(duration=5)
        p.set_visual(GaussianConvNoiseSphereSimuSaccade,
                     {GaussianConvNoiseSphereSimuSaccade.saccade_start_time: 10,
                      GaussianConvNoiseSphereSimuSaccade.saccade_duration: 100,
                      GaussianConvNoiseSphereSimuSaccade.saccade_azim_target: 15,
                      GaussianConvNoiseSphereSimuSaccade.saccade_direction: 1,
                      GaussianConvNoiseSphereSimuSaccade.flash_start_time: 10,
                      GaussianConvNoiseSphereSimuSaccade.flash_duration: 20,
                      GaussianConvNoiseSphereSimuSaccade.flash_polarity: 1})
        self.add_phase(p)

        # Simulated saccades + flashes
        for _ in range(3):
            for sacc_direction, flash_polarity, flash_delay in np.random.permutation(self.saccade_params):
                p = Phase(duration=5)
                sacc_start = 1.5  # 1. + np.random.rand()
                flash_start = sacc_start + flash_delay / 1000
                p.set_visual(GaussianConvNoiseSphereSimuSaccade,
                             {GaussianConvNoiseSphereSimuSaccade.saccade_start_time: sacc_start,
                              GaussianConvNoiseSphereSimuSaccade.saccade_duration: 100,
                              GaussianConvNoiseSphereSimuSaccade.saccade_azim_target: 15,
                              GaussianConvNoiseSphereSimuSaccade.saccade_direction: sacc_direction,
                              GaussianConvNoiseSphereSimuSaccade.flash_start_time: flash_start,
                              GaussianConvNoiseSphereSimuSaccade.flash_duration: 20,
                              GaussianConvNoiseSphereSimuSaccade.flash_polarity: flash_polarity})
                self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=15)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)'''
