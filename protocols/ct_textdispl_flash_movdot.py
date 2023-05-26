from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_movingDotOnTexture.gs_dot_on_texture import MovingDotOnTexture2000
from visuals.gs_movingDotOnTexture.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
import numpy as np


def saccade2000(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
    return {
        SimuSaccadeWithSineFlash2000.saccade_duration: sacc_duration,
        SimuSaccadeWithSineFlash2000.saccade_start_time: sacc_start,
        SimuSaccadeWithSineFlash2000.saccade_target_angle: sacc_ang,
        SimuSaccadeWithSineFlash2000.sine_start_time: sine_start,
        SimuSaccadeWithSineFlash2000.sine_duration: sine_dur,
        SimuSaccadeWithSineFlash2000.sine_amp: sine_amp,
        SimuSaccadeWithSineFlash2000.sine_freq: sine_freq,
        SimuSaccadeWithSineFlash2000.baseline_lum: baseline_lum,
        SimuSaccadeWithSineFlash2000.contrast: contrast
    }


def dot2000(luminance, contrast, motion_axis, dot_polarity, dot_start_ang, dot_ang_vel, dot_ang_diameter, dot_offset):
    return {
        MovingDotOnTexture2000.luminance: luminance,
        MovingDotOnTexture2000.contrast: contrast,
        MovingDotOnTexture2000.motion_axis: motion_axis,
        MovingDotOnTexture2000.dot_polarity: dot_polarity,
        MovingDotOnTexture2000.dot_start_angle: dot_start_ang,
        MovingDotOnTexture2000.dot_angular_velocity: dot_ang_vel,
        MovingDotOnTexture2000.dot_angular_diameter: dot_ang_diameter,
        MovingDotOnTexture2000.dot_offset_angle: dot_offset
    }


class TextureDisplacementFlashMovingDot(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed saccade parameters
        sacc_duration = 100
        sacc_start = 8000
        sine_dur = 0
        sine_freq = 0
        sine_amp = 0
        baseline_lum = 0.75
        contrast = 0.5

        # set fixed dot parameters
        luminance = 0.75
        motion_axis = 'vertical'
        dot_polarity = 'dark-on-light'
        dot_start_ang = 30
        dot_ang_vel = -60
        dot_ang_diameter = 20
        dot_offset = 10

        # experimental conditions, (sacc_target, delay, stim_type (-1 = flash, 1 = mov dot))
        conditions = [(-30, 100, -1), (-30, 250, -1), (-30, 500, -1), (-30, 1000, -1), (-30, 2000, -1), (-30, 4000, -1),
                      (30, 100, -1), (30, 250, -1), (30, 500, -1), (30, 1000, -1), (30, 2000, -1), (30, 4000, -1),
                      (0, 750, -1), (0, 750, -1), (-30, 100, 1), (-30, 250, 1), (-30, 500, 1), (-30, 1000, 1),
                      (-30, 2000, 1), (-30, 4000, -1), (30, 100, 1), (30, 250, 1), (30, 500, 1), (30, 1000, 1),
                      (30, 2000, 1), (30, 4000, 1), (0, 750, 1), (0, 750, 1)]

        # 4 repeats af all saccade, delay, and stim_type conditions, coarse texture only
        for i in range(4):
            # 5 seconds just texture (coarse)
            p = Phase(duration=5)
            p.set_visual(SimuSaccadeWithSineFlash2000, saccade2000(sacc_duration, sacc_start, 0, 111, sine_dur, sine_amp,
                                                                   sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # all saccade, delay, and stim_type conditions in coarse, shuffled
            for sacc_ang, sine_delay, stim_type in np.random.permutation(conditions):
                dot_start = (sine_delay + sacc_start) / 1000
                p = Phase(duration=dot_start)
                p.set_visual(SimuSaccadeWithSineFlash2000, saccade2000(sacc_duration, sacc_start, sacc_ang, sine_delay,
                                                                       sine_dur, sine_amp, sine_freq, baseline_lum,
                                                                       contrast))
                self.add_phase(p)

                if stim_type < 0:
                    p = Phase(duration=1)
                    p.set_visual(SimuSaccadeWithSineFlash2000, saccade2000(0, 0, 0, 0, 500, 0.5, 2, baseline_lum,
                                                                           contrast))
                    self.add_phase(p)
                elif stim_type > 0:
                    p = Phase(duration=1.5)
                    p.set_visual(MovingDotOnTexture2000, dot2000(luminance, contrast, motion_axis, dot_polarity,
                                                                 dot_start_ang, dot_ang_vel, dot_ang_diameter,
                                                                 dot_offset))
                    self.add_phase(p)

        # black at the end of the protocol
        p = Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)
