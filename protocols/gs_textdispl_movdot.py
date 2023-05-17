from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_movingDotOnTexture.gs_dot_on_texture import MovingDotOnTexture2000
from visuals.gs_movingDotOnTexture.gs_dot_on_texture import MovingDotOnTexture4000
from visuals.gs_movingDotOnTexture.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_movingDotOnTexture.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
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


def saccade4000(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
    return {
        SimuSaccadeWithSineFlash4000.saccade_duration: sacc_duration,
        SimuSaccadeWithSineFlash4000.saccade_start_time: sacc_start,
        SimuSaccadeWithSineFlash4000.saccade_target_angle: sacc_ang,
        SimuSaccadeWithSineFlash4000.sine_start_time: sine_start,
        SimuSaccadeWithSineFlash4000.sine_duration: sine_dur,
        SimuSaccadeWithSineFlash4000.sine_amp: sine_amp,
        SimuSaccadeWithSineFlash4000.sine_freq: sine_freq,
        SimuSaccadeWithSineFlash4000.baseline_lum: baseline_lum,
        SimuSaccadeWithSineFlash4000.contrast: contrast
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


def dot4000(luminance, contrast, motion_axis, dot_polarity, dot_start_ang, dot_ang_vel, dot_ang_diameter, dot_offset):
    return {
        MovingDotOnTexture4000.luminance: luminance,
        MovingDotOnTexture4000.contrast: contrast,
        MovingDotOnTexture4000.motion_axis: motion_axis,
        MovingDotOnTexture4000.dot_polarity: dot_polarity,
        MovingDotOnTexture4000.dot_start_angle: dot_start_ang,
        MovingDotOnTexture4000.dot_angular_velocity: dot_ang_vel,
        MovingDotOnTexture4000.dot_angular_diameter: dot_ang_diameter,
        MovingDotOnTexture4000.dot_offset_angle: dot_offset
    }


class TextureDisplacementMovingDot(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(0)

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

        # experimental conditions, (sacc_target, delay)
        conditions = [(-30, 100), (-30, 250), (-30, 500), (-30, 1000), (-30, 2000), (-30, 4000), (-30, 6000),
                      (30, 100), (30, 250), (30, 500), (30, 1000), (30, 2000), (30, 4000), (30, 6000), (0, 500), (0, 500)]

        # 4 repeats af all delay conditions coarse and fine alternating
        for i in range(4):
            # 5 seconds just texture (coarse)
            p = Phase(duration=5)
            p.set_visual(SimuSaccadeWithSineFlash2000, saccade2000(sacc_duration, sacc_start, 0, 111, sine_dur, sine_amp,
                                                                   sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # all delay and saccade conditions in coarse
            for sacc_ang, sine_delay in np.random.permutation(conditions):
                dot_start = (sine_delay + sacc_start) / 1000
                p = Phase(duration=dot_start)
                p.set_visual(SimuSaccadeWithSineFlash2000, saccade2000(sacc_duration, sacc_start, sacc_ang, sine_delay,
                                                                       sine_dur, sine_amp, sine_freq, baseline_lum,
                                                                       contrast))
                self.add_phase(p)

                p = Phase(duration=1.5)
                p.set_visual(MovingDotOnTexture2000, dot2000(luminance, contrast, motion_axis, dot_polarity,
                                                             dot_start_ang, dot_ang_vel, dot_ang_diameter, dot_offset))
                self.add_phase(p)

            # 10 seconds just texture (fine)
            p = Phase(duration=5)
            p.set_visual(SimuSaccadeWithSineFlash4000, saccade4000(sacc_duration, sacc_start, 0, 111, sine_dur,
                                                                   sine_amp, sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # 4 repeats of all delay and saccade conditions

            for sacc_ang, sine_delay in np.random.permutation(conditions):
                dot_start = (sine_delay + sacc_start) / 1000
                p = Phase(duration=dot_start)
                p.set_visual(SimuSaccadeWithSineFlash4000, saccade4000(sacc_duration, sacc_start, sacc_ang, sine_delay,
                                                                       sine_dur, sine_amp, sine_freq, baseline_lum,
                                                                       contrast))
                self.add_phase(p)

                p = Phase(duration=1.5)
                p.set_visual(MovingDotOnTexture4000, dot4000(luminance, contrast, motion_axis, dot_polarity,
                                                             dot_start_ang, dot_ang_vel, dot_ang_diameter, dot_offset))
                self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)
