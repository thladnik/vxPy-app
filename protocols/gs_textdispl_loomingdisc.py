from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_looming_disc.gs_looming_disc import LoomingDiscOnTexture2000
from visuals.gs_looming_disc.gs_looming_disc import LoomingDiscOnTexture4000
from visuals.gs_looming_disc.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_looming_disc.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
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


def disc2000(luminance, contrast, motion_axis, disc_polarity, disc_azimuth, disc_elevation, disc_starting_diameter, disc_expansion_lv):
    return {
        LoomingDiscOnTexture2000.luminance: luminance,
        LoomingDiscOnTexture2000.contrast: contrast,
        LoomingDiscOnTexture2000.motion_axis: motion_axis,
        LoomingDiscOnTexture2000.disc_polarity: disc_polarity,
        LoomingDiscOnTexture2000.disc_azimuth: disc_azimuth,
        LoomingDiscOnTexture2000.disc_elevation: disc_elevation,
        LoomingDiscOnTexture2000.disc_starting_diameter: disc_starting_diameter,
        LoomingDiscOnTexture2000.disc_expansion_lv: disc_expansion_lv
    }


def disc4000(luminance, contrast, motion_axis, disc_polarity, disc_azimuth, disc_elevation, disc_starting_diameter, disc_expansion_lv):
    return {
        LoomingDiscOnTexture4000.luminance: luminance,
        LoomingDiscOnTexture4000.contrast: contrast,
        LoomingDiscOnTexture4000.motion_axis: motion_axis,
        LoomingDiscOnTexture4000.disc_polarity: disc_polarity,
        LoomingDiscOnTexture4000.disc_azimuth: disc_azimuth,
        LoomingDiscOnTexture4000.disc_elevation: disc_elevation,
        LoomingDiscOnTexture4000.disc_starting_diameter: disc_starting_diameter,
        LoomingDiscOnTexture4000.disc_expansion_lv: disc_expansion_lv
    }


class TextureDisplacementLoomingDisc(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(0)

        # set fixed saccade parameters
        sacc_duration = 100
        sacc_start = 10000
        sine_dur = 0
        sine_freq = 0
        sine_amp = 0
        baseline_lum = 0.75
        contrast = 0.5

        # set fixed dot parameters
        luminance = 0.75
        motion_axis = 'vertical'
        disc_polarity = 'dark-on-light'
        disc_azimuth = 0
        disc_elevation = -90
        disc_starting_diameter = 2  # in °
        disc_expansion_lv = 240  # in ms

        # experimental conditions, (sacc_target, delay)
        conditions = [(-30, 100), (-30, 250), (-30, 500), (-30, 1000), (-30, 2000), (-30, 4000),
                      (30, 100), (30, 250), (30, 500), (30, 1000), (30, 2000), (30, 4000), (0, 555), (0, 555)]

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

                p = Phase(duration=2.5)
                p.set_visual(LoomingDiscOnTexture2000, disc2000(luminance, contrast, motion_axis, disc_polarity,
                                                             disc_azimuth, disc_elevation, disc_starting_diameter, disc_expansion_lv))
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

                p = Phase(duration=2.5)
                p.set_visual(LoomingDiscOnTexture4000, disc4000(luminance, contrast, motion_axis, disc_polarity,
                                                             disc_azimuth, disc_elevation, disc_starting_diameter, disc_expansion_lv))
                self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)
