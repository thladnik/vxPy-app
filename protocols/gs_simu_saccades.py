from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
import numpy as np


def params2000(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
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


def params4000(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
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


class TextureDisplacementSineFlash(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters
        sacc_duration = 100
        sacc_start = 1500
        sine_dur = 500
        sine_freq = 2
        baseline_lum = 0.75
        contrast = 0.5

        # experimental conditions, (sacc_ang, sine_start, sine_amp)
        conditions = [(-30, -500, 0.5), (-30, 20, 0.5), (-30, 100, 0.5), (-30, 250, 0.5), (-30, 500, 0.5),
                      (-30, 1000, 0.5), (-30, 2000, 0.5), (-30, 4000, 0.5), (-30, 500, 0), (30, -500, 0.5),
                      (30, 20, 0.5), (30, 100, 0.5), (30, 250, 0.5), (30, 500, 0.5), (30, 1000, 0.5),
                      (30, 2000, 0.5), (30, 4000, 0.5), (30, 500, 0), (0, 3000, 0.5), (0, 3000, 0.5)]

        # 10 seconds just texture (no flash)
        for i in range(2):
            # 10 seconds just texture (coarse)
            p = Phase(duration=10)
            p.set_visual(SimuSaccadeWithSineFlash2000, params2000(sacc_duration, sacc_start, 0, 500, sine_dur, 0,
                                                                  sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # 4 repeats of all delay and saccade conditions in coarse
            for j in range(1):
                for sacc_ang, sine_delay, sine_amp in np.random.permutation(conditions):
                    sine_start = sine_delay + sacc_start
                    p = Phase(duration=8)
                    p.set_visual(SimuSaccadeWithSineFlash2000, params2000(sacc_duration, sacc_start, sacc_ang,
                                                                          sine_start,  sine_dur, sine_amp, sine_freq,
                                                                          baseline_lum, contrast))
                    self.add_phase(p)

            # 10 seconds just texture (fine)
            p = Phase(duration=10)
            p.set_visual(SimuSaccadeWithSineFlash4000, params4000(sacc_duration, sacc_start, 0, 500, sine_dur, 0,
                                                                  sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # 4 repeats of all delay and saccade conditions
            for k in range(4):
                for sacc_ang, sine_delay, sine_amp in np.random.permutation(conditions):
                    sine_start = sine_delay + sacc_start
                    p = Phase(duration=8)
                    p.set_visual(SimuSaccadeWithSineFlash4000, params4000(sacc_duration, sacc_start, sacc_ang,
                                                                          sine_start, sine_dur, sine_amp, sine_freq,
                                                                          baseline_lum, contrast))
                    self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
            self.add_phase(p)
