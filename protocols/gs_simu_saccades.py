from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithStepFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithStepFlash4000
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


def params2000step(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
    return {
        SimuSaccadeWithStepFlash2000.saccade_duration: sacc_duration,
        SimuSaccadeWithStepFlash2000.saccade_start_time: sacc_start,
        SimuSaccadeWithStepFlash2000.saccade_target_angle: sacc_ang,
        SimuSaccadeWithStepFlash2000.sine_start_time: sine_start,
        SimuSaccadeWithStepFlash2000.sine_duration: sine_dur,
        SimuSaccadeWithStepFlash2000.sine_amp: sine_amp,
        SimuSaccadeWithStepFlash2000.sine_freq: sine_freq,
        SimuSaccadeWithStepFlash2000.baseline_lum: baseline_lum,
        SimuSaccadeWithStepFlash2000.contrast: contrast
    }


def params4000step(sacc_duration, sacc_start, sacc_ang, sine_start, sine_dur, sine_amp, sine_freq, baseline_lum, contrast):
    return {
        SimuSaccadeWithStepFlash4000.saccade_duration: sacc_duration,
        SimuSaccadeWithStepFlash4000.saccade_start_time: sacc_start,
        SimuSaccadeWithStepFlash4000.saccade_target_angle: sacc_ang,
        SimuSaccadeWithStepFlash4000.sine_start_time: sine_start,
        SimuSaccadeWithStepFlash4000.sine_duration: sine_dur,
        SimuSaccadeWithStepFlash4000.sine_amp: sine_amp,
        SimuSaccadeWithStepFlash4000.sine_freq: sine_freq,
        SimuSaccadeWithStepFlash4000.baseline_lum: baseline_lum,
        SimuSaccadeWithStepFlash4000.contrast: contrast
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
            for j in range(4):
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


class TextureDisplacementStepFlash50Hz(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters
        sacc_duration = 100
        sacc_start = 1500
        sine_dur = 30
        sine_freq = 2
        baseline_lum = 0.75
        contrast = 0.5

        # experimental conditions, (sacc_ang, sine_start, sine_amp)
        conditions = [(-30, -500, 0.5), (-30, 20, 0.5), (-30, 100, 0.5), (-30, 250, 0.5), (-30, 500, 0.5),
                      (-30, 1000, 0.5), (-30, 2000, 0.5), (-30, 4000, 0.5), (-30, 111, 0), (30, -500, 0.5),
                      (30, 20, 0.5), (30, 100, 0.5), (30, 250, 0.5), (30, 500, 0.5), (30, 1000, 0.5),
                      (30, 2000, 0.5), (30, 4000, 0.5), (30, 111, 0), (0, 333, 0.5), (0, 333, 0.5)]

        # 10 seconds just texture (no flash)
        for i in range(3):
            # 10 seconds just texture (coarse)
            p = Phase(duration=10)
            p.set_visual(SimuSaccadeWithStepFlash2000, params2000step(sacc_duration, sacc_start, 0, 777, sine_dur, 0,
                                                                      sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # 4 repeats of all delay and saccade conditions in coarse
            for j in range(1):
                for sacc_ang, sine_delay, sine_amp in np.random.permutation(conditions):
                    sine_start = sine_delay + sacc_start
                    p = Phase(duration=8)
                    p.set_visual(SimuSaccadeWithStepFlash2000, params2000step(sacc_duration, sacc_start, sacc_ang,
                                                                              sine_start,  sine_dur, sine_amp, sine_freq,
                                                                              baseline_lum, contrast))
                    self.add_phase(p)

            # 10 seconds just texture (fine)
            p = Phase(duration=10)
            p.set_visual(SimuSaccadeWithStepFlash4000, params4000step(sacc_duration, sacc_start, 0, 777, sine_dur, 0,
                                                                      sine_freq, baseline_lum, contrast))
            self.add_phase(p)

            # 4 repeats of all delay and saccade conditions
            for k in range(1):
                for sacc_ang, sine_delay, sine_amp in np.random.permutation(conditions):
                    sine_start = sine_delay + sacc_start
                    p = Phase(duration=8)
                    p.set_visual(SimuSaccadeWithStepFlash4000, params4000step(sacc_duration, sacc_start, sacc_ang,
                                                                              sine_start, sine_dur, sine_amp, sine_freq,
                                                                              baseline_lum, contrast))
                    self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)
