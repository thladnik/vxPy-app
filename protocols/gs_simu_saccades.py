from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
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

        # Fix seed
        np.random.seed(1)

        # set fixed parameters
        sacc_duration = 100
        sacc_start = 1500
        sine_dur = 500
        sine_freq = 2
        baseline_lum = 0.75
        contrast = 0.5

        # experimental conditions
        conditions = [()]

        # 10 seconds just texture (no flash)
        p = Phase(duration=10)
        p.set_visual(SimuSaccadeWithSineFlash4000, params4000(sacc_duration, sacc_start, 0, 500, sine_dur, 0, sine_freq,
                                                              baseline_lum, contrast))
        self.add_phase(p)

        #10 repeats of baseline flash with fine texture
        for i in range(4):


            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithStepFlash4000,
                     {SimuSaccadeWithStepFlash4000.saccade_duration: 100,
                      SimuSaccadeWithStepFlash4000.saccade_start_time: 1500,
                      SimuSaccadeWithStepFlash4000.saccade_target_angle: 0,
                      SimuSaccadeWithStepFlash4000.sine_start_time: 1500,
                      SimuSaccadeWithStepFlash4000.sine_duration: 30,
                      SimuSaccadeWithStepFlash4000.sine_amp: 0.5,
                      SimuSaccadeWithStepFlash4000.sine_freq: 30,
                      SimuSaccadeWithStepFlash4000.baseline_lum: 0.75,
                      SimuSaccadeWithStepFlash4000.contrast: 0.5})
            self.add_phase(p)

        p = Phase(duration=10)
        p.set_visual(SimuSaccadeWithStepFlash2000,
                     {SimuSaccadeWithStepFlash2000.saccade_duration: 100,
                      SimuSaccadeWithStepFlash2000.saccade_start_time: 1500,
                      SimuSaccadeWithStepFlash2000.saccade_target_angle: 0,
                      SimuSaccadeWithStepFlash2000.sine_start_time: 1500,
                      SimuSaccadeWithStepFlash2000.sine_duration: 500,
                      SimuSaccadeWithStepFlash2000.sine_amp: 0,
                      SimuSaccadeWithStepFlash2000.sine_freq: 2,
                      SimuSaccadeWithStepFlash2000.baseline_lum: 0.75,
                      SimuSaccadeWithStepFlash2000.contrast: 0.5})
        self.add_phase(p)

        for i in range(10):

            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithStepFlash2000,
                     {SimuSaccadeWithStepFlash2000.saccade_duration: 100,
                      SimuSaccadeWithStepFlash2000.saccade_start_time: 1500,
                      SimuSaccadeWithStepFlash2000.saccade_target_angle: 0,
                      SimuSaccadeWithStepFlash2000.sine_start_time: 1500,
                      SimuSaccadeWithStepFlash2000.sine_duration: 30,
                      SimuSaccadeWithStepFlash2000.sine_amp: 0.5,
                      SimuSaccadeWithStepFlash2000.sine_freq: 30,
                      SimuSaccadeWithStepFlash2000.baseline_lum: 0.75,
                      SimuSaccadeWithStepFlash2000.contrast: 0.5})
            self.add_phase(p)