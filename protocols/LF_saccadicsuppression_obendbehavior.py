from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithStepFlash2000
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
import numpy as np


def params2000step(sacc_duration, sacc_start, sacc_ang, flash_start, flash_dur, flash_amp, flash_freq, baseline_lum, contrast):
    return {
        SimuSaccadeWithStepFlash2000.saccade_duration: sacc_duration,
        SimuSaccadeWithStepFlash2000.saccade_start_time: sacc_start,
        SimuSaccadeWithStepFlash2000.saccade_target_angle: sacc_ang,
        SimuSaccadeWithStepFlash2000.sine_start_time: flash_start,
        SimuSaccadeWithStepFlash2000.sine_duration: flash_dur,
        SimuSaccadeWithStepFlash2000.sine_amp: flash_amp,
        SimuSaccadeWithStepFlash2000.sine_freq: flash_freq,
        SimuSaccadeWithStepFlash2000.baseline_lum: baseline_lum,
        SimuSaccadeWithStepFlash2000.contrast: contrast
    }


class FreeSwimmingTextureDisplacementFlash(StaticProtocol):
    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters
        sacc_duration = 100
        sacc_start = 1500
        flash_dur = 500
        flash_freq = 2
        flash_amp= 0.3
        baseline_lum = 0.75
        contrast = 0.5

        # experimental conditions: (delay, sacc_ang, sine_amp)
        conditions = [(-500, 30, flash_amp), (-500, -30, flash_amp),
                      (20, 30, flash_amp), (20, -30, flash_amp),
                      (100, 30, flash_amp), (100, -30, flash_amp),
                      (250, 30, flash_amp), (250, -30, flash_amp),
                      (500, 30, flash_amp), (500, -30, flash_amp),
                      (1000, 30, flash_amp), (1000, -30, flash_amp),
                      (2000, 30, flash_amp), (2000, -30, flash_amp),
                      (4000, 30, flash_amp), (4000, -30, flash_amp),
                      (-1, 30, 0), (-1, -30, 0),  # condition: texturedisplacement only
                      (0, 0, flash_amp), (0, 0, flash_amp)  # condition: flash only
                      ]

        # show only-texture at beginning -> habituation
        p = Phase(duration=5)
        p.set_visual(SimuSaccadeWithStepFlash2000, params2000step(0, 0, 0,
                                                          0, 0, 0,flash_freq, baseline_lum, contrast))
        self.add_phase(p)

        for i in range(5):
            # pause phase
            p = Phase(duration=10)
            p.set_visual(SimuSaccadeWithStepFlash2000, params2000step(sacc_duration, sacc_start, 0, 777, flash_dur, 0,
                                                                      flash_freq, baseline_lum, contrast))
            self.add_phase(p)

            # stimulus phase
            for flash_delay, sacc_ang, flash_amp in np.random.permutation(conditions):
                flash_start = flash_delay + sacc_start
                p = Phase(duration=6)
                p.set_visual(SimuSaccadeWithStepFlash2000, params2000step(sacc_duration, sacc_start, sacc_ang,
                                                                              flash_start, flash_dur, flash_amp, flash_freq,
                                                                              baseline_lum, contrast))
                self.add_phase(p)