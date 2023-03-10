from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000

class BaselineFlashTextureFine(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        #30 seconds just texture (no flash)
        p = Phase(duration=30)
        p.set_visual(SimuSaccadeWithSineFlash4000,
            {SimuSaccadeWithSineFlash4000.saccade_duration: 100,
            SimuSaccadeWithSineFlash4000.saccade_start_time: 1500,
            SimuSaccadeWithSineFlash4000.saccade_target_angle: 0,
            SimuSaccadeWithSineFlash4000.sine_start_time: 1500,
            SimuSaccadeWithSineFlash4000.sine_duration: 500,
            SimuSaccadeWithSineFlash4000.sine_amp: 0,
            SimuSaccadeWithSineFlash4000.sine_freq: 2,
            SimuSaccadeWithSineFlash4000.baseline_lum: 0.75,
            SimuSaccadeWithSineFlash4000.contrast: 0.5})
        self.add_phase(p)

        #10 repeats of baseline flash with fine texture
        for i in range(10):

            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithSineFlash4000,
                     {SimuSaccadeWithSineFlash4000.saccade_duration: 100,
                      SimuSaccadeWithSineFlash4000.saccade_start_time: 1500,
                      SimuSaccadeWithSineFlash4000.saccade_target_angle: 0,
                      SimuSaccadeWithSineFlash4000.sine_start_time: 1500,
                      SimuSaccadeWithSineFlash4000.sine_duration: 500,
                      SimuSaccadeWithSineFlash4000.sine_amp: 0.5,
                      SimuSaccadeWithSineFlash4000.sine_freq: 2,
                      SimuSaccadeWithSineFlash4000.baseline_lum: 0.75,
                      SimuSaccadeWithSineFlash4000.contrast: 0.5})
            self.add_phase(p)

class BaselineFlashTextureCoarse(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        #30 seconds just texture (no flash)
        p = Phase(duration=30)
        p.set_visual(SimuSaccadeWithSineFlash2000,
            {SimuSaccadeWithSineFlash2000.saccade_duration: 100,
            SimuSaccadeWithSineFlash2000.saccade_start_time: 1500,
            SimuSaccadeWithSineFlash2000.saccade_target_angle: 0,
            SimuSaccadeWithSineFlash2000.sine_start_time: 1500,
            SimuSaccadeWithSineFlash2000.sine_duration: 500,
            SimuSaccadeWithSineFlash2000.sine_amp: 0,
            SimuSaccadeWithSineFlash2000.sine_freq: 2,
            SimuSaccadeWithSineFlash2000.baseline_lum: 0.75,
            SimuSaccadeWithSineFlash2000.contrast: 0.5})
        self.add_phase(p)

        #10 repeats of baseline flash with fine texture
        for i in range(10):

            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithSineFlash2000,
                     {SimuSaccadeWithSineFlash2000.saccade_duration: 100,
                      SimuSaccadeWithSineFlash2000.saccade_start_time: 1500,
                      SimuSaccadeWithSineFlash2000.saccade_target_angle: 0,
                      SimuSaccadeWithSineFlash2000.sine_start_time: 1500,
                      SimuSaccadeWithSineFlash2000.sine_duration: 500,
                      SimuSaccadeWithSineFlash2000.sine_amp: 0.5,
                      SimuSaccadeWithSineFlash2000.sine_freq: 2,
                      SimuSaccadeWithSineFlash2000.baseline_lum: 0.75,
                      SimuSaccadeWithSineFlash2000.contrast: 0.5})
            self.add_phase(p)