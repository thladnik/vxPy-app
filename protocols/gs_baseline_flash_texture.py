from vxpy.core.protocol import StaticProtocol, Phase
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithStepFlash4000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithStepFlash2000

class BaselineFlashTextureScreening(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        #30 seconds just texture (no flash)
        p = Phase(duration=10)
        p.set_visual(SimuSaccadeWithStepFlash4000,
                     {SimuSaccadeWithStepFlash4000.saccade_duration: 100,
                      SimuSaccadeWithStepFlash4000.saccade_start_time: 1500,
                      SimuSaccadeWithStepFlash4000.saccade_target_angle: 0,
                      SimuSaccadeWithStepFlash4000.sine_start_time: 1500,
                      SimuSaccadeWithStepFlash4000.sine_duration: 500,
                      SimuSaccadeWithStepFlash4000.sine_amp: 0,
                      SimuSaccadeWithStepFlash4000.sine_freq: 30,
                      SimuSaccadeWithStepFlash4000.baseline_lum: 0.75,
                      SimuSaccadeWithStepFlash4000.contrast: 0.5})
        self.add_phase(p)

        #10 repeats of baseline flash with fine texture
        for i in range(10):

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

'''class BaselineFlashTextureCoarse(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        #30 seconds just texture (no flash)
        p = Phase(duration=30)
        p.set_visual(SimuSaccadeWithSineFlash2000,
            {SimuSaccadeWithSineFlash2000.rotation_duration: 100,
            SimuSaccadeWithSineFlash2000.rotation_start_time: 1500,
            SimuSaccadeWithSineFlash2000.rotation_amplitude: 0,
            SimuSaccadeWithSineFlash2000.flash_start_time: 1500,
            SimuSaccadeWithSineFlash2000.flash_duration: 500,
            SimuSaccadeWithSineFlash2000.sine_amp: 0,
            SimuSaccadeWithSineFlash2000.flash_freq: 2,
            SimuSaccadeWithSineFlash2000.baseline_lum: 0.75,
            SimuSaccadeWithSineFlash2000.contrast: 0.5})
        self.add_phase(p)

        #10 repeats of baseline flash with fine texture
        for i in range(10):

            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithSineFlash2000,
                     {SimuSaccadeWithSineFlash2000.rotation_duration: 100,
                      SimuSaccadeWithSineFlash2000.rotation_start_time: 1500,
                      SimuSaccadeWithSineFlash2000.rotation_amplitude: 0,
                      SimuSaccadeWithSineFlash2000.flash_start_time: 1500,
                      SimuSaccadeWithSineFlash2000.flash_duration: 500,
                      SimuSaccadeWithSineFlash2000.sine_amp: 0.5,
                      SimuSaccadeWithSineFlash2000.flash_freq: 2,
                      SimuSaccadeWithSineFlash2000.baseline_lum: 0.75,
                      SimuSaccadeWithSineFlash2000.contrast: 0.5})
            self.add_phase(p)'''