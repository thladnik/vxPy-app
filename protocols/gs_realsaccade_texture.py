import numpy as np

from vxpy.core.protocol import StaticProtocol, Phase
from vxpy.visuals import pause

from visuals.texture_flash_NOsaccadeTrigger.saccade_flash import create_parameterset_freq2, SaccadeFlash_PhaseStimulation_2000, SaccadeFlash_PhaseStimulation_4000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000

class saccadeNoVisualFeedback_protocoll(StaticProtocol):

    #clear black 15 sec
    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        for i in range(3):

            # Texture displacement saccade protocol for fine texture (3000 blobs)


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

        #fine texture
        # setting up parameters
        self.phaseDuration_pure = 10
        self.TimeBetweenSaccades = 8
        self.flash_conditions_list = [0.02, 0.1, 0.25, 0.5, 1, 2, None]
        self.flash_conditions_numIter = 10                            # eigentlich 4 -> da links und rechts oder noch mehr
        self.phase_end_buffer = 3  # buffer at end of phase to prevent abortion of stimulus
        self.phaseDuration = self.phaseDuration_pure + self.phase_end_buffer # total duration of stimulus presentation, including buffer!

        self.flash_conditions_fullList = np.repeat(self.flash_conditions_list,
                                                   self.flash_conditions_numIter)  # np.array with all iterations of stimulus conditions
        np.random.seed(0)  # fix seed
        self.flash_conditions_shuffeled = self.flash_conditions_fullList.copy()  # make deep (not affecting) copy of condition list
        np.random.shuffle(self.flash_conditions_shuffeled)                       # shuffle condition List


        # loop through phases
        for delay in self.flash_conditions_shuffeled:

            p = Phase(duration=self.phaseDuration)

            params = create_parameterset_freq2(flash_delay=delay,
                                               phase_duration=self.phaseDuration,
                                               phase_end_buffer = self.phase_end_buffer)
            if delay is None:  # contorll condition -> Amplitude 0
                params[SaccadeFlash_PhaseStimulation_4000.flash_cos_amplitude] = 0.0

            p.set_visual(SaccadeFlash_PhaseStimulation_4000, params)
            self.add_phase(p)

        for i in range(3):

            # Texture displacement saccade protocol for fine texture (3000 blobs)

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

        for i in range(3):

            # Texture displacement saccade protocol for fine texture (3000 blobs)

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

        #coarse texture
        # setting up parameters
        SaccadeFlash_PhaseStimulation_2000.phaseDuration_pure = 10
        self.TimeBetweenSaccades = 8
        self.flash_conditions_list = [0.02, 0.1, 0.25, 0.5, 1, 2, None]
        self.flash_conditions_numIter = 10                            # eigentlich 4 -> da links und rechts oder noch mehr
        self.phase_end_buffer = 3  # buffer at end of phase to prevent abortion of stimulus
        self.phaseDuration = self.phaseDuration_pure + self.phase_end_buffer # total duration of stimulus presentation, including buffer!

        self.flash_conditions_fullList = np.repeat(self.flash_conditions_list,
                                                   self.flash_conditions_numIter)  # np.array with all iterations of stimulus conditions
        np.random.seed(0)  # fix seed
        self.flash_conditions_shuffeled = self.flash_conditions_fullList.copy()  # make deep (not affecting) copy of condition list
        np.random.shuffle(self.flash_conditions_shuffeled)                       # shuffle condition List


        # loop through phases
        for delay in self.flash_conditions_shuffeled:

            p = Phase(duration=self.phaseDuration)

            params = create_parameterset_freq2(flash_delay=delay,
                                               phase_duration=self.phaseDuration,
                                               phase_end_buffer = self.phase_end_buffer)
            if delay is None:  # contorll condition -> Amplitude 0
                params[SaccadeFlash_PhaseStimulation_2000.flash_cos_amplitude] = 0.0

            p.set_visual(SaccadeFlash_PhaseStimulation_2000, params)
            self.add_phase(p)

        for i in range(3):

            # Texture displacement saccade protocol for fine texture (3000 blobs)
            p = Phase(duration=8)
            p.set_visual(SimuSaccadeWithSineFlash2000,
                     {SimuSaccadeWithSineFlash2000.saccade_duration: 100,
                      SimuSaccadeWithSineFlash2000.saccade_start_time: 1500,
                      SimuSaccadeWithSineFlash2000.saccade_target_angle: 0,
                      SimuSaccadeWithSineFlash2000.sine_start_time: 0,
                      SimuSaccadeWithSineFlash2000.sine_duration: 500,
                      SimuSaccadeWithSineFlash2000.sine_amp: 0.5,
                      SimuSaccadeWithSineFlash2000.sine_freq: 2,
                      SimuSaccadeWithSineFlash2000.baseline_lum: 0.75,
                      SimuSaccadeWithSineFlash2000.contrast: 0.5})
            self.add_phase(p)

