import numpy as np

import vxpy.core.protocol as vxprotocol
import vxpy.core.event as vxevent

from visuals.texture_flash_saccadeTrigger.saccade_flash_saccadeTrigger import SaccadeFlash_SaccadeTriggerStimulation4000, SaccadeFlash_SaccadeTriggerStimulation2000 # phase saccade triggered
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash2000
from visuals.gs_saccadic_suppression.gs_simu_saccade_sine_flash import SimuSaccadeWithSineFlash4000
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground  # for setting screen to black


class ROITriggered_Coarse(vxprotocol.TriggeredProtocol):
    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        # set parameters (phases saccade triggered)
        flash_conditions_list = [0.02, 0.1, 0.25, 0.5, 1, 2, 4, -1, 6.5, 6.5] # -1 for flash whithout saccade (wie oft soll das gezeigt werden?)
        flash_conditions_numIter = 1  # eigentlich 4 -> da links und rechts oder noch mehr

        flash_conditions_fullList = np.repeat(flash_conditions_list, flash_conditions_numIter)  # np.array with all iterations of stimulus conditions
        np.random.seed(0)  # fix seed
        flash_conditions_shuffeled = flash_conditions_fullList.copy()  # make deep (not affecting) copy of condition list
        np.random.shuffle(flash_conditions_shuffeled)  # shuffle condition List

        # Set tigger that controls progression of this protocol
        trigger = vxevent.RisingEdgeTrigger('roi_activity_trigger')
        self.set_phase_trigger(trigger)

        # delay für 1. phase
        # 30 seconds just texture (no flash)
        p = vxprotocol.Phase(duration=10)
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