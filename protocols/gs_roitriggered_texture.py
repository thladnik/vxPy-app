import numpy as np

import vxpy.core.protocol as vxprotocol
import vxpy.core.event as vxevent

from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.gs_SaccadeTriggeredTextDisplCombined.SaccadeTriggeredTextDisplCombined import TextureRotationCosineFlash

def paramstext(rot_duration, rot_start, rot_amp, rot_dir, flash_start, flash_dur, flash_amp, flash_freq, baseline_lum, contrast):
    return {
        TextureRotationCosineFlash.rotation_duration: rot_duration,
        TextureRotationCosineFlash.rotation_start_time: rot_start,
        TextureRotationCosineFlash.rotation_amplitude: rot_amp,
        TextureRotationCosineFlash.rotation_direction: rot_dir,
        TextureRotationCosineFlash.flash_start_time: flash_start,
        TextureRotationCosineFlash.flash_amp: flash_amp,
        TextureRotationCosineFlash.flash_duration: flash_dur,
        TextureRotationCosineFlash.flash_freq: flash_freq,
        TextureRotationCosineFlash.baseline_lum: baseline_lum,
        TextureRotationCosineFlash.contrast: contrast
    }


class ROITriggered_Coarse(vxprotocol.TriggeredProtocol):

    trigger_name = ''

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        np.random.seed(1)

        # set fixed parameters texture
        rot_duration = 0
        rot_start = 0
        rot_amp = 0
        rot_dir = 0
        flash_dur_text = 500
        flash_freq_text = 2
        baseline_lum_text = 0.75
        contrast = 0.5


        # experimental conditions, (delay, flash_amp) In total 10 Conditions:
        # 7 delays, 1 ctrl saccade (no flash, delay = 333), 2 ctrl flash (8s delay) = 10 conditions
        conditions = [(20, 0.5), (100, 0.5), (250, 0.5), (500, 0.5), (1000, 0.5), (2000, 0.5), (4000, 0.5), (333, 0),
                      (8000, 0.5), (8000, 0.5)]

        # Set tigger that controls progression of this protocol
        trigger = vxevent.RisingEdgeTrigger(self.trigger_name)
        self.set_phase_trigger(trigger)

        repeats = 4

        # 15 sec just texture (flash delay = 0)
        p = vxprotocol.Phase(duration=15)
        p.set_visual(TextureRotationCosineFlash,
                         paramstext(rot_duration, rot_start, rot_amp, rot_dir, 0, flash_dur_text,
                                    0, flash_freq_text, baseline_lum_text, contrast))
        self.add_phase(p)

        for i in range(repeats):
            # shuffled conditions, texture
            for delay, flash_amp in np.random.permutation(conditions):
                if delay > 4000:
                    phase_dur = 12
                else:
                    phase_dur = 8
                p = vxprotocol.Phase(phase_dur)
                p.set_visual(TextureRotationCosineFlash, paramstext(rot_duration, rot_start, rot_amp, rot_dir, delay,
                                                                    flash_dur_text, flash_amp, flash_freq_text,
                                                                    baseline_lum_text, contrast))
                self.add_phase(p)

        p = vxprotocol.Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)


class ROITriggerd_Coarse_Layer0(ROITriggered_Coarse):

    trigger_name = 'roi_activity_trigger_0'


class ROITriggerd_Coarse_Layer1(ROITriggered_Coarse):
    trigger_name = 'roi_activity_trigger_1'

