import numpy as np

import vxpy.core.protocol as vxprotocol
import vxpy.core.event as vxevent

from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.gs_SaccadeTriggeredTextDisplCombined.SaccadeTriggeredTextDisplCombined import TextureRotationCosineFlash


def params(rot_duration, rot_start, rot_amp, rot_dir, flash_start, flash_dur, flash_amp, flash_freq, baseline_lum, contrast):
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


class SaccadeTriggeredTextDisplCombined(vxprotocol.TriggeredProtocol):
    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters
        rot_duration = 100
        rot_start = 8000
        rot_amp = 30
        flash_dur = 500
        flash_freq = 2
        baseline_lum = 0.75
        contrast = 0.5

        # experimental conditions, (rot_direction (-1,0,1), delay, flash_amp) In total 26 Conditions:
        # 7 delays left textdispl + 7 delays right textdispl + 7 delays realsacc + 1 ctrl left textdispl no flash +
        # 1 ctrl right textdispl no flash + 1 ctrl realsacc no flash + 2 ctrl realsacc long delay flash (BL) = 26
        conditions = [(-1, 20, 0.5), (-1, 100, 0.5), (-1, 250, 0.5), (-1, 500, 0.5),
                      (-1, 1000, 0.5), (-1, 2000, 0.5), (-1, 4000, 0.5), (-1, 333, 0),
                      (1, 20, 0.5), (1, 100, 0.5), (1, 250, 0.5), (1, 500, 0.5), (1, 1000, 0.5),
                      (1, 2000, 0.5), (1, 4000, 0.5), (1, 333, 0), (0, 20, 0.5), (0, 100, 0.5),
                      (0, 250, 0.5), (0, 500, 0.5), (0, 1000, 0.5), (0, 2000, 0.5),
                      (0, 4000, 0.5), (0, 333, 0), (0, 6500, 0.5), (0, 6500, 0.5)]

        # Set tigger that controls progression of this protocol
        trigger = vxevent.RisingEdgeTrigger('eyepos_saccade_trigger')
        self.set_phase_trigger(trigger)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(TextureRotationCosineFlash, params(rot_duration, rot_start, rot_amp, 0, flash_dur, 0, flash_freq,
                                                        0, baseline_lum, contrast))
        self.add_phase(p)

        for j in range(3):
            for rot_dir, delay, flash_amp in np.random.permutation(conditions):
                if rot_dir == 0:
                    p = vxprotocol.Phase(duration=10)
                    p.set_visual(TextureRotationCosineFlash, params(rot_duration, rot_start, rot_amp, rot_dir, delay,
                                                                    flash_amp, flash_dur, flash_freq, baseline_lum,
                                                                    contrast))
                else:
                    p = vxprotocol.Phase(duration=18)
                    p.set_visual(TextureRotationCosineFlash, params(rot_duration, rot_start, rot_amp, rot_dir,
                                                                    rot_start + delay, flash_amp, flash_dur, flash_freq,
                                                                    baseline_lum, contrast))
                self.add_phase(p)

        p = vxprotocol.Phase(duration=5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([0, 0, 0])})
        self.add_phase(p)


