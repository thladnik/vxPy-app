import numpy as np

import vxpy.core.protocol as vxprotocol
import vxpy.core.event as vxevent
import vxpy.core.attribute as vxattribute

from visuals.ml_saccadetriggered_OKR.ml_rotating_texture import RotatingTexture2000
from vxpy.extras.zf_eyeposition_tracking_GS import EyePositionDetectionRoutine


# parameters function
def rot_texture_2000(rot_duration, rot_start, rot_ang, luminance, contrast):
    return {
        RotatingTexture2000.rotation_duration: rot_duration,
        RotatingTexture2000.rotation_start_time: rot_start,
        RotatingTexture2000.rotation_target_angle: rot_ang,
        RotatingTexture2000.luminance: luminance,
        RotatingTexture2000.contrast: contrast
    }


class SaccadeTriggeredOKR(vxprotocol.TriggeredProtocol):
    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters (all times in ms)
        luminance = 0.5
        contrast = 1.
        okr_duration = 10000
        ang_vel = 60

        # OKR delays in ms
        okr_delays = [100, 250, 500, 1000, 2000, 4000]

        # Set saccade trigger as phase trigger
        trigger = vxevent.NotNullTrigger(f'{EyePositionDetectionRoutine.le_sacc_direction_prefix}0')
        self.set_phase_trigger(trigger)

        p = vxprotocol.Phase(duration=15)
        p.set_visual(RotatingTexture2000, rot_texture_2000(okr_duration, 0, 0, luminance, contrast))
        self.add_phase(p)

        for j in range(4):
            for okr_delay in np.random.permutation(okr_delays):
                p = vxprotocol.Phase(duration=20)
                # if vxattribute.read_attribute(EyePositionDetectionRoutine.le_sacc_prefix) > 0:
                #     sacc_dir = vxattribute.read_attribute(EyePositionDetectionRoutine.le_sacc_direction_prefix)
                # else:
                #     sacc_dir = vxattribute.read_attribute(EyePositionDetectionRoutine.re_sacc_direction_prefix)
                p.set_visual(RotatingTexture2000,
                             rot_texture_2000(okr_duration, okr_delay, ang_vel * okr_duration * 1/ 1000, luminance,
                                              contrast))
                self.add_phase(p)
