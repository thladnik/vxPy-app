from vxpy.core.protocol import StaticProtocol, Phase
from visuals.ml_TextureDispl_OKR_grating import RotatingGrating
import numpy as np


# parameters function
def rot_grating(waveform, motion_type, motion_axis, ang_period, rot_start, rot_duration, rot_ang):
    return {
        RotatingGrating.waveform: waveform,
        RotatingGrating.motion_type: motion_type,
        RotatingGrating.motion_axis: motion_axis,
        RotatingGrating.angular_period: ang_period,
        RotatingGrating.rotation_start_time: rot_start,
        RotatingGrating.rotation_duration: rot_duration,
        RotatingGrating.rotation_target_angle: rot_ang
    }

class TextureDisplOKRGrating(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters (all times in ms)
        waveform = 'rectangular'
        motion_type = 'rotation'
        motion_axis = 'vertical'
        ang_period = 60
        sacc_start = 1500
        sacc_duration = 100
        okr_duration = 10000

        # experimental conditions, (sacc_ang, okr_start)
        conditions = [(-30, 100), (-30, 250), (-30, 500), (-30, 1000), (-30, 2000), (-30, 4000), (30, 100),
                      (30, 250), (30, 500), (30, 1000), (30, 2000), (30, 4000), (0, 3000), (0, 3001)]

        # 15 seconds just grating
        p = Phase(duration=15)
        p.set_visual(RotatingGrating, rot_grating(waveform, motion_type, motion_axis, ang_period, sacc_start,
                                                  sacc_duration, 0))
        self.add_phase(p)

        # 4 repeats of all delay and saccade conditions in coarse
        for j in range(4):
            for sacc_ang, okr_delay in np.random.permutation(conditions):

                # saccade phase
                sacc_phase_dur = (sacc_start + sacc_duration) / 1000
                p = Phase(duration=sacc_phase_dur)
                p.set_visual(RotatingGrating, rot_grating(waveform, motion_type, motion_axis, ang_period, sacc_start,
                                                          sacc_duration, sacc_ang))
                self.add_phase(p)

                # OKR phase
                okr_phase_dur = (okr_duration + okr_delay - 100) / 1000
                p = Phase(duration=okr_phase_dur)
                if sacc_ang == 0 and okr_delay == 3000:
                    rot_ang = 600
                elif sacc_ang == 0 and okr_delay == 3001:
                    rot_ang = -600
                else:
                    rot_ang = -sacc_ang * 20

                p.set_visual(RotatingGrating, rot_grating(waveform, motion_type, motion_axis, ang_period,
                                                          okr_delay - 100, okr_duration, rot_ang))
                self.add_phase(p)

                # Pause phase
                pause_dur = 10
                p = Phase(duration=pause_dur)
                p.set_visual(RotatingGrating, rot_grating(waveform, motion_type, motion_axis, ang_period, sacc_start,
                                                          sacc_duration, 0))
                self.add_phase(p)
