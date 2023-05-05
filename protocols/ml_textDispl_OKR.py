from vxpy.core.protocol import StaticProtocol, Phase
from visuals.ml_TextureDispl_OKR import RotatingTexture2000
import numpy as np


# parameters function
def rot_texture_2000(rot_duration, rot_start, rot_ang, luminance, contrast):
    return {
        RotatingTexture2000.rotation_duration: rot_duration,
        RotatingTexture2000.rotation_start_time: rot_start,
        RotatingTexture2000.rotation_target_angle: rot_ang,
        RotatingTexture2000.luminance: luminance,
        RotatingTexture2000.contrast: contrast
    }


class TextureDisplOKR2000(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # Fix seed
        np.random.seed(1)

        # set fixed parameters (all times in ms)
        sacc_duration = 100
        sacc_start = 1500
        luminance = 0.5
        contrast = 1.
        okr_duration = 10000

        # experimental conditions, (sacc_ang, okr_start)
        conditions = [(-30, 100), (-30, 250), (-30, 500), (-30, 1000), (-30, 2000), (-30, 4000), (30, 100),
                      (30, 250), (30, 500), (30, 1000), (30, 2000), (30, 4000), (0, 3000)]

        # 10 seconds just texture (no flash)
        # 10 seconds just texture (coarse)
        p = Phase(duration=15)
        p.set_visual(RotatingTexture2000, rot_texture_2000(sacc_duration, sacc_start, 0, luminance, contrast))
        self.add_phase(p)

        # 4 repeats of all delay and saccade conditions in coarse
        for j in range(4):
            for sacc_ang, okr_delay in np.random.permutation(conditions):
                sacc_phase_dur = (sacc_start + sacc_duration) / 1000
                p = Phase(duration=sacc_phase_dur)
                p.set_visual(RotatingTexture2000,
                             rot_texture_2000(sacc_duration, sacc_start, sacc_ang, luminance, contrast))
                self.add_phase(p)

                okr_phase_dur = (okr_duration + okr_delay)/1000
                p = Phase(duration=okr_phase_dur)
                p.set_visual(RotatingTexture2000,
                             rot_texture_2000(okr_duration, okr_delay, -sacc_ang * okr_duration / 1000 * 2, luminance,
                                              contrast))
                self.add_phase(p)

                pause_dur = 10
                p = Phase(duration=pause_dur)
                p.set_visual(RotatingTexture2000, rot_texture_2000(sacc_duration, sacc_start, 0, luminance, contrast))
                self.add_phase(p)

