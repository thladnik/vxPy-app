import vxpy.core.protocol as vxprotocol

from visuals.spherical_global_motion import GlobalOpticFlow


class MixedOFlowVaryingRotationTranslationSpeed(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        translation_speeds = [-40, -40, -40, -40, -40, 40,  40, 40,  40, 40]
        rotation_speeds    = [  0, -20,  20, -40,  40,  0, -20, 20, -40, 40]
        for i in range(len(translation_speeds)):
            p = vxprotocol.Phase(duration=120)
            p.set_visual(GlobalOpticFlow,
                         {GlobalOpticFlow.p_trans_azi: 90,
                          GlobalOpticFlow.p_trans_elv: 0,
                          GlobalOpticFlow.p_trans_speed: translation_speeds[i],
                          GlobalOpticFlow.p_rot_azi: 0,
                          GlobalOpticFlow.p_rot_elv: 90,
                          GlobalOpticFlow.p_rot_speed: rotation_speeds[i],
                          GlobalOpticFlow.p_tex_scale: -0.9,
                          })
            self.add_phase(p)
