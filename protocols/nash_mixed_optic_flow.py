import vxpy.core.protocol as vxprotocol

from visuals.cmn_demo_from_Nash.sph_cmn import SphGlobalFlow


class MixedOFlowVaryingRotationTranslationSpeed(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        translation_speeds = [-40, -40, -40, -40, -40, 40,  40, 40,  40, 40]
        rotation_speeds    = [  0, -20,  20, -40,  40,  0, -20, 20, -40, 40]
        for i in range(len(translation_speeds)):
            p = vxprotocol.Phase(duration=120)
            p.set_visual(SphGlobalFlow,
                         {SphGlobalFlow.p_trans_azi: 90,
                          SphGlobalFlow.p_trans_elv: 0,
                          SphGlobalFlow.p_trans_speed: translation_speeds[i],
                          SphGlobalFlow.p_rot_azi: 0,
                          SphGlobalFlow.p_rot_elv: 90,
                          SphGlobalFlow.p_rot_speed: rotation_speeds[i],
                          SphGlobalFlow.p_tex_scale: -0.9,
                          })
            self.add_phase(p)
