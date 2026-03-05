from vxpy.core.protocol import StaticProtocol, Phase
from vxpy.visuals import pause
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

from visuals.jitter_noise import \
    BinaryBlackWhiteJitterNoise8deg, \
    BinaryBlackWhiteJitterNoise16deg, \
    BinaryBlackWhiteJitterNoise8deg2Hz, \
    BinaryBlackWhiteJitterNoise60deg, \
    BinaryBlackWhiteJitterNoise30deg, \
    BinaryBlackWhiteJitterNoise10deg


class BinaryJitterNoise(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        p = Phase(1*30, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(30*60)
        p.set_visual(BinaryBlackWhiteJitterNoise8deg)
        self.add_phase(p)

        p = Phase(1*30, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(30*60)
        p.set_visual(BinaryBlackWhiteJitterNoise16deg)
        self.add_phase(p)

        p = Phase(1*30, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


class BinaryJitterNoiseSingle(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        p = Phase(1*30, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(30*60)
        p.set_visual(BinaryBlackWhiteJitterNoise8deg2Hz)
        self.add_phase(p)

        p = Phase(1*30, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

class BinaryJitterNoiseGS(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        p = Phase(60, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(600)
        p.set_visual(BinaryBlackWhiteJitterNoise60deg)
        self.add_phase(p)

        p = Phase(60, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(30*60)
        p.set_visual(BinaryBlackWhiteJitterNoise30deg)
        self.add_phase(p)

        p = Phase(60, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(60 * 60)
        p.set_visual(BinaryBlackWhiteJitterNoise10deg)
        self.add_phase(p)

        p = Phase(60, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        p = Phase(2, visual=SphereUniformBackground,
                  visual_params={SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)