import numpy as np

from vxpy.core.protocol import Phase, StaticPhasicProtocol
from vxpy.visuals import pause
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals import gs_flash_tests as gsft


class Protocol01(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        contrast = 0.5
        default_lum = 0.5

        # Baseline phase
        p = Phase(15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0., 0., 0.)})
        self.add_phase(p)

        # Static phase
        p = Phase(5)
        p.set_visual(gsft.TextureModulateContrLum,
                     {gsft.TextureModulateContrLum.contrast: contrast,
                      gsft.TextureModulateContrLum.luminance: default_lum})
        self.add_phase(p)

        for flash_sign in [-1, 1, -1, 1, -1, 1]:
            p = Phase(duration=6)
            p.set_visual(gsft.TextureFlash,
                         {gsft.TextureFlash.contrast: contrast,
                          gsft.TextureFlash.start_luminance: default_lum,
                          gsft.TextureFlash.flash_start: 1.0,
                          gsft.TextureFlash.flash_duration: 20,
                          gsft.TextureFlash.flash_luminance: default_lum + 0.25 * flash_sign,
                          gsft.TextureFlash.end_luminance: default_lum})
            self.add_phase(p)

        for flash_sign in [-1, 1, -1, 1, -1, 1]:
            p = Phase(duration=6)
            p.set_visual(gsft.TextureFlash,
                         {gsft.TextureFlash.contrast: contrast,
                          gsft.TextureFlash.start_luminance: default_lum,
                          gsft.TextureFlash.flash_start: 1.0,
                          gsft.TextureFlash.flash_duration: 500,
                          gsft.TextureFlash.flash_luminance: default_lum + 0.25 * flash_sign,
                          gsft.TextureFlash.end_luminance: default_lum})
            self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(gsft.TextureModulateContrLum,
                     {gsft.TextureModulateContrLum.contrast: contrast,
                      gsft.TextureModulateContrLum.luminance: default_lum})
        self.add_phase(p)

        for freq, dur in [(3, 0.5), (3, 1.0), (3, 3.0), (10, 0.5), (10, 1.0), (10, 3.0)]:
            p = Phase(duration=dur)
            p.set_visual(gsft.TextureSinusLumModulation,
                         {gsft.TextureSinusLumModulation.contrast: contrast,
                          gsft.TextureSinusLumModulation.luminance: default_lum,
                          gsft.TextureSinusLumModulation.sine_luminance_frequency: freq,
                          gsft.TextureSinusLumModulation.sine_luminance_amplitude: 0.5,
                          gsft.TextureSinusLumModulation.sine_mean_luminance: default_lum})
            self.add_phase(p)

            p = Phase(duration=8-dur)
            p.set_visual(gsft.TextureModulateContrLum,
                         {gsft.TextureModulateContrLum.contrast: contrast,
                          gsft.TextureModulateContrLum.luminance: default_lum})
            self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(gsft.TextureModulateContrLum,
                     {gsft.TextureModulateContrLum.contrast: contrast,
                      gsft.TextureModulateContrLum.luminance: default_lum})
        self.add_phase(p)

        for i in range(2):
            for sign in [1, -1]:
                p = Phase(12)
                p.set_visual(gsft.TextureRotation,
                             {gsft.TextureRotation.contrast: contrast,
                              gsft.TextureRotation.luminance: default_lum,
                              gsft.TextureRotation.angular_velocity: sign * 30})
                self.add_phase(p)

                p = Phase(duration=6)
                p.set_visual(gsft.TextureModulateContrLum,
                             {gsft.TextureModulateContrLum.contrast: contrast,
                              gsft.TextureModulateContrLum.luminance: default_lum})
                self.add_phase(p)

        # Baseline phase
        p = Phase(15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0., 0., 0.)})
        self.add_phase(p)
