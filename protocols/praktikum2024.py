import numpy as np
from vxpy.core.protocol import StaticProtocol
import vxpy.core.protocol as vxprotocol
from visuals.spherical_grating import SphericalBlackWhiteGrating
from controls.led_pwm_control import LedPWMControl
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class OKRStimulation(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        vel = 15
        sp = 1 / np.array([0.02, 0.06, 0.1])

        # detect spontaneous saccades
        p = vxprotocol.Phase(duration=300)
        p.set_visual(SphericalBlackWhiteGrating,
                     {SphericalBlackWhiteGrating.waveform: 'rectangular',
                      SphericalBlackWhiteGrating.motion_axis: 'vertical',
                      SphericalBlackWhiteGrating.motion_type: 'rotation',
                      SphericalBlackWhiteGrating.angular_period: 1/0.06,
                      SphericalBlackWhiteGrating.angular_velocity: 0})
        self.add_phase(p)

        # take all spatial frequencies
        for _sp in sp:
            # two directions
            for i in range(2):

                # pause phase
                p = vxprotocol.Phase(duration=30)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: _sp,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

                # stimulation phase
                p = vxprotocol.Phase(duration=30)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: _sp,
                              SphericalBlackWhiteGrating.angular_velocity: vel})
                self.add_phase(p)
                vel *= -1


class OptogeneticStimulationTail(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

    def create(self):

        for i in range(10):
            p = vxprotocol.Phase(0.5)
            p.set_control(LedPWMControl, {'light_intensity': 0})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

            p = vxprotocol.Phase(1)
            p.set_control(LedPWMControl, {'light_intensity': 1. / (i + 1)})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

        p = vxprotocol.Phase(8)
        p.set_control(LedPWMControl, {'light_intensity': 0})
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
        self.add_phase(p)


class OptogeneticStimulationTailB(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

    def create(self):

        for i in [1, 0.8, 0.6, 0.4, 0.2, 0]:
            p = vxprotocol.Phase(8)
            p.set_control(LedPWMControl, {'light_intensity': 0})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

            p = vxprotocol.Phase(8)
            p.set_control(LedPWMControl, {'light_intensity': i})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

        p = vxprotocol.Phase(8)
        p.set_control(LedPWMControl, {'light_intensity': 0})
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
        self.add_phase(p)


class OptogeneticStimulationTailShortPulse(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

    def create(self):

        for i in range(5):
            p = vxprotocol.Phase(6)
            p.set_control(LedPWMControl, {'light_intensity': 0})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

            p = vxprotocol.Phase(1.5)
            p.set_control(LedPWMControl, {'light_intensity': 1. / (i + 1)})
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
            self.add_phase(p)

        p = vxprotocol.Phase(8)
        p.set_control(LedPWMControl, {'light_intensity': 0})
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: (0., 0., 0.)})
        self.add_phase(p)


class OptogeneticStimulationOKR(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

    def create(self):
        for i in range(5):
            p = vxprotocol.Phase(8)
            p.set_control(LedPWMControl, {'light_intensity': 0})
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: 25.,
                          SphericalBlackWhiteGrating.angular_velocity: 45})
            self.add_phase(p)

            p = vxprotocol.Phase(8)
            p.set_control(LedPWMControl, {'light_intensity': 1. / (i+1)})
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: 25.,
                          SphericalBlackWhiteGrating.angular_velocity: 45})
            self.add_phase(p)


