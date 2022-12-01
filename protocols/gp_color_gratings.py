"""
vxpy_app ./protocols/spherical_gratings.py
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

import vxpy.core.protocol as vxprotocol

from visuals.spherical_grating import SphericalColorContrastGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class Protocol01(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        angular_period_degrees = 30
        angular_velocity_degrees = 30

        c = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        params = []
        for d in [-1, 1]:
            for c1 in c:
                for c2 in c:
                    params.append((c1, c2, d))
        np.random.seed(1)
        params = np.random.permutation(params)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(3):
            for c1, c2, d in params:

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: 0.,
                                  SphericalColorContrastGrating.waveform: 'sinusoidal',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: d * angular_velocity_degrees,
                                  SphericalColorContrastGrating.waveform: 'sinusoidal',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolGP(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        angular_period_degrees = 30
        angular_velocity_degrees = 30

        c = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        params = []
        for d in [-1, 1]:
            for c1 in c:
                for c2 in c:
                    params.append((c1, c2, d))
        np.random.seed(1)
        params = np.random.permutation(params)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(3):
            for c1, c2, d in params:

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: 0.,
                                  SphericalColorContrastGrating.waveform: 'rectangular',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(SphericalColorContrastGrating,
                                 {SphericalColorContrastGrating.angular_period: angular_period_degrees,
                                  SphericalColorContrastGrating.angular_velocity: d * angular_velocity_degrees,
                                  SphericalColorContrastGrating.waveform: 'rectangular',
                                  SphericalColorContrastGrating.motion_type: 'rotation',
                                  SphericalColorContrastGrating.motion_axis: 'vertical',
                                  SphericalColorContrastGrating.red01: c1,
                                  SphericalColorContrastGrating.green01: 0.0,
                                  SphericalColorContrastGrating.blue01: 0.0,
                                  SphericalColorContrastGrating.red02: 0.0,
                                  SphericalColorContrastGrating.green02: c2,
                                  SphericalColorContrastGrating.blue02: 0.0,
                                  })
                self.add_phase(phase)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)
        
