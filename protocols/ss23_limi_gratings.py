"""
vxpy ./protocols/spherical_gratings.py - Example protocol for demonstration.
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
from visuals.spherical_grating import SphericalBlackWhiteGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class RotatingGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        moving_phase_dur = 6  # seconds
        pause_phase_dur = 6  # seconds
        num_repeat = 3  # number of repeats

        spatial_periods = [90, 45, 22.5, 11.25, 5.625]   # deg/cycle

        # Add pre-phase
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.0, .0, .0])})
        self.add_phase(p)

        for i in range(num_repeat):
            for sf in spatial_periods:
                for direction in [1, -1]:
                    # Static phase
                    p = vxprotocol.Phase(pause_phase_dur)
                    p.set_visual(SphericalBlackWhiteGrating,
                                 {SphericalBlackWhiteGrating.waveform: 'rectangular',
                                  SphericalBlackWhiteGrating.motion_type: 'rotation',
                                  SphericalBlackWhiteGrating.motion_axis: 'vertical',
                                  SphericalBlackWhiteGrating.angular_velocity: 0,
                                  SphericalBlackWhiteGrating.angular_period: sf}
                                 )
                    self.add_phase(p)

                    # Moving phase
                    p = vxprotocol.Phase(moving_phase_dur)
                    p.set_visual(SphericalBlackWhiteGrating,
                                 {SphericalBlackWhiteGrating.waveform: 'rectangular',
                                  SphericalBlackWhiteGrating.motion_type: 'rotation',
                                  SphericalBlackWhiteGrating.motion_axis: 'vertical',
                                  SphericalBlackWhiteGrating.angular_velocity: 30 * direction,
                                  SphericalBlackWhiteGrating.angular_period: sf}
                                 )
                    self.add_phase(p)

        # Add post-phase
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.0, .0, .0])})
        self.add_phase(p)

