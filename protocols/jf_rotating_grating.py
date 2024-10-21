"""
vxpy_app ./protocols/spherical_gratings.py
Copyright (C) 2024 Julian Fix

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
import itertools
import vxpy.core.protocol as vxprotocol
from visuals.spherical_grating import SphericalBlackWhiteGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class RotatingGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        moving_phase_dur = 30  # seconds
        pause_phase_dur = 15  # seconds
        num_repeat = 2  # number of repeats

        spatial_periods = 1. / np.array([0.01, 0.05, 0.1, 0.2])  # deg/cycle
        angular_velocities = np.array([3, 6.25, 12.5, 25])

        for i in range(num_repeat):
            all_combinations = list(itertools.product(spatial_periods, angular_velocities, [1, -1]))
            np.random.shuffle(all_combinations)

            for sp, av, direction in all_combinations:
                # Static phase
                p = vxprotocol.Phase(pause_phase_dur)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.angular_velocity: 0,
                              SphericalBlackWhiteGrating.angular_period: sp}
                             )
                self.add_phase(p)

                # Moving phase
                p = vxprotocol.Phase(moving_phase_dur)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.angular_velocity: av * direction,
                              SphericalBlackWhiteGrating.angular_period: sp}
                             )
                self.add_phase(p)