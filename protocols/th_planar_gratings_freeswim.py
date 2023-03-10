"""
vxpy_app ./protocols/planar_gratings.py
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

from vxpy.core.protocol import Phase, StaticProtocol
from vxpy.visuals import pause

from visuals.planar_grating import BlackAndWhiteGrating


class AlternatingGratings(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        spatial_periods = 10 * [10] + 10 * [15]
        velocities = 5 * [10] + 5 * [-10] + 5 * [15] + 5 * [-15]
        combos = list(zip(spatial_periods, velocities))
        combos = np.random.permutation(combos)

        for sp, v in combos:
            p = Phase(duration=15)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: v})
            self.add_phase(p)

