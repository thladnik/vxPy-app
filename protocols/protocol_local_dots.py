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
from vxpy.core.protocol import StaticPhasicProtocol

from visuals.local_dot import IcoDot


class TestLocalDotProtocol(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(3):
            self.add_phase(IcoDot, None,
                           {IcoDot.p_interval: 2000})
