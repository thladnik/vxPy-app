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
from vxpy.core.protocol import StaticPhasicProtocol
from visuals.sph_shader import Dot,Blank


class sphDotProtocol(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)
        angSize = np.array([np.random.permutation([2,5,10,20,40]) for k in range(3)]).flatten()
        self.add_visual_phase(Blank, 20, {"u_color":(0, 0, 0)})
        for i in range(len(angSize)):
            self.add_visual_phase(Blank, 10, {"u_color":(0, 0, 0)})
            self.add_visual_phase(Dot, 10,
                                  {Dot.u_elv: 0,
                           Dot.u_period: 10,
                           Dot.u_ang_size: angSize[i]})
        self.add_visual_phase(Blank, 20, {"u_color":(0, 0, 0)})
