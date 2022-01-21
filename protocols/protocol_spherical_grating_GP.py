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
from visuals.sph_shader import Translation, Rotation, Blank


class sphDotProtocol(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        # Translations
        np.random.seed(1)
        stimuli = []
        for v in [-10,-20,-40,-80,-160,10,20,40,80,160]:
            stimuli.append((Translation, v))
            stimuli.append((Rotation, v))

        all_stimuli = []
        for i in range(3):
            all_stimuli.extend(np.random.permutation(stimuli))

        self.add_visual_phase(Blank, 20, {"u_color":(0.5, 0.5, 0.5)})

        for stim, v in all_stimuli:

            self.add_visual_phase(stim, 5,
                                  {stim.u_ang_velocity: 0,
                            stim.u_spat_period: 15})
            self.add_visual_phase(stim, 5,
                                  {stim.u_ang_velocity: v,
                            stim.u_spat_period: 15})

        self.add_visual_phase(Blank, 20, {"u_color":(0.5, 0.5, 0.5)})
