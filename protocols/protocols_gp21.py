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
from visuals.sph_shader import Translation, Dot, TranslationWithDot, Blank


dot_sizes = [2, 5, 10, 20, 40]
trans_speeds = [-80, 80]


class GP21TranslationsWithDot(StaticPhasicProtocol):
    # 2 * 5 * 3 * 15 + 20 + 20 = 490
    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        # Translations
        np.random.seed(1)
        stimuli = []
        for v in trans_speeds:
            for s in dot_sizes:
                stimuli.append((TranslationWithDot, v, s))

        all_stimuli = []
        for i in range(3):
            all_stimuli.extend(np.random.permutation(stimuli))

        self.add_phase(Blank, 20, {"u_color": (0.5, 0.5, 0.5)})

        for stim, v, s in all_stimuli:
            self.add_phase(stim, 5,
                           {stim.u_ang_velocity: 0,
                            stim.u_spat_period: 15,
                            Dot.u_elv: 0,
                            Dot.u_period: 10**8,
                            Dot.u_ang_size: s})

            self.add_phase(stim, 10,
                           {stim.u_ang_velocity: v,
                            stim.u_spat_period: 15,
                            Dot.u_elv: 0,
                            Dot.u_period: 10,
                            Dot.u_ang_size: s})

        self.add_phase(Blank, 20, {"u_color": (0.5, 0.5, 0.5)})

        # Translations
        np.random.seed(1)
        stimuli = []
        for v in trans_speeds:
            stimuli.append((Translation, v))

        all_stimuli = []
        for i in range(3):
            all_stimuli.extend(np.random.permutation(stimuli))

        self.add_phase(Blank, 20, {"u_color": (0.5, 0.5, 0.5)})

        for stim, v in all_stimuli:
            self.add_phase(stim, 5,
                           {stim.u_ang_velocity: 0,
                            stim.u_spat_period: 15})
            self.add_phase(stim, 10,
                           {stim.u_ang_velocity: v,
                            stim.u_spat_period: 15})

        self.add_phase(Blank, 20, {"u_color": (0.5, 0.5, 0.5)})

        all_sizes = []
        for i in range(3):
            all_sizes.extend(np.random.permutation(dot_sizes))

        self.add_phase(Blank, 20, {"u_color": (0, 0, 0)})

        for s in all_sizes:
            self.add_phase(Blank, 10, {"u_color": (0, 0, 0)})

            self.add_phase(Dot, 10,
                           {Dot.u_elv: 0,
                            Dot.u_period: 10,
                            Dot.u_ang_size: s})

        self.add_phase(Blank, 20, {"u_color": (0, 0, 0)})
