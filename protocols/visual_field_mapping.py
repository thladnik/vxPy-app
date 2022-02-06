"""
vxpy_app ./protocols/visual_field_mapping.py
Copyright (C) 2022 Tim Hladnik

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
from vxpy.core.protocol import StaticPhasicProtocol, Phase
from vxpy.visuals import pause

from visuals.sphere_visual_field_mapping import \
    BinaryNoiseVisualFieldMapping8deg, \
    BinaryNoiseVisualFieldMapping16deg, \
    BinaryNoiseVisualFieldMapping32deg


class BinaryNoise(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for inv in [False, True]:
            for bias in [0.1, 0.2]:
                p = Phase(10, visual=pause.ClearBlack)
                self.add_phase(p)

                p = Phase(30)
                p.set_visual(BinaryNoiseVisualFieldMapping32deg,
                             **{BinaryNoiseVisualFieldMapping32deg.p_bias: bias,
                                BinaryNoiseVisualFieldMapping32deg.p_interval: 1000,
                                BinaryNoiseVisualFieldMapping32deg.p_inverted: inv})
                self.add_phase(p)

                p = Phase(30)
                p.set_visual(BinaryNoiseVisualFieldMapping16deg,
                             **{BinaryNoiseVisualFieldMapping16deg.p_bias: bias,
                                BinaryNoiseVisualFieldMapping16deg.p_interval: 1000,
                                BinaryNoiseVisualFieldMapping16deg.p_inverted: inv})
                self.add_phase(p)

                p = Phase(30)
                p.set_visual(BinaryNoiseVisualFieldMapping8deg,
                             **{BinaryNoiseVisualFieldMapping8deg.p_bias: bias,
                                BinaryNoiseVisualFieldMapping8deg.p_interval: 1000,
                                BinaryNoiseVisualFieldMapping8deg.p_inverted: inv})
                self.add_phase(p)