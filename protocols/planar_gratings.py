"""
vxpy_app ./protocols/planar_gratings.py - Example protocol for demonstration.
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
from vxpy.core.protocol import Phase, StaticPhasicProtocol
from vxpy.visuals import pause

from visuals.planar_grating import BlackAndWhiteGrating


class GratingsKeepUntilEnd(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = i + 1
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 5.})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=2)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)


class GratingsBlankPauses(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = i + 1
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 5.})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)

            # Blank between phases
            p = Phase(duration=2)
            p.set_visual(pause.ClearBlack)
            self.add_phase(p)


class GratingsKeepForever(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = i + 1
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         **{BlackAndWhiteGrating.p_direction: 'horizontal',
                            BlackAndWhiteGrating.p_shape: 'rectangular',
                            BlackAndWhiteGrating.u_spat_period: 2 * sp,
                            BlackAndWhiteGrating.u_lin_velocity: 5.})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)
