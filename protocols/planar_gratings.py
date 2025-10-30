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
from vxpy.core.protocol import Phase, StaticProtocol
from vxpy.visuals import pause

from visuals.planar_grating import BlackAndWhiteGrating


class GratingsKeepUntilEnd(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = (i + 1) * 5
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 2 * sp})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)

        # Blank at end of protocol
        p = Phase(duration=2)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)


class GratingsBlankPauses(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = (i + 1) * 10
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 2 * sp})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)

            # Blank between phases
            p = Phase(duration=2)
            p.set_visual(pause.ClearBlack)
            self.add_phase(p)


class GratingsKeepForever(StaticProtocol):

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        for i in range(4):
            sp = (i + 1) * 10
            p = Phase(duration=2)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 0.})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp,
                          BlackAndWhiteGrating.linear_velocity: 2 * sp})
            self.add_phase(p)

            p = Phase(duration=2)
            p.set_visual(pause.KeepLast)
            self.add_phase(p)
