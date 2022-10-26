"""
vxpy_app ./protocols/spherical_gratings.py
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

from vxpy.core.protocol import StaticPhasicProtocol, Phase
from vxpy.visuals import pause

from visuals.spherical_grating import SphericalBlackWhiteGrating


class MiniTestProtocol(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        mylist = [2, 0.5, 0.2]
        for i in range(3):
            sp = 10 * 2 ** i
            p = Phase(duration=4)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: sp,
                          SphericalBlackWhiteGrating.angular_velocity: 0})
            self.add_phase(p)


class StaticGratings(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for i in range(5):
            sp = 10 * 2 ** i
            p = Phase(4)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: sp,
                          SphericalBlackWhiteGrating.angular_velocity: 0})
            self.add_phase(p)


class MovingGratings(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        mov_duration = 3

        for i in list(range(3))[::-1]:
            sp = 30 * 2 ** (i - 1)
            for j in range(5):
                v = (j + 1) * sp / mov_duration

                p = Phase(2)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

                p = Phase(mov_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: v})
                self.add_phase(p)

                p = Phase(mov_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: -v})
                self.add_phase(p)

                self.keep_last_frame_for(2)

        p = Phase(2, visual=pause.ClearBlack)
        self.add_phase(p)

