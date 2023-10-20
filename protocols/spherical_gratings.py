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
import vxpy.core.protocol as vxprotocol

from controls.control_tests import TestControl01
from visuals.spherical_grating import SphericalBlackWhiteGrating


class GratingProtocol(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        # Add spherical gratings with varying spatial period
        for i in range(3):
            sp = 15 * 2 ** i
            p = vxprotocol.Phase(duration=10)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: sp,
                          SphericalBlackWhiteGrating.angular_velocity: 0})
            self.add_phase(p)


class RepeatsTestProtocol(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        # Go through all 3 repeats
        for r in range(3):
            # Start repeat
            self.start_repeat()

            # Add phases in repeat
            for i in range(3):
                sp = 10 * 2 ** i
                p = vxprotocol.Phase(duration=4)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

            # End repeat
            self.end_repeat()


class StaticGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        for i in range(5):
            sp = 10 * 2 ** i
            p = vxprotocol.Phase(4)
            p.set_visual(SphericalBlackWhiteGrating,
                         {SphericalBlackWhiteGrating.waveform: 'rectangular',
                          SphericalBlackWhiteGrating.motion_axis: 'vertical',
                          SphericalBlackWhiteGrating.motion_type: 'rotation',
                          SphericalBlackWhiteGrating.angular_period: sp,
                          SphericalBlackWhiteGrating.angular_velocity: 0})
            self.add_phase(p)


class MovingGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        mov_duration = 3

        for i in list(range(3))[::-1]:
            sp = 30 * 2 ** (i - 1)
            for j in range(5):
                v = (j + 1) * sp / mov_duration

                p = vxprotocol.Phase(2)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: 0})
                self.add_phase(p)

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: v})
                p.set_control(TestControl01)
                self.add_phase(p)

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(SphericalBlackWhiteGrating,
                             {SphericalBlackWhiteGrating.waveform: 'rectangular',
                              SphericalBlackWhiteGrating.motion_axis: 'vertical',
                              SphericalBlackWhiteGrating.motion_type: 'rotation',
                              SphericalBlackWhiteGrating.angular_period: sp,
                              SphericalBlackWhiteGrating.angular_velocity: -v})
                self.add_phase(p)

                self.keep_last_frame_for(2)

