"""
vxpy_app ./protocols/triggered_protocol_test.py
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
import vxpy.core.event as vxevent
import vxpy.core.protocol as vxprotocol

from visuals.spherical_grating import SphericalBlackWhiteGrating
from vxpy.extras.zf_eyeposition_tracking import ZFEyeTracking


class TriggeredGratingProtocol(vxprotocol.TriggeredProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        # Set trigger that controls progression of this protocol
        trigger = vxevent.RisingEdgeTrigger(ZFEyeTracking.sacc_trigger_name)
        self.set_phase_trigger(trigger)

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
