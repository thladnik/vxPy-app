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

import vxpy.core.protocol as vxprotocol
from visuals.gs_characterization_stims.sft_grating import SphericalSFTGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

def paramsSFGrat(waveform, motion_type, motion_axis, ang_vel, ang_period, offset):
    return {
        SphericalSFTGrating.waveform: waveform,
        SphericalSFTGrating.motion_type: motion_type,
        SphericalSFTGrating.motion_axis: motion_axis,
        SphericalSFTGrating.angular_velocity: ang_vel,
        SphericalSFTGrating.angular_period: ang_period,
        SphericalSFTGrating.offset: offset
    }


class SFTRotatingGratings(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # set fixed parameters PART 1: SF Tuning Motion
        waveform = 'rectangular'
        motion_type = 'rotation'
        motion_axis = 'vertical'
        moving_phase_dur = 6  # sec
        static_phase_dur = 6  # sec
        num_repeat = 3  # number of repeats

        ang_periods = [90, 45, 22.5, 11.25, 5.625] # deg/cycle

        # Add pre-phase
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.0, .0, .0])})
        self.add_phase(p)

        for i in range(num_repeat):
            for sf in ang_periods:
                for direction in [1, -1]:
                    # Static phase
                    p = vxprotocol.Phase(static_phase_dur)
                    p.set_visual(SphericalSFTGrating,
                                 paramsSFGrat(waveform, motion_type, motion_axis, 0, sf, 0))
                    self.add_phase(p)

                    # Moving phase
                    p = vxprotocol.Phase(moving_phase_dur)
                    p.set_visual(SphericalSFTGrating,
                                 paramsSFGrat(waveform, motion_type, motion_axis, 30 * direction, sf, 0))
                    self.add_phase(p)

        # Add post-phase
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.0, .0, .0])})
        self.add_phase(p)

