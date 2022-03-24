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

import vxpy.core.protocol as vxprotocol

from visuals.partial_spherical_grating import PartialSphericalBlackWhiteGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


def create_partially_mov_grating_params(period, velocity, az_center, az_range, el_center, el_range):
    return {PartialSphericalBlackWhiteGrating.waveform: 'rectangular',
            PartialSphericalBlackWhiteGrating.motion_axis: 'vertical',
            PartialSphericalBlackWhiteGrating.motion_type: 'rotation',
            PartialSphericalBlackWhiteGrating.angular_period: period,
            PartialSphericalBlackWhiteGrating.angular_velocity: velocity,
            PartialSphericalBlackWhiteGrating.mask_azimuth_center: az_center,
            PartialSphericalBlackWhiteGrating.mask_azimuth_range: az_range,
            PartialSphericalBlackWhiteGrating.mask_elevation_center: el_center,
            PartialSphericalBlackWhiteGrating.mask_elevation_range: el_range}


class BaseProtocol(vxprotocol.StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticPhasicProtocol.__init__(self, *args, **kwargs)

    def _create_protocol(self, angular_period_degrees, angular_velocity_degrees):

        # Blank at start of protocol for baseline
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)

        # Basic parameters
        mov_duration = 4
        pause_duration = 4
        # angular_period_degrees = 30
        # angular_velocity_degrees = 15

        for i in range(3):

            # Full field static
            p = vxprotocol.Phase(mov_duration)
            p.set_visual(PartialSphericalBlackWhiteGrating,
                         create_partially_mov_grating_params(angular_period_degrees,
                                                             angular_velocity_degrees,
                                                             0, 0,
                                                             0, 0))
            self.add_phase(p)
            # Pause
            self.keep_last_frame_for(pause_duration)

            # Full field moving
            p = vxprotocol.Phase(mov_duration)
            p.set_visual(PartialSphericalBlackWhiteGrating,
                         create_partially_mov_grating_params(angular_period_degrees,
                                                             angular_velocity_degrees,
                                                             90, 180,
                                                             0, 90))
            self.add_phase(p)
            # Pause
            self.keep_last_frame_for(pause_duration)

            # LARGE
            large_rows = 2
            large_cols = 2
            large_patch_borders_el = np.linspace(-45, 45, large_rows + 1)
            large_patch_borders_az = np.linspace(0, 180, large_cols + 1)

            # Large rows
            for lel, hel in zip(large_patch_borders_el[:-1], large_patch_borders_el[1:]):
                el_center = (lel + hel) / 2
                el_range = hel - lel

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(PartialSphericalBlackWhiteGrating,
                             create_partially_mov_grating_params(angular_period_degrees,
                                                                 angular_velocity_degrees,
                                                                 90, 180,
                                                                 el_center, el_range))
                self.add_phase(p)
                # Pause
                self.keep_last_frame_for(pause_duration)

            # Large cols
            for laz, haz in zip(large_patch_borders_az[:-1], large_patch_borders_az[1:]):
                az_center = (laz + haz) / 2
                az_range = haz - laz

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(PartialSphericalBlackWhiteGrating,
                             create_partially_mov_grating_params(angular_period_degrees,
                                                                 angular_velocity_degrees,
                                                                 az_center, az_range,
                                                                 0, 90))
                self.add_phase(p)
                # Pause
                self.keep_last_frame_for(pause_duration)

            # Large patches
            for lel, hel in zip(large_patch_borders_el[:-1], large_patch_borders_el[1:]):
                el_center = (lel + hel) / 2
                el_range = hel - lel

                for laz, haz in zip(large_patch_borders_az[:-1], large_patch_borders_az[1:]):
                    az_center = (laz + haz) / 2
                    az_range = haz - laz

                    p = vxprotocol.Phase(mov_duration)
                    p.set_visual(PartialSphericalBlackWhiteGrating,
                                 create_partially_mov_grating_params(angular_period_degrees,
                                                                     angular_velocity_degrees,
                                                                     az_center, az_range,
                                                                     el_center, el_range))
                    self.add_phase(p)
                    # Pause
                    self.keep_last_frame_for(pause_duration)

            # SMALL
            rows = 6
            cols = 6
            small_patch_borders_el = np.linspace(-45, 45, rows + 1)
            small_patch_borders_az = np.linspace(0, 180, cols + 1)

            # Small rows
            for lel, hel in zip(small_patch_borders_el[:-1], small_patch_borders_el[1:]):
                el_center = (lel + hel) / 2
                el_range = hel - lel

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(PartialSphericalBlackWhiteGrating,
                             create_partially_mov_grating_params(angular_period_degrees,
                                                                 angular_velocity_degrees,
                                                                 90, 180,
                                                                 el_center, el_range))
                self.add_phase(p)
                # Pause
                self.keep_last_frame_for(pause_duration)

            # Small cols
            for laz, haz in zip(small_patch_borders_az[:-1], small_patch_borders_az[1:]):
                az_center = (laz + haz) / 2
                az_range = haz - laz

                p = vxprotocol.Phase(mov_duration)
                p.set_visual(PartialSphericalBlackWhiteGrating,
                             create_partially_mov_grating_params(angular_period_degrees,
                                                                 angular_velocity_degrees,
                                                                 az_center, az_range,
                                                                 0, 90))
                self.add_phase(p)
                # Pause
                self.keep_last_frame_for(pause_duration)

            # Small patches
            for lel, hel in zip(small_patch_borders_el[:-1], small_patch_borders_el[1:]):
                el_center = (lel + hel) / 2
                el_range = hel - lel

                for laz, haz in zip(small_patch_borders_az[:-1], small_patch_borders_az[1:]):
                    az_center = (laz + haz) / 2
                    az_range = haz - laz

                    p = vxprotocol.Phase(mov_duration)
                    p.set_visual(PartialSphericalBlackWhiteGrating,
                                 create_partially_mov_grating_params(angular_period_degrees,
                                                                     angular_velocity_degrees,
                                                                     az_center, az_range,
                                                                     el_center, el_range))
                    self.add_phase(p)

                    # Pause
                    self.keep_last_frame_for(pause_duration)

        # Blank at end of protocol for baseline
        p = vxprotocol.Phase(duration=15)
        p.set_visual(SphereUniformBackground,
                     {SphereUniformBackground.u_color: (0.0,) * 3})
        self.add_phase(p)


class ProtocolCW(BaseProtocol):

    def __init__(self):
        BaseProtocol.__init__(self)

        angular_period_degrees = 30
        angular_velocity_degrees = 30

        self._create_protocol(angular_period_degrees, angular_velocity_degrees)


class ProtocolCCW(BaseProtocol):

    def __init__(self):
        BaseProtocol.__init__(self)

        angular_period_degrees = 30
        angular_velocity_degrees = -30

        self._create_protocol(angular_period_degrees, angular_velocity_degrees)
