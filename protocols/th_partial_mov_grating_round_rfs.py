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
from vxpy.utils.sphere import IcosahedronSphere
from vxpy.utils.geometry import cart2sph1


from visuals.partial_spherical_grating_round_rfs import PartialSphericalGratingWithRoundRFs
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


def create_params(period, velocity, center_az, center_el, field_diameter):
    return {
        PartialSphericalGratingWithRoundRFs.angular_period: period,
        PartialSphericalGratingWithRoundRFs.angular_velocity: velocity,
        PartialSphericalGratingWithRoundRFs.rf_center_azimuth: center_az,
        PartialSphericalGratingWithRoundRFs.rf_center_elevation: center_el,
        PartialSphericalGratingWithRoundRFs.rf_diameter: field_diameter
    }


class Protocol01(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 30
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        selected = np.logical_and(az > 0.0, el < 30)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            combos.append((15, *coords))
            combos.append((40, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for size, az, el in shuffled_combos:
            p = vxprotocol.Phase(duration=4)
            p.set_visual(PartialSphericalGratingWithRoundRFs,
                         create_params(period, 0, az, el, size))
            self.add_phase(p)

            p = vxprotocol.Phase(duration=4)
            p.set_visual(PartialSphericalGratingWithRoundRFs,
                         create_params(period, velocity, az, el, size))
            self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)
