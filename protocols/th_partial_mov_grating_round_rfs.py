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


def create_partially_mov_grating_params(period, velocity):
    return {
        PartialSphericalBlackWhiteGrating.angular_period: period,
        PartialSphericalBlackWhiteGrating.angular_velocity: velocity
    }

class BaseProtocol(vxprotocol.StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticPhasicProtocol.__init__(self, *args, **kwargs)

