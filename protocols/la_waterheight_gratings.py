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
import math

import numpy as np

from vxpy.core.protocol import Phase, StaticProtocol
from vxpy.visuals import pause

from visuals.planar_grating import BlackAndWhiteGrating


def calculate_linear(angle, height):
    return height * 2 * math.tan(math.pi / 180 * 0.5 * angle)


class GratingsWaterForWaterHeight(StaticProtocol):
    water_height = None

    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        corralling_period = calculate_linear(1./0.02, self.water_height)
        corralling_velocity = calculate_linear(75, self.water_height)

        # Create sequence
        angular_periods = [1 / 0.01, 1 / 0.02, 1 / 0.04]
        linear_periods = []
        for sp_ang in angular_periods:
            sp_lin = calculate_linear(sp_ang, self.water_height)
            linear_periods.append(sp_lin)

        angular_velocities = [25, 50, 100]
        linear_velocities = [28, 143, 286]
        for vel_ang in angular_velocities:
            vel_lin = calculate_linear(vel_ang, self.water_height)
            linear_velocities.append(vel_lin)

        parameters = []
        for sp_lin in linear_periods:
            for vel_lin in linear_velocities:
                parameters.append((sp_lin, vel_lin))
                parameters.append((sp_lin, -vel_lin))

        np.random.seed(1)
        parameters = np.random.permutation(parameters)

        # Build protocol from parameters
        p = Phase(duration=5)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)

        for sp_lin, vel_lin in parameters:
            p = Phase(duration=30)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: corralling_period,
                          BlackAndWhiteGrating.linear_velocity: -np.sign(vel_lin) * corralling_velocity})
            self.add_phase(p)

            p = Phase(duration=5)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp_lin,
                          BlackAndWhiteGrating.linear_velocity: 0})
            self.add_phase(p)

            p = Phase(duration=30)
            p.set_visual(BlackAndWhiteGrating,
                         {BlackAndWhiteGrating.direction: 'horizontal',
                          BlackAndWhiteGrating.waveform: 'rectangular',
                          BlackAndWhiteGrating.spatial_period: sp_lin,
                          BlackAndWhiteGrating.linear_velocity: vel_lin})
            self.add_phase(p)

        p = Phase(duration=5)
        p.set_visual(pause.ClearBlack)
        self.add_phase(p)


class GratingsWaterheight30mm(GratingsWaterForWaterHeight):
    water_height = 30

    def __init__(self, *args, **kwargs):
        GratingsWaterForWaterHeight.__init__(self, *args, **kwargs)


class GratingsWaterheight60mm(GratingsWaterForWaterHeight):
    water_height = 60

    def __init__(self, *args, **kwargs):
        GratingsWaterForWaterHeight.__init__(self, *args, **kwargs)


class GratingsWaterheight120mm(GratingsWaterForWaterHeight):
    water_height = 120

    def __init__(self, *args, **kwargs):
        GratingsWaterForWaterHeight.__init__(self, *args, **kwargs)
