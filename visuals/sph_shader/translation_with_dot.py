"""
vxpy ./visuals/spherical/grating.py
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
from vispy import gloo

from vxpy.core import visual
from vxpy.utils import sphere


class TranslationWithDot(visual.SphericalVisual):
    u_ang_velocity = 'u_ang_velocity'
    u_spat_period = 'u_spat_period'
    u_ang_size = 'u_ang_size'
    u_period = 'u_period'
    u_elv = 'u_elv'

    interface = [
        # Translation grating
        (u_ang_velocity, 15., -100., 100., {'step_size': 1.}),
        (u_spat_period, 10., 2., 360., {'step_size': 1.}),
        # Dot
        (u_ang_size, 20., 0., 100., {'step_size': 1.}),
        (u_period, 10., -40., 40., {'step_size': 1.}),
        (u_elv, 0., -45., 45., {'step_size': 1.})
    ]

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)

        # Translation grating with dot
        vert = self.load_vertex_shader('./sphericalShader.vert')
        frag = self.load_shader('./trans_grating_with_dot.frag')
        self.trans_with_dot = gloo.Program(vert, frag)
        self.trans_with_dot['a_position'] = self.position_buffer

    def initialize(self, **params):
        self.trans_with_dot['u_stime'] = 0.0
        self.update(**params)

    def render(self, dt):
        self.trans_with_dot['u_stime'] += dt

        self.apply_transform(self.trans_with_dot)
        self.trans_with_dot.draw('triangles', self.index_buffer)
