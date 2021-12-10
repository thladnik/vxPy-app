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
from __future__ import annotations

import numpy as np
from vispy import gloo

from vxpy.core import logging, visual
from vxpy.utils import sphere

log = logging.getLogger(__name__)


class BinaryNoiseVisualFieldMapping(visual.SphericalVisual):

    p_interval = 'p_interval'
    p_bias = 'p_bias'
    p_inverted = 'p_inverted'

    VERT_LOC = './sphere.vert'
    FRAG_LOC = './binary_sphere.frag'

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        self.position_buffer = gloo.VertexBuffer(self.ico_sphere.get_vertices())

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.binary_noise = gloo.Program(VERT, FRAG)
        self.binary_noise['a_position'] = self.position_buffer

    def initialize(self, **params):
        # Set seed!
        np.random.seed(1)

        self.parameters[self.p_bias] = .2
        self.parameters[self.p_inverted] = False
        self.parameters[self.p_interval] = 1.

        # Set initial vertex states
        self.states = np.ascontiguousarray(np.random.rand(self.position_buffer.size) < (1. - self.parameters[self.p_bias]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)
        self.binary_noise['a_state'] = self.state_buffer


        self.binary_noise['u_time'] = 0.0
        self.last = 0

    def render(self, dt):
        self.binary_noise['u_time'] += dt

        bias = self.parameters.get(self.p_bias)
        inverted = self.parameters.get(self.p_inverted)
        interval = self.parameters.get(self.p_interval)

        if bias is None or inverted is None or interval is None:
            return

        now = int(self.binary_noise['u_time'][0] / interval)
        # print(self.binary_noise['u_time'], now)
        if now > self.last:
            states = np.random.rand(self.position_buffer.size) < (1. - bias)
            if inverted:
                states = np.abs(states - 1)

            self.state_buffer[:] = np.ascontiguousarray(states, dtype=np.float32)
            self.last = now

        # Draw
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', self.index_buffer)
