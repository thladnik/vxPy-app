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
from vispy import app, gloo

from vxpy.core import logging, visual
from vxpy.utils import sphere

log = logging.getLogger(__name__)


class BinaryNoiseVisualFieldMapping(visual.SphericalVisual):

    p_interval = 'p_interval'
    p_bias = 'p_bias'
    p_inverted = 'p_inverted'

    VERT_LOC = './sphere.vert'
    FRAG_LOC = './binary_sphere.frag'

    interface = [
        (p_interval, 1000, 100, 10000, {'step_size': 100}),
        (p_bias, .10, .01, .50, {'step_size': .01}),
        (p_inverted, False),
    ]

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)

        # For now, just set this to parameters
        self._add_data_appendix('vertex_coords', vertices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.binary_noise = gloo.Program(VERT, FRAG)
        self.binary_noise['a_position'] = self.position_buffer

        # self.all_states = np.nan * np.ones((1000, vertices.shape[0]))
        # self.tmr = app.Timer()
        # self.tmr.connect(self.print_states_params)
        # self.tmr.start(1)

    def initialize(self, **params):
        # Set seed!
        np.random.seed(1)

        # Set initial vertex states
        self.states = np.ascontiguousarray(np.random.rand(self.position_buffer.size) < (1. - self.parameters[self.p_bias]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)
        self.binary_noise['a_state'] = self.state_buffer

        self.binary_noise['u_time'] = 0.0
        self.last = 0.0
        self.states_idx = 0
        self._set_new_states(self.parameters[self.p_bias], self.parameters[self.p_inverted])

    def _set_new_states(self, bias, inverted):

        states = np.random.rand(self.position_buffer.size) < (1. - bias)
        if inverted:
            states = np.logical_not(states)

        self.parameters.update(vertex_states=states)
        self.all_states[self.states_idx] = states
        self.states_idx += 1
        self.state_buffer[:] = np.ascontiguousarray(states, dtype=np.float32)

    def print_states_params(self, dt):
        pass
        # print('Timer?')
        # mean_states = np.nanmean(self.all_states, axis=0)
        # print(mean_states.shape)
        # print(np.mean(mean_states), '+/-', np.std(mean_states))

    def render(self, dt):
        self.binary_noise['u_time'] += dt

        bias = self.parameters.get(self.p_bias)
        inverted = self.parameters.get(self.p_inverted)
        interval = self.parameters.get(self.p_interval) / 1000

        if bias is None or inverted is None or interval is None:
            return

        now = self.binary_noise['u_time'][0]
        if now > self.last + interval:
            self._set_new_states(bias, inverted)
            self.last = now

        # Draw
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', self.index_buffer)
