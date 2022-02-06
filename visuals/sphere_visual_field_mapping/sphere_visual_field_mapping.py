"""
vxpy_app ./visuals/sphere_visual_field_mapping/sphere_visual_field_mapping.py
Copyright (C) 2022 Tim Hladnik

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

    interface = [
        (p_interval, 1000, 100, 10000, {'step_size': 100}),
        (p_bias, .10, .01, .50, {'step_size': .01}),
        (p_inverted, False),
    ]

    def __init__(self, subdiv_lvl, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=subdiv_lvl)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        self.vertices = self.ico_sphere.get_vertices()
        vertex_lvls = self.ico_sphere.get_vertex_levels()

        # For now, just set this to parameters
        self._add_data_appendix('vertex_coords', self.vertices)

        # Set up program for noise pattern
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.binary_noise = gloo.Program(VERT, FRAG, count=self.vertices.shape[0])
        self.binary_noise['a_position'] = self.vertices
        self.binary_noise['a_vertex_lvl'] = np.ascontiguousarray(vertex_lvls, dtype=np.float32)

        # Program for mesh
        self.mesh = gloo.Program(VERT, 'void main() { gl_FragColor = vec4(1., 0., 0., 1.); }',
                                 count=self.vertices.shape[0])
        self.mesh['a_position'] = self.vertices

    def initialize(self, **params):
        # Set seed!
        np.random.seed(1)

        # Set initial vertex states
        self.states = np.ascontiguousarray(np.zeros(self.vertices.shape[0]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)
        self.binary_noise['a_state'] = self.state_buffer

        self.binary_noise['u_time'] = 0.0
        self.last = 0.0
        self.states_idx = 0
        self._set_new_states(self.parameters[self.p_bias], self.parameters[self.p_inverted])

    def _set_new_states(self, bias, inverted):

        states = np.random.rand(self.vertices.shape[0]) > (1. - bias)
        if inverted:
            states = np.logical_not(states)

        self.parameters.update(vertex_states=states)
        # self.all_states[self.states_idx] = states
        self.states_idx += 1
        self.binary_noise['a_state'][:] = np.ascontiguousarray(states, dtype=np.float32)

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
        from vispy.gloo import gl
        self.apply_transform(self.mesh)
        gl.glLineWidth(2)
        self.mesh.draw('line_loop', indices=self.index_buffer)
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', indices=self.index_buffer)



class BinaryNoiseVisualFieldMapping2deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 5, *args, **kwargs)


class BinaryNoiseVisualFieldMapping4deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 4, *args, **kwargs)


class BinaryNoiseVisualFieldMapping8deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 3, *args, **kwargs)


class BinaryNoiseVisualFieldMapping16deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 2, *args, **kwargs)


class BinaryNoiseVisualFieldMapping32deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 1, *args, **kwargs)


class BinaryNoiseVisualFieldMapping64deg(BinaryNoiseVisualFieldMapping):

    def __init__(self, *args, **kwargs):
        BinaryNoiseVisualFieldMapping.__init__(self, 0, *args, **kwargs)
