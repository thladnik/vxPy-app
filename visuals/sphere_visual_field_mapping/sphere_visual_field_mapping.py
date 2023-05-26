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
from vispy.util import transforms

from vxpy.core import logger, visual
from vxpy.utils import sphere

log = logger.getLogger(__name__)


class BinaryState(visual.Attribute):

    def __init__(self, *args, **kwargs):
        visual.Attribute.__init__(self, 'binary_state', *args, **kwargs)

    def upstream_updated(self):
        print(BinaryNoiseVisualFieldMapping.pattern_bias.data)
        self.update_binary_state()

    def update_binary_state(self):
        bias = BinaryNoiseVisualFieldMapping.pattern_bias.data
        self.data = np.ascontiguousarray(np.random.rand(BinaryNoiseVisualFieldMapping.xyz_coordinate.data.shape[0]) < bias, dtype=np.float32)


class Rotation(visual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        visual.Mat4Parameter.__init__(self, 'rotation', *args, **kwargs)
        self.data = np.eye(4)
        self.incr_in_degree_per_sec = 5.  # deg/s
        self.lt = 0.0

    def upstream_updated(self):
        self.t = BinaryNoiseVisualFieldMapping.time.data[0]
        v = self.incr_in_degree_per_sec * (self.t - self.lt)
        r1 = transforms.rotate(v * 2 * (np.random.rand() - 0.5), (0, 0, 1))
        r2 = transforms.rotate(v * 2 * (np.random.rand() - 0.5), (0, 1, 0))
        self.data = np.dot(self.data, np.dot(r1, r2))
        self.lt = self.t


class BinaryNoiseVisualFieldMapping(visual.SphericalVisual):

    time = visual.FloatParameter('time', internal=True)
    rotation = Rotation(internal=True)
    switching_interval = visual.IntParameter('switching_interval', static=True, limits=(100, 5000), default=1000, step_size=100)
    pattern_bias = visual.FloatParameter('pattern_bias', static=True, limits=(0.01, 0.99), default=0.5, step_size=0.01)
    xyz_coordinate = visual.Attribute('xyz_coordinate', static=True)
    binary_state = BinaryState()

    VERT_LOC = './sphere.vert'
    FRAG_LOC = './binary_sphere.frag'

    def __init__(self, subdiv_lvl, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=subdiv_lvl)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        self.vertices = self.ico_sphere.get_vertices()

        # Set up program for noise pattern
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.binary_noise = gloo.Program(VERT, FRAG, count=self.vertices.shape[0])

        # Initialze visual attributes
        self.xyz_coordinate.data = self.vertices
        self.binary_state.data = np.ascontiguousarray(np.random.rand(self.vertices.shape[0]) < 0.5, dtype=np.float32)
        self.pattern_bias.add_downstream_link(self.binary_state)

        # Update rotation with each new state
        self.binary_state.add_downstream_link(self.rotation)

        self.xyz_coordinate.connect(self.binary_noise)
        self.binary_state.connect(self.binary_noise)
        self.rotation.connect(self.binary_noise)

        # Program for mesh (debugging)
        # self.mesh = gloo.Program(VERT, 'void main() { gl_FragColor = vec4(1., 0., 0., 1.); }',
        #                          count=self.vertices.shape[0])
        # self.mesh['a_position'] = self.vertices
        self.last = None


    def initialize(self, **params):
        # Set seed!
        np.random.seed(1)

        self.time.data = 0.0
        self.last = 0.0
        self.binary_noise['binary_state'] = gloo.VertexBuffer(self.binary_state.data)

    def _set_new_states(self, bias, inverted):
        pass

    def render(self, dt):
        self.time.data += dt

        now = self.time.data[0]
        interval = self.switching_interval.data
        if now > self.last + interval / 1000:
            self.binary_state.update_binary_state()
            self.last = now

        # Draw
        # from vispy.gloo import gl
        # self.apply_transform(self.mesh)
        # gl.glLineWidth(2)
        # self.mesh.draw('line_loop', indices=self.index_buffer)
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
