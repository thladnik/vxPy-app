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
from vispy import scene
from vispy.util import transforms

from vxpy.core import visual
from vxpy.utils import sphere


class IcoBinaryNoiseSphere_SimuSaccade(visual.SphericalVisual):

    p_sacc_duration = 'p_sacc_duration'
    p_sacc_azim_target = 'p_sacc_azim_target'

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        # self.states = np.ascontiguousarray(np.random.randint(2, size=vertices.shape[0]), dtype=np.float32)
        self.states = np.ascontiguousarray(np.random.rand(vertices.shape[0]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set up programs
        VERT = self.load_vertex_shader('./binary_noise_sphere.vert')
        FRAG = self.load_shader('./binary_noise_sphere.frag')

        # Set dot
        self.binary_noise = gloo.Program(VERT, FRAG)
        self.binary_noise['a_position'] = self.position_buffer
        self.binary_noise['a_state'] = self.state_buffer

        self.sacc_start_time = None
        self.u_rotate = np.eye(4)
        self.cur_azim = 0.

    def trigger_mock_saccade(self):
        self.sacc_start_time = self.binary_noise['u_time']

    def initialize(self, **params):
        self.binary_noise['u_time'] = 0.0

    def render(self, dt):

        sacc_duration = self.parameters.get(self.p_sacc_duration) / 1000
        sacc_azim_target = self.parameters.get(self.p_sacc_azim_target)

        if sacc_duration is None or sacc_azim_target is None:
            return

        # Reset azimuth
        if self.cur_azim < -360.:
            self.cur_azim += 360.
        elif self.cur_azim > 360:
            self.cur_azim -= 360.

        self.binary_noise['u_time'] += dt

        cur_time = self.binary_noise['u_time']
        if self.sacc_start_time is not None and cur_time > self.sacc_start_time:
            if cur_time - self.sacc_start_time <= sacc_duration:
                # Increment azimuth rotation
                self.cur_azim += sacc_azim_target * dt / sacc_duration
            else:
                # Saccade is over
                self.sacc_start_time = None

        # Apply rotation
        self.binary_noise['u_rotate'] = transforms.rotate(self.cur_azim, (0, 0, 1))

        # Draw dots
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', self.index_buffer)

    interface = [
        (p_sacc_duration, 200, 40, 2000, dict(step_size=5)),
        (p_sacc_azim_target, 30, -90, 90, dict(step_size=1)),
        ('trigger_saccade', trigger_mock_saccade)
    ]


class IcoNoise(visual.BaseVisual):

    interface = []


    def __init__(self, *args):
        visual.BaseVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=3)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        np.random.seed(1)
        self.states = np.ascontiguousarray(np.random.randint(2, size=vertices.shape[0]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set up program
        VERT = self.load_vertex_shader('./filtered_noise_sphere.vert')
        FRAG = self.load_shader('./filtered_noise_sphere.frag')
        self.binary_noise = gloo.Program(VERT, FRAG)
        self.binary_noise['a_position'] = self.position_buffer
        self.binary_noise['a_state'] = self.state_buffer

    def initialize(self, **params):
        self.binary_noise['u_time'] = 0.0

    def render(self, dt):
        self.binary_noise['u_time'] += dt

        # Draw dots
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', self.index_buffer)



class IcoGaussianNoise(visual.BaseVisual):

    interface = []


    def __init__(self, *args):
        visual.BaseVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=3)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        np.random.seed(1)
        self.states = np.ascontiguousarray(np.random.randint(2, size=vertices.shape[0]), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set up program
        VERT = self.load_vertex_shader('./filtered_noise_sphere.vert')
        FRAG = self.load_shader('./filtered_noise_sphere.frag')
        self.binary_noise = gloo.Program(VERT, FRAG)
        self.binary_noise['a_position'] = self.position_buffer
        self.binary_noise['a_state'] = self.state_buffer

    def initialize(self, **params):
        self.binary_noise['u_time'] = 0.0

    def render(self, dt):
        self.binary_noise['u_time'] += dt

        # Draw dots
        self.apply_transform(self.binary_noise)
        self.binary_noise.draw('triangles', self.index_buffer)
