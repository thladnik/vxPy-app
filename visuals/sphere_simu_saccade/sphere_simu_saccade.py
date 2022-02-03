"""
vxpy_app ./visuals/sphere_simu_saccade/sphere_simu_saccade.py
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
from vispy import scene
from vispy.util import transforms
from scipy import io

from vxpy.core import visual
from vxpy.utils import sphere


class IcoSphereWithSimulatedHorizontalSaccade(visual.SphericalVisual):
    p_sacc_duration = 'p_sacc_duration'
    p_sacc_azim_target = 'p_sacc_azim_target'
    p_sacc_start_time = 'p_sacc_start_time'
    p_sacc_direction = 'p_sacc_direction'
    p_flash_delay = 'p_flash_delay'
    p_flash_start_time = 'p_flash_start_time'
    p_flash_duration = 'p_flash_duration'

    VERT_LOC = './sphere.vert'
    FRAG_LOC = ''

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.binary_noise = gloo.Program(VERT, FRAG)

        # Set saccade start time
        # self.sacc_start_time = None

        # Set initial rotation matrix
        self.u_rotate = np.eye(4)

        # Set initial azimuth
        self.cur_azim = 0.

        # Set up buffer
        self.states: np.ndarray = None
        self.flash: np.ndarray = None
        self.state_buffer: gloo.VertexBuffer = None

        # Keep seed fixed for now
        np.random.seed(1)

    def trigger_mock_saccade(self):
        self.parameters[self.p_sacc_start_time] = self.binary_noise['u_time']
        # self.sacc_start_time = self.binary_noise['u_time']

    def initialize(self, **params):
        self.binary_noise['u_time'] = 0.0

        if self.protocol is not None and hasattr(self.protocol, 'p_cur_azim'):
            self.cur_azim = self.protocol.p_cur_azim
        else:
            self.cur_azim = 0.0

    def render(self, dt):

        # Get saccade parameters
        sacc_duration = self.parameters.get(self.p_sacc_duration) / 1000
        sacc_azim_target = self.parameters.get(self.p_sacc_azim_target)
        sacc_direction = self.parameters.get(self.p_sacc_direction)

        if sacc_duration is None or sacc_azim_target is None:
            return

        # Reset azimuth (unlikely to overflow, but you never know)
        if self.cur_azim < -360.:
            self.cur_azim += 360.
        elif self.cur_azim > 360:
            self.cur_azim -= 360.

        # Increment time
        self.binary_noise['u_time'] += dt
        cur_time = self.binary_noise['u_time']

        # Check if saccade was triggered
        sacc_start_time = self.parameters.get(self.p_sacc_start_time)
        if sacc_start_time is not None:
            # Set flash start time
            # flash_delay = self.parameters.get(self.p_flash_delay)
            # if flash_delay is not None:
            #     self.parameters[self.p_flash_start_time] = sacc_start_time + flash_delay / 1000

            # Perform saccade
            if cur_time > sacc_start_time:
                # If saccade is still happening: increment azimuth rotation
                if cur_time - sacc_start_time <= sacc_duration:
                    self.cur_azim += sacc_direction * sacc_azim_target * dt / sacc_duration
                # Saccade is over
                else:
                    self.parameters[self.p_sacc_start_time] = -np.inf

        # Perform flash
        flash_start_time = self.parameters.get(self.p_flash_start_time)
        flash_duration = self.parameters.get(self.p_flash_duration) / 1000
        if cur_time > flash_start_time:
            if cur_time - flash_start_time <= flash_duration:
                self.state_buffer[:] = self.flash
            else:
                self.state_buffer[:] = self.states

        if self.protocol is not None:
            self.protocol.p_cur_azim = self.cur_azim

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


class IcoNoiseSphereWithSimulatedHorizontalSaccade(IcoSphereWithSimulatedHorizontalSaccade):
    FRAG_LOC = './smooth_noise_sphere.frag'

    def __init__(self, *args, **kwargs):
        IcoSphereWithSimulatedHorizontalSaccade.__init__(self, *args, **kwargs)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        self.binary_noise['a_position'] = self.position_buffer

        self.states = np.ascontiguousarray(np.random.rand(self.position_buffer.size), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set vertex states
        self.binary_noise['a_state'] = self.state_buffer


class IcoBinaryNoiseSphereWithSimulatedHorizontalSaccade(IcoSphereWithSimulatedHorizontalSaccade):
    FRAG_LOC = './binary_noise_sphere.frag'

    def __init__(self, *args, **kwargs):
        IcoSphereWithSimulatedHorizontalSaccade.__init__(self, *args, **kwargs)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        self.binary_noise['a_position'] = self.position_buffer

        bias = 0.1
        self.states = np.ascontiguousarray(np.random.rand(self.position_buffer.size) < (1. - bias), dtype=np.float32)
        self.flash = np.ascontiguousarray(np.zeros(self.position_buffer.size), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set vertex states
        self.binary_noise['a_state'] = self.state_buffer


class IcoGaussianConvolvedNoiseSphereWithSimulatedHorizontalSaccade(IcoSphereWithSimulatedHorizontalSaccade):
    FRAG_LOC = './smooth_noise_sphere.frag'

    def __init__(self, *args, **kwargs):
        IcoSphereWithSimulatedHorizontalSaccade.__init__(self, *args, **kwargs)

        d = io.loadmat('visuals/sphere_simu_saccade/blobstimtest.mat')
        step_size = 100
        x, y, z = d['grid']['x'][0][0][::step_size, ::step_size], \
                  d['grid']['y'][0][0][::step_size, ::step_size], \
                  d['grid']['z'][0][0][::step_size, ::step_size]

        v = np.array([x.flatten(), y.flatten(), z.flatten()])
        vertices = np.ascontiguousarray(v.T)
        self.position_buffer = gloo.VertexBuffer(vertices)
        self.binary_noise['a_position'] = self.position_buffer

        idcs = list()
        azim_lvls = x.shape[0]
        elev_lvls = x.shape[1]
        for i in np.arange(elev_lvls):
            for j in np.arange(azim_lvls):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
        self.indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)
        self.index_buffer = gloo.IndexBuffer(self.indices)

        self.flash = np.ascontiguousarray(np.zeros(self.position_buffer.size), dtype=np.float32)
        self.states = np.ascontiguousarray(np.repeat(d['totalintensity'][::step_size, ::step_size].flatten(), 3, axis=-1))
        self.state_buffer = gloo.VertexBuffer(self.states)
        self.binary_noise['a_state'] = self.state_buffer
