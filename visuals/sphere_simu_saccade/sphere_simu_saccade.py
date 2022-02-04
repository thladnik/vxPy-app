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
    p_flash_polarity = 'p_flash_polarity'
    p_flash_start_time = 'p_flash_start_time'
    p_flash_duration = 'p_flash_duration'

    VERT_LOC = './sphere.vert'
    FRAG_LOC = ''

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        self.program: gloo.Program = None

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
        self.parameters[self.p_sacc_start_time] = self.program['u_time']

    def initialize(self, **params):
        self.program['u_time'] = 0.0

        if self.protocol is not None and hasattr(self.protocol, 'p_cur_azim'):
            self.cur_azim = self.protocol.p_cur_azim
        else:
            self.cur_azim = 0.0

    def render(self, dt):

        # Reset azimuth (unlikely to overflow, but you never know)
        if self.cur_azim < -360.:
            self.cur_azim += 360.
        elif self.cur_azim > 360:
            self.cur_azim -= 360.

        # Increment time
        self.program['u_time'] += dt
        cur_time = self.program['u_time']

        # Check if saccade was triggered
        sacc_start_time = self.parameters.get(self.p_sacc_start_time)
        if sacc_start_time is not None:

            # Get saccade parameters
            sacc_duration = self.parameters.get(self.p_sacc_duration) / 1000
            sacc_azim_target = self.parameters.get(self.p_sacc_azim_target)
            sacc_direction = self.parameters.get(self.p_sacc_direction)

            # Perform saccade
            if cur_time > sacc_start_time:
                if cur_time - sacc_start_time <= sacc_duration:
                    # If saccade is still happening: increment azimuth rotation
                    self.cur_azim += sacc_direction * sacc_azim_target * dt / sacc_duration
                else:
                    # If saccade is over
                    self.parameters[self.p_sacc_start_time] = -np.inf

        # Perform flash
        flash_start_time = self.parameters.get(self.p_flash_start_time)
        if flash_start_time is not None:
            flash_duration = self.parameters.get(self.p_flash_duration) / 1000
            flash_polarity = self.parameters.get(self.p_flash_polarity)
            if cur_time > flash_start_time and cur_time - flash_start_time <= flash_duration:
                self.program['u_flash_polarity'] = flash_polarity
            else:
                self.program['u_flash_polarity'] = 0

            if self.protocol is not None:
                self.protocol.p_cur_azim = self.cur_azim

        # Apply rotation
        self.program['u_rotate'] = transforms.rotate(self.cur_azim, (0, 0, 1))

        # Draw
        self.apply_transform(self.program)
        self.program.draw('triangles', self.index_buffer)

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

        lum_decrease = 0.2
        lum_increase = 0.2

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=5)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(vertices)
        self.binary_noise['a_position'] = self.position_buffer

        bias = 0.1
        intensity = np.ascontiguousarray(np.random.rand(self.position_buffer.size) < (1. - bias), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set vertex states
        self.program['a_position'] = vertices
        self.program['a_texture_normal'] = intensity
        self.program['a_texture_dark'] = intensity - lum_decrease
        self.program['a_texture_light'] = intensity + lum_increase


class IcoGaussianConvolvedNoiseSphereWithSimulatedHorizontalSaccade(IcoSphereWithSimulatedHorizontalSaccade):
    FRAG_LOC = './smooth_noise_sphere.frag'

    def __init__(self, *args, **kwargs):
        IcoSphereWithSimulatedHorizontalSaccade.__init__(self, *args, **kwargs)

        lum_decrease = 0.2
        lum_increase = 0.2
        # texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest20220228a.mat'
        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.mat'

        configuration_key = str((texture_file, lum_decrease, lum_increase))

        # Check if configuration is stored in protocol instance
        if self.protocol is not None:
            if not hasattr(self.protocol, 'configurations'):
                self.protocol.configurations = {}

            # Fetch stored program with same configuration
            stored = self.protocol.configurations.get(configuration_key)

            if stored is not None:
                self.program, self.index_buffer = stored
                return

        # Load model data
        d = io.loadmat(texture_file)

        # Vertices
        x, y, z = d['grid']['x'][0][0], \
                  d['grid']['y'][0][0], \
                  d['grid']['z'][0][0]

        vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T

        # Faces
        idcs = list()
        azim_lvls = x.shape[1]
        elev_lvls = x.shape[0]
        for i in np.arange(elev_lvls):
            for j in np.arange(azim_lvls):
                idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
                idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
        indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)
        indices = indices[indices < azim_lvls * elev_lvls]
        self.index_buffer = gloo.IndexBuffer(indices)

        # Intensities
        intensity = d['totalintensity'].flatten()

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.program = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.program['a_position'] = vertices
        self.program['a_texture_normal'] = intensity
        self.program['a_texture_dark'] = intensity - lum_decrease
        self.program['a_texture_light'] = intensity + lum_increase

        self.protocol.configurations[configuration_key] = self.program, self.index_buffer

