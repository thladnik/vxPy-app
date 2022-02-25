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
import h5py
import numpy as np
import scipy.io
from vispy import gloo
from vispy.util import transforms

import vxpy.core.visual as vxvisual


class GaussianConvNoiseSphereSimuSaccade(vxvisual.SphericalVisual):

    VERT_LOC = './sphere.vert'
    FRAG_LOC = './smooth_noise_sphere.frag'

    u_time = vxvisual.FloatParameter('u_time', default=0.0, limits=(0.0, 20.0))

    saccade_start_time = vxvisual.FloatParameter('saccade_start_time', static=True, default=2.0, limits=(0.1, 10.0))
    saccade_duration = vxvisual.UIntParameter('saccade_duration', static=True, default=200, limits=(20, 1000))
    saccade_azim_target = vxvisual.FloatParameter('saccade_azim_target', static=True, default=15.0, limits=(1.0, 60.0))
    saccade_direction = vxvisual.IntParameter('saccade_direction', static=True, default=1, limits=(-1, 1))

    flash_start_time = vxvisual.FloatParameter('flash_start_time', static=True, default=2.5, limits=(-1.0, 20.0))
    flash_duration = vxvisual.UIntParameter('flash_duration', static=True, default=50, limits=(20, 1000))
    flash_polarity = vxvisual.IntParameter('flash_polarity', static=True, default=1, limits=(-1, 1))

    texture_normal = vxvisual.Texture2D('texture_normal', static=True)
    texture_light = vxvisual.Texture2D('texture_light', static=True)
    texture_dark = vxvisual.Texture2D('texture_dark', static=True)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Additional parameters
        lum_decrease = 0.2
        lum_increase = 0.2
        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Set initial rotation matrix
        self.u_rotate = np.eye(4)

        # Set initial azimuth
        self.cur_azim = 0.

        # Add triggers to visual
        self.trigger_functions.append(self.trigger_simulated_saccade)
        self.trigger_functions.append(self.reset_time)

        # Load model and texture
        # vertices, indices, intensity = self._import_texture_from_mat(texture_file)
        vertices, indices, intensity = self._import_texture_from_hdf(texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.program = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.program['a_position'] = vertices

        self.texture_normal.data = intensity
        self.texture_light.data = intensity + lum_increase
        self.texture_dark.data = intensity - lum_decrease

        self.u_time.connect(self.program)
        self.texture_normal.connect(self.program)
        self.texture_light.connect(self.program)
        self.texture_dark.connect(self.program)

    def trigger_simulated_saccade(self):
        self.u_time.data = self.saccade_start_time.data

    def reset_time(self):
        self.u_time.data = 0.0

    def initialize(self, **params):
        self.reset_time()

        # If given protocol has an azimuth angle stored, use this as start value
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
        self.u_time.data = self.u_time.data + dt
        cur_time = self.u_time.data

        # Check if saccade was triggered
        sacc_start_time = self.saccade_start_time.data
        if sacc_start_time is not None:

            # Get saccade parameters
            sacc_duration = self.saccade_duration.data / 1000
            sacc_azim_target = self.saccade_azim_target.data
            sacc_direction = self.saccade_direction.data

            # Perform saccade
            if cur_time > sacc_start_time:
                if cur_time - sacc_start_time <= sacc_duration:
                    # If saccade is still happening: increment azimuth rotation
                    self.cur_azim += sacc_direction * sacc_azim_target * dt / sacc_duration

        # Perform flash
        flash_start_time = self.flash_start_time.data
        if flash_start_time is not None:
            flash_duration = self.flash_duration.data / 1000
            flash_polarity = self.flash_polarity.data
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

    @staticmethod
    def _import_texture_from_hdf(texture_file):
        with h5py.File(texture_file, 'r') as f:
            vertices = f['vertices'][:]
            indices = f['indices'][:]
            intensity = f['intensity'][:]

        return vertices, indices, intensity

    @staticmethod
    def _import_texture_from_mat(texture_file):
        # Load model data
        d = scipy.io.loadmat(texture_file)

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

        # Get intensity
        intensity = d['totalintensity'].flatten()

        return vertices, indices, intensity