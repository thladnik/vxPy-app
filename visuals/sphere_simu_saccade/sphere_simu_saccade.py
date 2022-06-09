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


def _import_texture_from_hdf(texture_file):
    with h5py.File(texture_file, 'r') as f:
        vertices = f['vertices'][:]
        indices = f['indices'][:]
        intensity = f['intensity'][:]

    return vertices, indices, intensity


class SinusAmplitude(vxvisual.FloatParameter):
    def __init__(self, *args, **kwargs):
        vxvisual.FloatParameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):

        time = TextureSinusLumModulation2000.time.data[0]
        amp = TextureSinusLumModulation2000.sine_luminance_amplitude.data[0]
        freq = TextureSinusLumModulation2000.sine_luminance_frequency.data[0]
        mean = TextureSinusLumModulation2000.sine_mean_luminance.data[0]

        lum = mean + np.sin(freq * time * 2.0 * np.pi) * amp / 2.0
        TextureSinusLumModulation2000.luminance.data = lum


class TextureSinusLumModulation2000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_meanlum_mod.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # Absolute contrast
    contrast = vxvisual.FloatParameter('contrast', default=0.3, limits=(0.0, 1.0), step_size=0.01)

    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(-360.0, 360.0))
    texture_default = vxvisual.Attribute('texture_default', static=True)

    sine_luminance_amplitude = SinusAmplitude('sine_luminance_amplitude', static=True, default=0.30, limits=(0.0, 1.0), step_size=0.01)
    sine_luminance_frequency = vxvisual.FloatParameter('sine_luminance_frequency', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)
    sine_mean_luminance = vxvisual.FloatParameter('sine_mean_luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.time.add_downstream_link(self.sine_luminance_amplitude)
        self.rotation.value_map = self._rotate

        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Rotation
        self.rotation.data = 0

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        self.time.connect(self.tex_mod)
        self.rotation.connect(self.tex_mod)
        self.luminance.connect(self.tex_mod)
        self.contrast.connect(self.tex_mod)
        self.texture_default.connect(self.tex_mod)

    @staticmethod
    def _rotate(angle):
        return transforms.rotate(angle, (0, 0, 1))

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        self.apply_transform(self.tex_mod)
        self.tex_mod.draw('triangles', indices=self.index_buffer)


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

    def _import_texture_from_hdf(texture_file):
        with h5py.File(texture_file, 'r') as f:
            vertices = f['vertices'][:]
            indices = f['indices'][:]
            intensity = f['intensity'][:]

        return vertices, indices, intensity

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Additional parameters
        lum_decrease = 0.2
        lum_increase = 0.2
        texture_file = 'visuals/gs_flash_tests/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'

        # Set initial rotation matrix
        self.u_rotate = np.eye(4)

        # Set initial azimuth
        self.cur_azim = 0.

        # Add triggers to visual
        self.trigger_functions.append(self.trigger_simulated_saccade)
        self.trigger_functions.append(self.reset_time)

        # Load model and texture
        # vertices, indices, intensity = self._import_texture_from_mat(texture_file)
        #vertices, indices, intensity = self._import_texture_from_hdf(texture_file)
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

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


class TextureRotation2000(vxvisual.SphericalVisual):


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

    def _import_texture_from_hdf(texture_file):
        with h5py.File(texture_file, 'r') as f:
            vertices = f['vertices'][:]
            indices = f['indices'][:]
            intensity = f['intensity'][:]

        return vertices, indices, intensity

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Additional parameters
        lum_decrease = 0.2
        lum_increase = 0.2
        texture_file = 'visuals/gs_flash_tests/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'

        # Set initial rotation matrix
        self.u_rotate = np.eye(4)

        # Set initial azimuth
        self.cur_azim = 0.

        # Add triggers to visual
        self.trigger_functions.append(self.trigger_simulated_saccade)
        self.trigger_functions.append(self.reset_time)

        # Load model and texture
        # vertices, indices, intensity = self._import_texture_from_mat(texture_file)
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

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


class TextureRotation4000(vxvisual.SphericalVisual):

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

    def _import_texture_from_hdf(texture_file):
        with h5py.File(texture_file, 'r') as f:
            vertices = f['vertices'][:]
            indices = f['indices'][:]
            intensity = f['intensity'][:]

        return vertices, indices, intensity

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Additional parameters
        lum_decrease = 0.2
        lum_increase = 0.2
        texture_file = 'visuals/gs_flash_tests/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

        # Set initial rotation matrix
        self.u_rotate = np.eye(4)

        # Set initial azimuth
        self.cur_azim = 0.

        # Add triggers to visual
        self.trigger_functions.append(self.trigger_simulated_saccade)
        self.trigger_functions.append(self.reset_time)

        # Load model and texture
        # vertices, indices, intensity = self._import_texture_from_mat(texture_file)
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

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
