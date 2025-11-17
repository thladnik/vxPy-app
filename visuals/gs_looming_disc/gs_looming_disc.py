"""
vxPy_app ./visuals/spherical_grating/ml_rotating_grating.py
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
from vispy import gloo
import scipy.io
from vispy.util import transforms
import numpy as np

from vxpy.core import visual
from vxpy.utils import sphere
import vxpy.core.visual as vxvisual
from vxpy.utils.geometry import sph2cart1


def _convert_mat_texture_to_hdf(mat_path: str, hdf_path: str):
    d = scipy.io.loadmat(mat_path)
    x, y, z = d['grid']['x'][0][0], \
              d['grid']['y'][0][0], \
              d['grid']['z'][0][0]

    v = np.array([x.flatten(), y.flatten(), z.flatten()])
    vertices = np.ascontiguousarray(v.T, dtype=np.float32)

    idcs = list()
    azim_lvls = x.shape[1]
    elev_lvls = x.shape[0]
    for i in np.arange(elev_lvls):
        for j in np.arange(azim_lvls):
            idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
            idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
    indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)
    indices = indices[indices < azim_lvls * elev_lvls]

    states = np.ascontiguousarray(d['totalintensity'].flatten(), dtype=np.float32)

    with h5py.File(hdf_path, 'w') as f:
        f.create_dataset('vertices', data=vertices)
        f.create_dataset('indices', data=indices)
        f.create_dataset('intensity', data=states)


def _import_texture_from_hdf(texture_file):
    with h5py.File(texture_file, 'r') as f:
        vertices = f['vertices'][:]
        indices = f['indices'][:]
        intensity = f['intensity'][:]

    return vertices, indices, intensity


class MotionAxis(visual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        visual.Mat4Parameter.__init__(self, *args, **kwargs)

        self.value_map = {'forward': self._rotate_forward,
                          'sideways': self._rotate_sideways,
                          'vertical': np.eye(4)}

    @staticmethod
    def _rotate_forward():
        return transforms.rotate(90, (0, 1, 0))

    @staticmethod
    def _rotate_sideways():
        return transforms.rotate(90, (1, 0, 0))


class LoomingDiscOnTexture2000(vxvisual.SphericalVisual):

    # Define general Parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0), step_size=0.01)

    # Define Texture Parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    texture_default = vxvisual.Attribute('texture_default', static=True)
    luminance = vxvisual.FloatParameter('luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'

    # Define Moving Dot parameters
    motion_axis = MotionAxis('motion_axis', static=True, default='vertical')
    disc_polarity = vxvisual.IntParameter('disc_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    disc_azimuth = vxvisual.FloatParameter('disc_azimuth', default=0, limits=(-180, 180), step_size=5, static=True) # in °
    disc_current_azimuth = vxvisual.FloatParameter('disc_current_azimuth', default = 0)
    disc_elevation = vxvisual.FloatParameter('disc_elevation', default=-90, limits=(-90, 90), step_size=5,
                                           static=True) # in °
    disc_starting_diameter = vxvisual.FloatParameter('disc_starting_diameter', default=2, limits=(1, 90), step_size=1, static=True) # in °
    disc_expansion_lv = vxvisual.FloatParameter('disc_expansion_lv', default = 200, limits=(5,500), step_size=5, static=True) # in ms
    disc_diameter = vxvisual.FloatParameter('disc_diameter', default=0) # in °



    # Paths to shaders
    VERT_PATH = 'gs_looming_disc.vert'
    FRAG_PATH = 'gs_looming_disc.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        self.looming_disc = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH),
                                         count=vertices.shape[0])
        self.looming_disc['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.looming_disc)
        self.rotation.connect(self.looming_disc)
        self.luminance.connect(self.looming_disc)
        self.contrast.connect(self.looming_disc)
        self.texture_default.connect(self.looming_disc)
        self.motion_axis.connect(self.looming_disc)
        self.disc_polarity.connect(self.looming_disc)
        self.disc_azimuth.connect(self.looming_disc)
        self.disc_current_azimuth.connect(self.looming_disc)
        self.disc_elevation.connect(self.looming_disc)
        self.disc_starting_diameter.connect(self.looming_disc)
        self.disc_expansion_lv.connect(self.looming_disc)
        self.disc_diameter.connect(self.looming_disc)

        self.protocol.global_visual_props['azim_angle'] = 0.

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Set rotation
        self.rotation.data = transforms.rotate(self.protocol.global_visual_props['azim_angle'], (0, 0, 1))

        # Set luminance
        baseline_lum = self.luminance.data[0]
        self.luminance.data = baseline_lum

        # set global texture azimuth
        text_azim = self.protocol.global_visual_props['azim_angle'] / 180. * np.pi

        # disc location (adjusted for change in texture location)
        disc_azim = self.disc_azimuth.data[0]

        current_azim = (disc_azim + text_azim) + (time / 1000) * np.pi

        self.disc_current_azimuth.data = current_azim

        # disc size
        start_size = self.disc_starting_diameter.data[0]
        expansion_lv = self.disc_expansion_lv.data[0]

        current_diameter = start_size * np.exp(time/expansion_lv)

        self.disc_diameter.data = current_diameter

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.looming_disc)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.looming_disc.draw('triangles', indices=self.index_buffer)


class LoomingDiscOnTexture4000(vxvisual.SphericalVisual):

    # Define general Parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0), step_size=0.01)

    # Define Texture Parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    texture_default = vxvisual.Attribute('texture_default', static=True)
    luminance = vxvisual.FloatParameter('luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

    # Define Moving Dot parameters
    motion_axis = MotionAxis('motion_axis', static=True, default='vertical')
    disc_polarity = vxvisual.IntParameter('disc_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    disc_azimuth = vxvisual.FloatParameter('disc_azimuth', default=0, limits=(-180, 180), step_size=5, static=True) # in °
    disc_current_azimuth = vxvisual.FloatParameter('disc_current_azimuth', default = 0)
    disc_elevation = vxvisual.FloatParameter('disc_elevation', default=-90, limits=(-90, 90), step_size=5,
                                           static=True) # in °
    disc_starting_diameter = vxvisual.FloatParameter('disc_starting_diameter', default=2, limits=(1, 90), step_size=1, static=True) # in °
    disc_expansion_lv = vxvisual.FloatParameter('disc_expansion_lv', default = 200, limits=(5,500), step_size=5, static=True) # in ms
    disc_diameter = vxvisual.FloatParameter('disc_diameter', default=0) # in °



    # Paths to shaders
    VERT_PATH = 'gs_looming_disc.vert'
    FRAG_PATH = 'gs_looming_disc.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        self.looming_disc = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH),
                                         count=vertices.shape[0])
        self.looming_disc['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.looming_disc)
        self.rotation.connect(self.looming_disc)
        self.luminance.connect(self.looming_disc)
        self.contrast.connect(self.looming_disc)
        self.texture_default.connect(self.looming_disc)
        self.motion_axis.connect(self.looming_disc)
        self.disc_polarity.connect(self.looming_disc)
        self.disc_azimuth.connect(self.looming_disc)
        self.disc_current_azimuth.connect(self.looming_disc)
        self.disc_elevation.connect(self.looming_disc)
        self.disc_starting_diameter.connect(self.looming_disc)
        self.disc_expansion_lv.connect(self.looming_disc)
        self.disc_diameter.connect(self.looming_disc)

        self.protocol.global_visual_props['azim_angle'] = 0.

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Set rotation
        self.rotation.data = transforms.rotate(self.protocol.global_visual_props['azim_angle'], (0, 0, 1))

        # Set luminance
        baseline_lum = self.luminance.data[0]
        self.luminance.data = baseline_lum

        # set global texture azimuth
        text_azim = self.protocol.global_visual_props['azim_angle'] / 180. * np.pi

        # disc location (adjusted for change in texture location)
        disc_azim = self.disc_azimuth.data[0]

        current_azim = (disc_azim + text_azim) + (time / 1000) * np.pi

        self.disc_current_azimuth.data = current_azim

        # disc size
        start_size = self.disc_starting_diameter.data[0]
        expansion_lv = self.disc_expansion_lv.data[0]

        current_diameter = start_size * np.exp(time/expansion_lv)

        self.disc_diameter.data = current_diameter

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.looming_disc)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.looming_disc.draw('triangles', indices=self.index_buffer)