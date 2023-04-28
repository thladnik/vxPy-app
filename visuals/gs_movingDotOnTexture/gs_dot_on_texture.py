"""
vxPy_app ./visuals/spherical_grating/spherical_grating.py
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


class MovingDotOnTexture2000(vxvisual.SphericalVisual):

    # Define general Parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Define Texture Parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    texture_default = vxvisual.Attribute('texture_default', static=True)
    luminance = vxvisual.FloatParameter('luminance', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'

    # Define Moving Dot parameters
    motion_axis = MotionAxis('motion_axis', static=True, default='vertical')
    dot_polarity = vxvisual.IntParameter('dot_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    dot_start_angle = vxvisual.FloatParameter('dot_start_angle', default=30, limits=(-180, 180), step_size=5, static=True)
    dot_angular_velocity = vxvisual.FloatParameter('dot_angular_velocity', default=-60, limits=(-360, 360), step_size=5, static=True)
    dot_angular_diameter = vxvisual.FloatParameter('dot_angular_diameter', default=20, limits=(1, 90), step_size=1, static=True)
    dot_offset_angle = vxvisual.FloatParameter('dot_offset_angle', default=-20, limits=(-85, 85), step_size=5, static=True)
    dot_location = vxvisual.Vec3Parameter('dot_location', default=0)

    # Paths to shaders
    VERT_PATH = './gs_dot_on_texture.vert'
    FRAG_PATH = './gs_dot_on_texture.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH),
                                         count=vertices.shape[0])
        self.rotating_dot['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)
        self.rotation.connect(self.rotating_dot)
        self.luminance.connect(self.rotating_dot)
        self.contrast.connect(self.rotating_dot)
        self.texture_default.connect(self.rotating_dot)
        self.motion_axis.connect(self.rotating_dot)
        self.dot_polarity.connect(self.rotating_dot)
        self.dot_start_angle.connect(self.rotating_dot)
        self.dot_angular_velocity.connect(self.rotating_dot)
        self.dot_angular_diameter.connect(self.rotating_dot)
        self.dot_offset_angle.connect(self.rotating_dot)
        self.dot_location.connect(self.rotating_dot)

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

        # dot location
        start_ang = self.dot_start_angle.data[0]
        ang_vel = self.dot_angular_velocity.data[0]
        dot_offset = self.dot_offset_angle.data[0]

        t_switch = 500
        t_stop = 1500
        if time < t_switch:
            dot_azim = start_ang + (time / 1000) * ang_vel / 180.0 * np.pi
            dot_elev = dot_offset / 180. * np.pi
            self.dot_location.data = sph2cart1(dot_azim, dot_elev, 1.)
        elif t_switch < time < t_stop:
            dot_azim = (start_ang + (t_switch / 1000) * ang_vel / 180.0 * np.pi) - \
                       ((time - t_switch) / 1000) * ang_vel / 180.0 * np.pi
            dot_elev = dot_offset / 180. * np.pi
            self.dot_location.data = sph2cart1(dot_azim, dot_elev, 1.)

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', indices=self.index_buffer)


class MovingDotOnTexture4000(vxvisual.SphericalVisual):

    # Define general Parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Define Texture Parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    texture_default = vxvisual.Attribute('texture_default', static=True)
    luminance = vxvisual.FloatParameter('luminance', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

    # Define Moving Dot parameters
    motion_axis = MotionAxis('motion_axis', static=True, default='vertical')
    dot_polarity = vxvisual.IntParameter('dot_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    dot_start_angle = vxvisual.FloatParameter('dot_start_angle', default=30, limits=(-180, 180), step_size=5, static=True)
    dot_angular_velocity = vxvisual.FloatParameter('dot_angular_velocity', default=-60, limits=(-360, 360), step_size=5, static=True)
    dot_angular_diameter = vxvisual.FloatParameter('dot_angular_diameter', default=20, limits=(1, 90), step_size=1, static=True)
    dot_offset_angle = vxvisual.FloatParameter('dot_offset_angle', default=-20, limits=(-85, 85), step_size=5, static=True)
    dot_location = vxvisual.Vec3Parameter('dot_location', default=0)

    # Paths to shaders
    VERT_PATH = './gs_dot_on_texture.vert'
    FRAG_PATH = './gs_dot_on_texture.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH),
                                         count=vertices.shape[0])
        self.rotating_dot['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)
        self.rotation.connect(self.rotating_dot)
        self.luminance.connect(self.rotating_dot)
        self.contrast.connect(self.rotating_dot)
        self.texture_default.connect(self.rotating_dot)
        self.motion_axis.connect(self.rotating_dot)
        self.dot_polarity.connect(self.rotating_dot)
        self.dot_start_angle.connect(self.rotating_dot)
        self.dot_angular_velocity.connect(self.rotating_dot)
        self.dot_angular_diameter.connect(self.rotating_dot)
        self.dot_offset_angle.connect(self.rotating_dot)
        self.dot_location.connect(self.rotating_dot)

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

        # dot location
        start_ang = self.dot_start_angle.data[0]
        ang_vel = self.dot_angular_velocity.data[0]
        dot_offset = self.dot_offset_angle.data[0]

        t_switch = 500
        t_stop = 1500
        if time < t_switch:
            dot_azim = start_ang + (time / 1000) * ang_vel / 180.0 * np.pi
            dot_elev = dot_offset / 180. * np.pi
            self.dot_location.data = sph2cart1(dot_azim, dot_elev, 1.)
        elif t_switch < time < t_stop:
            dot_azim = (start_ang + (t_switch / 1000) * ang_vel / 180.0 * np.pi) - \
                       ((time - t_switch) / 1000) * ang_vel / 180.0 * np.pi
            dot_elev = dot_offset / 180. * np.pi
            self.dot_location.data = sph2cart1(dot_azim, dot_elev, 1.)

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', indices=self.index_buffer)
