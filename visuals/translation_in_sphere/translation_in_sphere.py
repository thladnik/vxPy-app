"""
vxPy_app ./visuals/translation_in_sphere.py
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
from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class MotionAxis(vxvisual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        elevation = TranslationGrating.elevation.data[0]
        azimuth = TranslationGrating.azimuth.data[0]

        rot_elevation = transforms.rotate(90. - elevation, (0, 1, 0))
        rot_azimuth = transforms.rotate(azimuth, (0, 0, 1))
        self.data = np.dot(rot_elevation, rot_azimuth)


class TranslationGrating(vxvisual.SphericalVisual):

    # (optional) Add a short description
    description = ''

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    elevation = vxvisual.FloatParameter('elevation', default=00, limits=(-90, 90), step_size=1, static=True)
    azimuth = vxvisual.FloatParameter('azimuth', default=00, limits=(-180, 180), step_size=1, static=True)
    motion_axis = MotionAxis('motion_axis', static=True, internal=True)
    angular_velocity = vxvisual.FloatParameter('angular_velocity', default=30, limits=(-180, 180), step_size=5, static=True)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=5, static=True)

    # Paths to shaders
    VERT_PATH = './sphere.vert'
    FRAG_PATH = './translation_grating.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi/2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.motion_axis.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)

        # Link motion axis to be updated when elevation or azimuth changes
        self.elevation.add_downstream_link(self.motion_axis)
        self.azimuth.add_downstream_link(self.motion_axis)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # TEMP!
        # self.azimuth.data = 2 * self.time.data[0]

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
