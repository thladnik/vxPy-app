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
from vispy import gloo
from vispy.util import transforms
import numpy as np

from vxpy.core import visual as vxvisual
from vxpy.utils import sphere


class SphericalBlackWhiteGrating(vxvisual.SphericalVisual):
    """Black und white contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'Spherical black und white contrast grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    motion_type = vxvisual.IntParameter('motion_type', static=True)
    angular_velocity = vxvisual.FloatParameter('angular_velocity', default=30, limits=(0, 360), step_size=5, static=True)
    color = vxvisual.Vec3Parameter('color', default=30, limits=(0, 360), step_size=5)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=5, static=True)

    # Paths to shaders
    VERT_PATH = 'spherical_grating.vert'
    FRAG_PATH = 'spherical_grating.frag'


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
        self.waveform.connect(self.grating)
        self.motion_type.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.color.connect(self.grating)

        # Alternative way of setting value_map: during instance creation
        self.motion_type.value_map = {'translation': 1, 'rotation': 2}

    def saccade_happened(self):
        self.saccade_start.data = self.time.data[0] + 0.200

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0
        self.saccade_start.data = 1.5

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        self.color.data = [1., 1., 1.]

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)

