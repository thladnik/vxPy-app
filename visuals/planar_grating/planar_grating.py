"""
vxpy ./visuals/planar/grating.py
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
from vispy import gloo

from vxpy.api.visual import PlanarVisual
from vxpy.utils import plane
import vxpy.core.visual as vxvisual


class BlackAndWhiteGrating(PlanarVisual):
    # (optional) Add a short description
    description = 'Black und white contrast grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    direction = vxvisual.IntParameter('direction', value_map={'vertical': 1, 'horizontal': 2}, static=True)
    linear_velocity = vxvisual.FloatParameter('linear_velocity', default=10, limits=(-100, 100), step_size=5, static=True)
    spatial_period = vxvisual.FloatParameter('spatial_period', default=10, limits=(-100, 100), step_size=5, static=True)

    def __init__(self, *args, **kwargs):
        PlanarVisual.__init__(self, *args, **kwargs)

        # Set up model of a 2d plane
        self.plane_2d = plane.XYPlane()

        # Get vertex positions and corresponding face indices
        faces = self.plane_2d.indices
        vertices = self.plane_2d.a_position

        # Create vertex and index buffers
        self.index_buffer = gloo.IndexBuffer(faces)
        self.position_buffer = gloo.VertexBuffer(vertices)

        # Create a shader program
        vert = self.load_vertex_shader('./planar_grating.vert')
        frag = self.load_shader('./planar_grating.frag')
        self.grating = gloo.Program(vert, frag)

        self.time.connect(self.grating)
        self.waveform.connect(self.grating)
        self.direction.connect(self.grating)
        self.linear_velocity.connect(self.grating)
        self.spatial_period.connect(self.grating)

    def initialize(self, *args, **kwargs):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with vertex buffer
        self.grating['a_position'] = self.position_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
