# -*- coding: utf-8 -*-
"""Collection of grating stimuli


"""
from vispy import gloo

import vxpy.core.visual as vxvisual
from vxpy.utils import plane


class BlackAndWhiteGrating(vxvisual.PlanarVisual):
    """Display a black and white grating stimulus on a planar surface
    """
    # (optional) Add a short description
    description = 'Black und white contrast grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    direction = vxvisual.IntParameter('direction', value_map={'vertical': 1, 'horizontal': 2}, static=True)
    linear_velocity = vxvisual.FloatParameter('linear_velocity', default=10, limits=(-100, 100), step_size=5, static=True)
    spatial_period = vxvisual.FloatParameter('spatial_period', default=10, limits=(-100, 100), step_size=5, static=True)

    def __init__(self, *args, **kwargs):
        vxvisual.PlanarVisual.__init__(self, *args, **kwargs)

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

        # Set positions with vertex buffer
        self.grating['a_position'] = self.position_buffer

        # Connect
        self.time.connect(self.grating)
        self.waveform.connect(self.grating)
        self.direction.connect(self.grating)
        self.linear_velocity.connect(self.grating)
        self.spatial_period.connect(self.grating)

    def initialize(self, *args, **kwargs):
        # Reset time to 0.0 on each visual initialization
        self.time.data = 0.0

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
