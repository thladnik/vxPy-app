import numpy as np
from vispy import gloo
from vispy.util import transforms

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class SingleLoomingDot(vxvisual.SphericalVisual):

    # Define general Parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Define Moving Dot parameters, static
    dot_polarity = vxvisual.IntParameter('dot_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, default='dark-on-light', static=True)
    dot_expansion_velocity = vxvisual.FloatParameter('dot_expansion_velocity', default=100, limits=(0, 500), step_size=5, static=True)
    dot_azimuth = vxvisual.FloatParameter('dot_azimuth', default=0, static=True)
    dot_elevation = vxvisual.FloatParameter('dot_elevation', default=0, static=True)

    # Varying
    dot_angular_diameter = vxvisual.FloatParameter('dot_angular_diameter', default=20, limits=(1, 90), step_size=1)

    # Paths to shaders
    VERT_PATH = 'dot_at_position.vert'
    FRAG_PATH = 'dot_at_position.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)

        # Set up program
        self.looming_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        # Set positions with buffers
        self.looming_dot['a_position'] = self.position_buffer
        self.looming_dot['static_rotation'] = transforms.rotate(0, (0, 0, 1))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.looming_dot)
        self.dot_polarity.connect(self.looming_dot)
        self.dot_expansion_velocity.connect(self.looming_dot)
        self.dot_angular_diameter.connect(self.looming_dot)
        self.dot_azimuth.connect(self.looming_dot)
        self.dot_elevation.connect(self.looming_dot)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Get times
        time = self.time.data[0]  # s

        self.dot_angular_diameter.data = time * self.dot_expansion_velocity.data[0]

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.looming_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.looming_dot.draw('triangles', indices=self.index_buffer)