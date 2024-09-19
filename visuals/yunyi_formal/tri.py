

from vispy import gloo

import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class Triangle(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    # Paths to shaders
    VERT_PATH = 'position.vert'
    FRAG_PATH = 'texture.frag'

    # center = geometry.sph2cart(5 * np.pi / 9, -np.pi / 6, 1.)
    # distance_verticle_pos = geometry.sph2cart(5 * np.pi / 9, 0, 1.)
    # distance_horizontal_pos = geometry.sph2cart(13 * np.pi / 18, -np.pi / 6, 1.)
    # distance_verticle = distance_verticle_pos[2] - center[2]
    # distance_horizontal = distance_horizontal_pos[1] - center[1]

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)
        # self.distance_verticle.connect(self.rotating_dot)
        # self.distance_horizontal.connect(self.rotating_dot)
        # self.center.connect(self.rotating_dot)
        # Add reset trigger
        self.trigger_functions.append(self.reset_time)

    def reset_time(self):
        self.time.data = 0.0

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.reset_time()
        # Set positions with buffers
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)


