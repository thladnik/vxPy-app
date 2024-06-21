from vispy import gloo

import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class Triangle(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    # Paths to shaders
    VERT_PATH = 'position.vert'
    FRAG_PATH = 'texture.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)
        self.sphere = sphere.Trispherep
        self.vertices = self.sphere.vertices
        self.indices = gloo.IndexBuffer(self.sphere.indices)
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        self.position_buffer = gloo.VertexBuffer(self.vertices)
        self.index_buffer = gloo.IndexBuffer(self.indices)


        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)

        # Add reset trigger
        self.trigger_functions.append(self.reset_time)

    def reset_time(self):
        self.time.data = 0.0

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.reset_time()

        # Set positions with buffers
        self.rotating_dot['a_position'] = self.position_buffer
    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)