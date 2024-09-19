import vxpy.core.visual as vxvisual
from vispy import gloo
from vxpy.utils import sphere
import numpy as np


class CMN_foreground(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.simple_tri(azimuth_range=2 * np.pi,
                                        elev_range=3 * np.pi / 4,
                                        radius=1.0)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        #self.position_buffer = gloo.VertexBuffer(self.sphere.verticies)

        # Create program
        vert = self.load_vertex_shader('./position.vert')
        frag = self.load_shader('./texture.frag')
        self.sphere_program = gloo.Program(vert, frag)

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.sphere_program)

        # Add reset trigger
        self.trigger_functions.append(self.reset_time)

    def reset_time(self):
        self.time.data = 0.0

    def initialize(self, **params):
        self.reset_time()
        #self.sphere_program['a_position'] = self.position_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt
        self.apply_transform(self.sphere_program)
        self.sphere_program.draw('triangles', self.sphere.indices)
