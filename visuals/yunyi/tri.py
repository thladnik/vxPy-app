import vxpy.utils.geometry as Geometry
from vispy import gloo
import numpy as np
import vxpy.core.visual as vxvisual
class triangle(vxvisual.SphericalVisual):
 def __init__(self, *args, **kwargs):
    # Create program
    vert = self.load_vertex_shader('./position.vert')
    frag = self.load_shader('./texture.frag')
    self.sphere_program = gloo.Program(vert, frag)
    self.r = (1 + np.sqrt(5)) / 2
    self.a_position = np.array([
       [-1.0, self.r, 0.0],
       [1.0, self.r, 0.0],
       [-1.0, -self.r, 0.0],
       [1.0, -self.r, 0.0],
       [0.0, -1.0, self.r],
       [0.0, 1.0, self.r],
       [0.0, -1.0, -self.r],
       [0.0, 1.0, -self.r],
       [self.r, 0.0, -1.0],
       [self.r, 0.0, 1.0],
       [-self.r, 0.0, -1.0],
       [-self.r, 0.0, 1.0]
    ])
    self.indices = np.array([
       [0, 11, 5],
       [0, 5, 1],
       [0, 1, 7],
       [0, 7, 10],
       [0, 10, 11],
       [1, 5, 9],
       [5, 11, 4],
       [11, 10, 2],
       [10, 7, 6],
       [7, 1, 8],
       [3, 9, 4],
       [3, 4, 2],
       [3, 2, 6],
       [3, 6, 8],
       [3, 8, 9],
       [4, 9, 5],
       [2, 4, 11],
       [6, 2, 10],
       [8, 6, 7],
       [9, 8, 1]
    ])

    self.index_buffer = gloo.IndexBuffer(self.indices)
    self.position_buffer = gloo.VertexBuffer(np.float32(self.a_position))
    self.sphere_program['a_position'] = self.position_buffer

    self.binary_texture = np.uint8(np.random.randint(0, 2, [75, 75, 1]) * np.array([[[1, 1, 1]]]) * 255)
    self.sphere_program['u_texture'] = self.binary_texture
    #self.sphere_program['u_texture'].wrapping = 'repeat'

    self.texture_start_coords = Geometry.cen2tri(0, 0, .1)

    self.sphere_program['a_texcoord'] = gloo.VertexBuffer(self.texture_coords)
