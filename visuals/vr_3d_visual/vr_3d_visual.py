"""Demo VR environment
"""
from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
import ctypes
import multiprocessing
import time

import numpy as np
from vispy.util.transforms import perspective, rotate, translate
from vispy import gloo

from .renderer import cubemap_gen
from .renderer import models

VERT_SHADER = """

// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec3 a_position;

// Varyings
varying vec3 v_texcoord;

// Main
void main (void) {

    v_texcoord = a_position;
    gl_Position = transform_position(a_position);
}
"""

FRAG_SHADER = """
#version 130

uniform samplerCube u_texture;

varying vec3 v_texcoord;

void main() {

    vec4 color = texture(u_texture, v_texcoord);
    
    // Switch red and blue, because cubemap channel layout seems to be different
    // between vispy and panda3d, even though both should use rgba
    gl_FragColor.r  = color.b;
    gl_FragColor.ga  = color.ga;
    gl_FragColor.b  = color.r;
    //gl_FragColor.a = 1.0;
}

"""


class VRVisual(vxvisual.SphericalVisual):
    """Black und white contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'A rotating spherical grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)

    vertices = models.UnitSphere.vertices()
    indices = models.UnitSphere.indices()

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.fb_size = 512

        self.raw_frame_data = multiprocessing.Array(ctypes.c_uint8, int(6 * self.fb_size * self.fb_size * 4))
        self.frame_data = np.frombuffer(self.raw_frame_data.get_obj(), dtype=np.uint8).reshape((6, self.fb_size, self.fb_size, 4))

        self.child = multiprocessing.Process(target=cubemap_gen.run, args=(self.raw_frame_data, self.fb_size))
        self.child.start()

        self.program = gloo.Program(self.parse_vertex_shader(VERT_SHADER), FRAG_SHADER)

        # A simple texture quad
        self.data = np.zeros(self.vertices.shape[0], dtype=[('a_position', np.float32, 3)])

        self.index_buffer = gloo.IndexBuffer(self.indices)
        self.data['a_position'] = self.vertices

        self.texture = gloo.TextureCube(shape=(6, self.fb_size, self.fb_size, 4), format='rgba',
                                        interpolation='linear')
        self.program['u_texture'] = self.texture

        self.program.bind(gloo.VertexBuffer(self.data))

        # self.view = np.eye(4, dtype=np.float32)
        # self.model = np.eye(4, dtype=np.float32)
        # self.projection = np.eye(4, dtype=np.float32)

        # self.program['u_model'] = self.model
        # self.program['u_view'] = self.view
        # self.projection = perspective(60.0, 1., 1, 100)
        # self.program['u_projection'] = self.projection

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.program)

        self.texture.set_data(self.frame_data)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.program.draw('triangles', self.index_buffer)