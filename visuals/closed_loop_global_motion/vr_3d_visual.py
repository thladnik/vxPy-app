"""Demo VR environment
"""
from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
import vxpy.core.attribute as vxattribute
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


# LOOK AT THIS TUTORIAL https://docs.panda3d.org/1.10/python/introduction/tutorial/index


class VRVisual(vxvisual.SphericalVisual):
    """Virtual reality demo of fish in aquatic environment
    """

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    orientation = vxvisual.FloatParameter('orientation', internal=True)
    position = vxvisual.Vec2Parameter('position', internal=True)

    vertices = models.UnitSphere.vertices()
    indices = models.UnitSphere.indices()

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.fb_size = 512

        self.pos_x = multiprocessing.Value(ctypes.c_float)
        self.pos_y = multiprocessing.Value(ctypes.c_float)
        self.heading = multiprocessing.Value(ctypes.c_float)
        self.raw_frame_data = multiprocessing.Array(ctypes.c_uint8, int(6 * self.fb_size * self.fb_size * 4))
        self.manager = multiprocessing.Manager()

        # this is the vector containing the virtual movement of the fish. The first component is translational speed
        # the second component is the angular speed
        self.fish_motion_vec = self.manager.list([0, 0, 0])
        self.frame_data = np.frombuffer(self.raw_frame_data.get_obj(), dtype=np.uint8).reshape((6, self.fb_size, self.fb_size, 4))

        self.child = multiprocessing.Process(target=cubemap_gen.run,
                                             args=(self.raw_frame_data, self.fish_motion_vec, self.pos_x, self.pos_y, self.heading, self.fb_size))
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

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        self.position.data = np.array([0,0])
        # vxattribute.ArrayAttribute('fish_motion_vec_VR_input', (2,), vxattribute.ArrayType.float64)
        # vxattribute.write_to_file(self, 'fish_motion_vec_VR_input')

    def render(self, dt):

        # calculate velocity vector of fish:
        # vec2d = vxattribute.read_attribute(...)
        trans_speed = vxattribute.read_attribute('translational_speed')[2][0]
        ang_speed = vxattribute.read_attribute('angular_speed')[2][0]
        vec2d = np.array([trans_speed,ang_speed])



        self.fish_motion_vec[:] = vec2d #np.array(vec2d[0],vec2d[1],0]) # 2d vector


        #vxattribute.write_attribute('fish_motion_vec_VR_input', vec2d)


        # Add elapsed time to u_time
        self.time.data += dt
        # #self.position.data = np.array([self.pos_x.value, self.pos_y.value])
        # self.pos_x.value += vec2d[0] * dt * 1
        # self.pos_y.value += vec2d[1] * dt
        self.position.data = np.array([self.pos_x.value , self.pos_y.value])
        #print(f'Position: {self.position.data}, vec 0: {vec2d[0] }, dt: {dt}, x val: {self.pos_x.value}')

        #self.position.data += np.array([0.01,0.01])

        self.orientation.data = self.heading.value

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.program)

        self.texture.set_data(self.frame_data)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.program.draw('triangles', self.index_buffer)