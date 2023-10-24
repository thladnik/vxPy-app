import time

import numpy as np
import scipy
from scipy.spatial import ConvexHull
from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate

import vxpy.core.visual as vxvisual

vertex = """
uniform int u_lines;
uniform mat4 m_transform;

attribute vec3 a_position;
attribute float a_color;

varying float v_color;

void main(void) {
    
    vec4 pos = vec4(a_position, 1.0);
    
    if(u_lines == 1) {
        pos = pos * 1.2;
    }
    
    gl_Position = m_transform * pos;

    v_color = a_color;

}
"""

fragment = """
uniform int u_lines;

varying float v_color;

void main(void) {

    if(u_lines == 1) {
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);            
    } else {
        gl_FragColor = vec4(vec3(step(v_color, 0.5)), 1.0);
    }
}
"""


class BinaryBlackWhiteJitterNoise(vxvisual.SphericalVisual):

    time = 0.0  # s
    jitter_update_rate = 2.0  # Hz
    last_jitter_update = 0.0  # s
    pattern_update_rate = 2.0  # Hz
    last_pattern_update = 0.0  # s

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Model
        self.vertices = np.ascontiguousarray(np.array(scipy.io.loadmat('./visuals/jitter_noise/repulsive_sphere_640.mat')['ans'][0, 0], dtype=np.float32))
        self.indices = np.ascontiguousarray(self.create_simplices(self.vertices).flatten())
        self.indices_buffer = IndexBuffer(self.indices)
        self.vertices_buffer = VertexBuffer(self.vertices)

        # Program
        self.program = Program(self.parse_vertex_shader(vertex), fragment)
        self.program.bind(self.vertices_buffer)
        self.program['a_position'] = self.vertices_buffer
        self.program['m_transform'] = np.eye(4)

        # Set initial pattern
        self.program['a_color'] = self.generate_pattern()

        self.start_time = time.perf_counter()

    def initialize(self, **kwargs):
        self.time = 0.0

    def create_simplices(self, verts):

        hull = ConvexHull(verts)
        indices = hull.simplices

        return indices.astype(np.uint32)

    def generate_pattern(self):

        nf = self.vertices_buffer.size
        pattern = np.random.randint(2, size=(nf,))
        pattern = np.ascontiguousarray(pattern, dtype=np.float32)

        return pattern

    def generate_rotation(self):

        angle = np.random.randint(10) * 0.5

        v3 = np.random.rand(3)
        v3 /= np.linalg.norm(v3)

        return rotate(angle, v3)

    def update_visual(self):

        if self.last_pattern_update + 1/self.pattern_update_rate < self.time:
            self.program['a_color'] = self.generate_pattern()
            self.last_pattern_update = self.time

        if self.last_jitter_update + 1/self.jitter_update_rate < self.time:
            self.program['m_transform'] = self.generate_rotation()
            self.last_jitter_update = self.time

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def render(self, dt):

        self.time += dt

        self.update_visual()

        self.program['u_lines'] = 0
        self.program.draw('triangles', self.indices_buffer)
        self.program['u_lines'] = 1
        self.program.draw('lines', self.indices_buffer)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import math

    # function for finding the angle
    def angle_triangle(x1, x2, x3, y1, y2, y3, z1, z2, z3):

        num = (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) + (z2 - z1) * (z3 - z1)

        den = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) * \
              math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2)

        angle = math.degrees(math.acos(num / den))

        return round(angle, 3)

    c = BinaryBlackWhiteJitterNoise()
    verts = c.vertices
    indices = c.create_simplices(verts)

    angles = []
    for i, j, k in indices:
        v1 = verts[i]
        v2 = verts[j]
        v3 = verts[k]

        angles.append([
            angle_triangle(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]),
            angle_triangle(v2[0], v3[0], v1[0], v2[1], v3[1], v1[1], v2[2], v3[2], v1[2]),
            angle_triangle(v3[0], v2[0], v1[0], v3[1], v2[1], v1[1], v3[2], v2[2], v1[2]),
        ])

    angles = np.array(angles)
    test = angles.sum(axis=1)
    print(test.shape)
    plt.plot(test, "o")
    plt.show()
