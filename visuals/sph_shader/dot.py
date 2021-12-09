from vispy import gloo

from vxpy.core import visual
from vxpy.utils import sphere

class Blank(visual.SphericalVisual):
    u_color = 'u_color'

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)

        # Set up program
        vert = self.load_vertex_shader('./sphericalShader.vert')
        frag = self.load_shader('./blank.frag')
        self.grating = gloo.Program(vert, frag)
        self.grating['a_position'] = self.position_buffer

    def initialize(self, **params):
        self.grating['u_stime'] = 0.0
        self.update(**params)

    def render(self, dt):
        # self.grating['u_stime'] += dt
        self.apply_transform(self.grating)
        self.grating.draw('triangles', self.index_buffer)


class Dot(visual.SphericalVisual):
    u_ang_size = 'u_ang_size'
    u_period = 'u_period'
    u_elv = 'u_elv'

    interface = [(u_ang_size, 5., 0., 100., {'step_size': 1.}),
                 (u_period, 10., -40., 40., {'step_size': 1.}),
                 (u_elv, 0., -45., 45., {'step_size': 1.})]

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)

        # Set up program
        vert = self.load_vertex_shader('./sphericalShader.vert')
        frag = self.load_shader('./dot.frag')
        self.dot = gloo.Program(vert, frag)
        self.dot['a_position'] = self.position_buffer

    def initialize(self, **params):
        self.dot['u_stime'] = 0.0
        self.update(**params)

    def render(self, dt):
        self.dot['u_stime'] += dt
        self.apply_transform(self.dot)
        self.dot.draw('triangles', self.index_buffer)
