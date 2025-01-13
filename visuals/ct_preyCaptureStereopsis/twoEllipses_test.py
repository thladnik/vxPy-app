from vispy import gloo
import numpy as np

from vxpy.core import visual
from vxpy.utils import sphere


class TwoEllipsesRotatingAroundAxis(visual.SphericalVisual):

    # Define parameters
    time = visual.FloatParameter('time', internal=True)
    polarity = visual.IntParameter('polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    pos_azimut_angle = visual.FloatParameter('pos_azimut_angle', default=45, limits=(-180, 180), step_size=5, static=True)
    pos_elev_angle = visual.FloatParameter('pos_elev_angle', default=-35, limits=(-180, 180), step_size=5, static=True)
    movement_major_axis = visual.FloatParameter('movement_major_axis', default=0, limits=(0, 180), step_size=1, static=True)
    movement_minor_axis = visual.FloatParameter('movement_minor_axis', default=0, limits=(0, 180), step_size=1, static=True)
    movement_angular_velocity = visual.FloatParameter('movement_angular_velocity', default=0, limits=(-360, 360), step_size=5, static=True)
    el_diameter_horiz = visual.FloatParameter('el_diameter_horiz', default=5, limits=(1, 180), step_size=1, static=True)  # in deg
    el_diameter_vert = visual.FloatParameter('el_diameter_vert', default=2, limits=(1, 180), step_size=1, static=True)   # in deg
    el_rotating = visual.IntParameter('el_rotating', value_map={'rotating-yes': 1, 'rotating-no': 0}, static=True)
    el_mirror = visual.IntParameter('el_mirror', value_map={'mirror-dot-no': 0, 'mirror-dot-yes': 1}, static=True)
    el_color_r = visual.FloatParameter('el_color_r', default=1.0, limits=(0, 1), step_size=0.1, static=True)
    el_color_g = visual.FloatParameter('el_color_g', default=1.0, limits=(0, 1), step_size=0.1, static=True)
    el_color_b = visual.FloatParameter('el_color_b', default=1.0, limits=(0, 1), step_size=0.1, static=True)
    pos_azimut_offset = visual.FloatParameter('pos_azimut_offset', default=90, limits=(-180, 180), step_size=1, static=True)

    # Paths to shaders
    VERT_PATH = './vertexShader.vert'
    FRAG_PATH = './fragmentShader_ellipses.frag'

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi/2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)
        self.polarity.connect(self.rotating_dot)
        self.pos_azimut_angle.connect(self.rotating_dot)
        self.pos_elev_angle.connect(self.rotating_dot)
        self.movement_major_axis.connect(self.rotating_dot)
        self.movement_minor_axis.connect(self.rotating_dot)
        self.movement_angular_velocity.connect(self.rotating_dot)
        self.el_diameter_horiz.connect(self.rotating_dot)
        self.el_diameter_vert.connect(self.rotating_dot)
        self.el_rotating.connect(self.rotating_dot)
        self.pos_azimut_offset.connect(self.rotating_dot)
        self.el_mirror.connect(self.rotating_dot)
        self.el_color_r.connect(self.rotating_dot)
        self.el_color_g.connect(self.rotating_dot)
        self.el_color_b.connect(self.rotating_dot)
        #self.test.connect(self.rotating_dot)

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

    def do_updates(self):
        pass

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        self.do_updates()

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)
