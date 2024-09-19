import vxpy.core.visual as vxvisual

from vispy import gloo
from vispy.util import transforms
import numpy as np

from vxpy.core import visual
from vxpy.utils import sphere


class CMN_foreground(vxvisual.SphericalVisual):
    time = visual.FloatParameter('time', internal=True)

    #dot_polarity = visual.IntParameter('dot_polarity', value_map={'dark-on-light': 1, 'light-on-dark': 2}, static=True)
    #dot_start_angle = visual.FloatParameter('dot_start_angle', default=0, limits=(-180, 180), step_size=5, static=True)
    #dot_angular_velocity = visual.FloatParameter('dot_angular_velocity', default=60, limits=(-360, 360), step_size=5,static=True)
    #dot_angular_diameter = visual.FloatParameter('dot_angular_diameter', default=10, limits=(1, 90), step_size=1,static=True)
    #dot_offset_angle = visual.FloatParameter('dot_offset_angle', default=0, limits=(-85, 85), step_size=5, static=True)

    # Paths to shaders
    VERT_PATH = './position.vert'
    FRAG_PATH = './texture.frag'

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
       #self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.rotating_dot)

        #self.dot_polarity.connect(self.rotating_dot)
        #self.dot_start_angle.connect(self.rotating_dot)
        #self.dot_angular_velocity.connect(self.rotating_dot)
        #self.dot_angular_diameter.connect(self.rotating_dot)
        #self.dot_offset_angle.connect(self.rotating_dot)

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
        #self.rotating_dot['a_elevation'] = self.elevation_buffer



    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt


        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)