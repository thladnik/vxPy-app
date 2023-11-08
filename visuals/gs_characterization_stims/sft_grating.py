"""Different flavors of visual spherical grating stimuli
"""
from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class MotionAxis(vxvisual.Mat4Parameter):
    """Example for a custom mapping with different methods, based on input data to the parameter"""
    def __init__(self, *args, **kwargs):
        vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)

        self.value_map = {'forward': self._rotate_forward,
                          'sideways': self._rotate_sideways,
                          'vertical': np.eye(4)}

    @staticmethod
    def _rotate_forward():
        return transforms.rotate(90, (0, 1, 0))

    @staticmethod
    def _rotate_sideways():
        return transforms.rotate(90, (1, 0, 0))


class SphericalSFTGrating(vxvisual.SphericalVisual):
    """Black und white contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'Spherical black und white contrast grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    motion_type = vxvisual.IntParameter('motion_type', static=True, value_map = {'translation': 1, 'rotation': 2})
    motion_axis = MotionAxis('motion_axis', static=True)
    angular_velocity = vxvisual.FloatParameter('angular_velocity', default=30, step_size=5, static=True)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=5, static=True)
    offset = vxvisual.FloatParameter('offset', default = 0, limits=(-180,180),step_size=1, static=True)

    # Paths to shaders
    VERT_PATH = 'sft_grating.vert'
    FRAG_PATH = 'sft_grating.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi/2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.waveform.connect(self.grating)
        self.motion_type.connect(self.grating)
        self.motion_axis.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.offset.connect(self.grating)


    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)

