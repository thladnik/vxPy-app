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


class RotatingGrating(vxvisual.SphericalVisual):

    # Paths to shaders
    VERT_PATH = './ml_rotating_grating.vert'
    FRAG_PATH = './ml_rotating_grating.frag'

    # Define general parameters
    time = vxvisual.FloatParameter('time', internal=True)

    # define texture displacement parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    motion_type = vxvisual.IntParameter('motion_type', static=True, value_map = {'translation': 1, 'rotation': 2})
    motion_axis = MotionAxis('motion_axis', static=True)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=5, static=True)
    rotation_start_time = vxvisual.IntParameter('rotation_start_time', static=True, default=1500, limits=(0.0, 5000),
                                                step_size=100)  # ms
    rotation_duration = vxvisual.IntParameter('rotation_duration', static=True, default=10000, limits=(20, 200),
                                              step_size=10)  # ms
    rotation_target_angle = vxvisual.FloatParameter('rotation_target_angle', static=True, default=300.,
                                                    limits=(-90.0, 90.0), step_size=0.01)  # deg



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
        self.rotation.connect(self.grating)
        self.waveform.connect(self.grating)
        self.motion_type.connect(self.grating)
        self.motion_axis.connect(self.grating)
        self.angular_period.connect(self.grating)

        # Global azimuth angle
        self.protocol.global_visual_props['azim_angle'] = 0.

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

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        rot_start_time = self.rotation_start_time.data[0]
        rot_duration = self.rotation_duration.data[0]
        rot_target_angle = self.rotation_target_angle.data[0]

        time_in_rotation = time - rot_start_time
        if 0.0 < time_in_rotation <= rot_duration:
            # Calculate rotation
            angle = rot_target_angle * 1000 * dt / rot_duration
            self.protocol.global_visual_props['azim_angle'] += angle

            # Set rotation
            self.rotation.data = transforms.rotate(self.protocol.global_visual_props['azim_angle'], (0, 0, 1))

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.apply_transform(self.grating)
        self.grating.draw('triangles', indices=self.index_buffer)


