"""Different flavors of visual spherical grating stimuli
"""
from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere


class RotatingSphericalGrating(vxvisual.SphericalVisual):
    """Black und white contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'A rotating spherical grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    angular_velocity = vxvisual.FloatParameter('angular_velocity', static=True)
    angular_period = vxvisual.FloatParameter('angular_period', static=True)

    # Paths to shaders
    VERT_PATH = './spherical_grating.vert'
    FRAG_PATH = './spherical_grating.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere()
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)

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

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)


class MotionAxis(vxvisual.Mat4Parameter):
    """Example for a custom mapping with different methods, based on input data to the parameter"""
    def __init__(self, *args, **kwargs):
        vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)

        self.value_map = {'forward': self._rotate_forward,
                          'sideways': self._rotate_sideways,
                          'vertical': self._rotate_vertical()}

    @staticmethod
    def _rotate_forward():
        return transforms.rotate(90, (0, 1, 0))

    @staticmethod
    def _rotate_sideways():
        return transforms.rotate(90, (1, 0, 0))

    @staticmethod
    def _rotate_vertical():
        return transforms.rotate(180, (0, 0, 1))


class SphericalBlackWhiteGrating(vxvisual.SphericalVisual):
    """Black und white contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'Spherical black und white contrast grating stimulus'

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    waveform = vxvisual.IntParameter('waveform', value_map={'rectangular': 1, 'sinusoidal': 2}, static=True)
    motion_type = vxvisual.IntParameter('motion_type', static=True, value_map = {'translation': 1, 'rotation': 2})
    motion_axis = MotionAxis('motion_axis', static=True)
    angular_velocity = vxvisual.FloatParameter('angular_velocity', default=30, limits=(-180, 180), step_size=5, static=True)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=5, static=True)

    # Paths to shaders
    VERT_PATH = './spherical_grating.vert'
    FRAG_PATH = './spherical_grating.frag'

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


class RGB01(vxvisual.Vec3Parameter):

    def __init__(self, *args, **kwargs):
        vxvisual.Vec3Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        self.data = [SphericalColorContrastGrating.red01.data[0],
                     SphericalColorContrastGrating.green01.data[0],
                     SphericalColorContrastGrating.blue01.data[0]]


class RGB02(vxvisual.Vec3Parameter):

    def __init__(self, *args, **kwargs):
        vxvisual.Vec3Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        self.data = [SphericalColorContrastGrating.red02.data[0],
                     SphericalColorContrastGrating.green02.data[0],
                     SphericalColorContrastGrating.blue02.data[0]]


class SphericalColorContrastGrating(SphericalBlackWhiteGrating):
    """Color contrast grating stimulus on a sphere
    """
    # (optional) Add a short description
    description = 'Spherical color contrast grating stimulus'

    # Define parameters
    rgb01 = RGB01('rgb01', static=True, internal=True)
    red01 = vxvisual.FloatParameter('red01', static=True, default=1.0, limits=(0.0, 1.0), step_size=0.01)
    green01 = vxvisual.FloatParameter('green01', static=True, default=0.0, limits=(0.0, 1.0), step_size=0.01)
    blue01 = vxvisual.FloatParameter('blue01', static=True, default=0.0, limits=(0.0, 1.0), step_size=0.01)
    rgb02 = RGB02('rgb02', static=True, internal=True)
    red02 = vxvisual.FloatParameter('red02', static=True, default=0.0, limits=(0.0, 1.0), step_size=0.01)
    green02 = vxvisual.FloatParameter('green02', static=True, default=1.0, limits=(0.0, 1.0), step_size=0.01)
    blue02 = vxvisual.FloatParameter('blue02', static=True, default=0.0, limits=(0.0, 1.0), step_size=0.01)

    # Paths to shaders
    FRAG_PATH = './spherical_color_contrast_grating.frag'

    def __init__(self, *args, **kwargs):
        SphericalBlackWhiteGrating.__init__(self, *args, **kwargs)

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.red01.add_downstream_link(self.rgb01)
        self.green01.add_downstream_link(self.rgb01)
        self.blue01.add_downstream_link(self.rgb01)
        self.red02.add_downstream_link(self.rgb02)
        self.green02.add_downstream_link(self.rgb02)
        self.blue02.add_downstream_link(self.rgb02)

        self.rgb01.connect(self.grating)
        self.rgb02.connect(self.grating)

    def initialize(self, **params):
        SphericalBlackWhiteGrating.initialize(self, **params)

