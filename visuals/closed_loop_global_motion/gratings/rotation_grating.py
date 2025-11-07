from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere

# allow the script to get the attributes set in tracking routine
import vxpy.core.attribute as vxattribute



class RotationMotionAxis(vxvisual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        elevation = ClosedLoopRotationGrating.elevation.data[0]
        azimuth = ClosedLoopRotationGrating.azimuth.data[0]

        rot_elevation = transforms.rotate(0. - elevation, (0, 1, 0))
        rot_azimuth = transforms.rotate(azimuth, (0, 0, -1))
        self.data = np.dot(rot_elevation, rot_azimuth)



class ClosedLoopRotationGrating(vxvisual.SphericalVisual):
    # (optional) Add a short description
    description = ''

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)
    fish_rel_position = vxvisual.FloatParameter('fish_rel_position', internal=True)
    elevation = vxvisual.FloatParameter('elevation', default=00, limits=(-90, 90), step_size=1, static=True)
    azimuth = vxvisual.FloatParameter('azimuth', default=00, limits=(-180, 180), step_size=1, static=True)
    motion_axis = RotationMotionAxis('motion_axis', static=True, internal=True)
    motion_type = vxvisual.IntParameter('motion_type')
    external_angular_velocity = vxvisual.FloatParameter('external_angular_velocity', default=-0.5, limits=(-40, 40), step_size=0.1)
    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=1, static=True)
    fish_vel_gain = vxvisual.FloatParameter('fish_vel_gain', default=1, limits=(-10, 10), step_size=0.01)

    # radius of the arena. Needed to calculate the speed at the equator
    shpere_radius = 4


    # Paths to shaders
    VERT_PATH = 'sphere.vert'
    FRAG_PATH = 'rotation_grating.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.fish_rel_position.connect(self.grating)
        self.motion_axis.connect(self.grating)
        self.external_angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.fish_vel_gain.connect(self.grating)

        # ?? connect time
        self.time.connect(self.grating)

        # Link motion axis to be updated when elevation or azimuth changes
        self.elevation.add_downstream_link(self.motion_axis)
        self.azimuth.add_downstream_link(self.motion_axis)



    def initialize(self, **params):
        self.fish_rel_position.data = 0

        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # set motion tyoe to be rotational movememt
        self.motion_type.data = 2

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):

        # samuel_todo: make sure angle is converted correctly to the velocity of gratings on the sphere

        # 1) get instantaneous velocity attribute
        # calculate postion and write in .frag, make it dependent on time, debug,

        # get swim angle in rad/s
        self_ang_vel = vxattribute.read_attribute('angular_speed')[2]
        self_ang_vel = self_ang_vel[0,0]
        # if self_ang_vel > 1e-2:
        #     print(self_ang_vel)

        # get the angular velocity of the fish due to fictive external forces
        external_ang_vel = self.external_angular_velocity.data[0]

        # convert the translation speed into degrees
        gain = self.fish_vel_gain.data[0]

        # Add elapsed time to u_time
        self.time.data += dt

        # sum up angular velocity and convert to deg/s
        # positive self movement of the fish is CCW. the Gratings should then move CC. This is positive movemt of the
        # gratings in their reference frame. Self motion is therefore positive when translated
        total_ang_vel = 180 * (self_ang_vel * gain + external_ang_vel)/np.pi

        # the fish postition relative to the gratings
        self.fish_rel_position.data += dt * total_ang_vel

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)


