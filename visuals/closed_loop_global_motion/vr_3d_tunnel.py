"""This script projects gratings in a speed
such that the speed at the equator of the arena matches the fictive swimming velocity of the fish"""


from vispy import gloo
from vispy.util import transforms
import numpy as np

import vxpy.core.visual as vxvisual
from vxpy.utils import sphere

# allow the script to get the attributes set in tracking routine
import vxpy.core.attribute as vxattribute


class TranslationMotionAxis(vxvisual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        elevation = ClosedLoopTranslationGrating.elevation.data[0]
        azimuth = ClosedLoopTranslationGrating.azimuth.data[0]

        rot_elevation = transforms.rotate(90. - elevation, (0, 1, 0))
        rot_azimuth = transforms.rotate(azimuth, (0, 0, -1))
        self.data = np.dot(rot_elevation, rot_azimuth)


class VRTunnel(vxvisual.SphericalVisual):
    # (optional) Add a short description
    description = ''

    # Define parameters
    time = vxvisual.FloatParameter('time', internal=True)


    fish_radial_position = vxvisual.FloatParameter('fish_radial_position', internal=True)
    fish_axial_position = vxvisual.FloatParameter('fish_axial_position', internal=True)
    fish_orientation = vxvisual.FloatParameter('fish_orientation', internal=True)

    elevation = vxvisual.FloatParameter('elevation', default=00, limits=(-90, 90), step_size=1, static=True)
    azimuth = vxvisual.FloatParameter('azimuth', default=00, limits=(-180, 180), step_size=1, static=True)

    motion_axis = TranslationMotionAxis('motion_axis', static=True, internal=True)

    angular_period = vxvisual.FloatParameter('angular_period', default=45, limits=(5, 360), step_size=1, static=True)


    fish_vel_gain = vxvisual.FloatParameter('fish_vel_gain', default=1, limits=(0, 5), step_size=0.001)


    # this is the fictive velocity of the fish if it were currently stationary
    ext_vel_middle = vxvisual.FloatParameter('ext_vel_middle', default= -1, limits=(-40, 40), step_size=0.01)

    # Paths to shaders
    VERT_PATH = 'gratings/sphere.vert'
    FRAG_PATH = 'translation_grating.frag'

    # a vector representing the current position in the tube. First component is position along radius (positive is to
    # the right).
    # Second axial position (in units of period of gratings). Third is orientation of the fish (unit is rad, 0 is forward and
    # positive is CCW
    fish_position_vector = np.array([0,0,0])


    # characteristics of the tunenl
    radius = 5 # cm
    grating_size = 2 # cm

    # the maximum radial position of the fish.
    max_fish_rad_pos = radius - 0.1 # fish can get up to 1 mm to border




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
        self.fish_radial_position.connect(self.grating)
        self.fish_axial_position.connect(self.grating)
        self.fish_orientation.connect(self.grating)
        self.motion_axis.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.ext_vel_middle.connect(self.grating)
        self.fish_vel_gain.connect(self.grating)

        # ?? connect time
        self.time.connect(self.grating)

        # Link motion axis to be updated when elevation or azimuth changes
        self.elevation.add_downstream_link(self.motion_axis)
        self.azimuth.add_downstream_link(self.motion_axis)



    def initialize(self, **params):
        self.fish_radial_position.data = 0
        self.fish_axial_position.data = 0
        self.fish_orientation.data = 0

        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0


        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer


    def radial_position_to_external_velocity(self,ext_vel_middle):

        return ext_vel_middle * (1 - self.fish_position_vector[0]**2 / self.radius ** 2)


    def update_fish_position(self,axial_speed,ang_speed,ext_speed):
        pass

    def apply_bounds_to_radial_position(self,radial_pos):

        return np.minimum(self.max_fish_rad_pos, np.maximum(radial_pos,- self.max_fish_rad_pos))


    def render(self, dt):

        # 1) get fictive swimming characteristics of fish

        # get the fictive translational speed of the fish in cm/s
        speed_cms_est = vxattribute.read_attribute('translational_speed')[2] * 100
        self_trans_velocity = speed_cms_est[0,0] * self.fish_vel_gain.data[0]

        speed_cms_est = vxattribute.read_attribute('angular_speed')[2] * 100
        self_ang_velocity = speed_cms_est[0, 0] * self.fish_vel_gain.data[0]

        # 2) get the external velocity
        ext_vel_middle = self.ext_vel_middle.data[0]

        # the external velocity
        external_velocity = self.radial_position_to_external_velocity(ext_vel_middle)

        # 3) update the orientation of the fish by integrating the angular speed
        self.fish_orientation.data += self_ang_velocity * dt # this is radians

        # 4) update the position in the tunnel

        # add the effect of the external velocity: no effect on radial position but on axial position. It
        # decreases it because the water is flowing from the front to the back
        self.fish_axial_position.data -= external_velocity * dt

        # add effect of self translational self speed. Positive angles are CCW movement and zero degrees is parallel
        # to the tunnel walls. Negative radius values are to the left.
        fish_ori = self.fish_orientation.data[0]
        radial_pos = self.fish_radial_position.data[0] - np.sin(fish_ori) * self_trans_velocity * dt
        self.fish_axial_position.data += np.cos(fish_ori) * self_trans_velocity * dt

        # make sure fish does not come too close to tunnel boundary
        self.fish_radial_position.data = self.apply_bounds_to_radial_position(radial_pos)



        # set azimuth such that it counteracts the orientation of the fish.

        # Add elapsed time to u_time
        self.time.data += dt


        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
