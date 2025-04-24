"""A global flow stimulus on spherical surface
Original author: Yue Zhang
"""
import numpy as np
from vispy import gloo

from vxpy.core import visual as vxvisual
from vxpy.utils import sphere, geometry as Geometry
import vxpy.core.attribute as vxattribute



class GlobalOpticFlow(vxvisual.SphericalVisual):
    """Visual stimulus that simulates combined global optic flow for translation and rotation movement
    along and around two different axes (originally created by Yue Zhang)
    """
    time = vxvisual.FloatParameter('time', internal=True)
    p_trans_azi = vxvisual.FloatParameter('p_trans_azi', default=0, limits=(-180, +180), step_size=1)
    p_trans_elv = vxvisual.FloatParameter('p_trans_elv', default=0, limits=(-90, +90), step_size=1)
    p_trans_speed = vxvisual.FloatParameter('p_trans_speed', default=0, limits=(-100, +100), step_size=1)
    p_rot_azi = vxvisual.FloatParameter('p_rot_azi', default=0, limits=(-180, +180), step_size=1)
    p_rot_elv = vxvisual.FloatParameter('p_rot_elv', default=90, limits=(-90, +90), step_size=1)
    p_rot_speed = vxvisual.FloatParameter('p_rot_speed', default=0, limits=(-100, +100), step_size=1)
    p_tex_scale = vxvisual.FloatParameter('p_tex_scale', default=0, limits=(-2, +2), step_size=0.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        vert = self.load_vertex_shader('./position_transform.vert')
        frag = self.load_shader('./texture_map.frag')
        self.sphere_program = gloo.Program(vert, frag)

        texture = np.uint8(np.random.randint(0, 2, [100, 100, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.sphere_program['u_texture'] = texture
        self.sphere_program['u_texture'].wrapping = "repeat"

        # Set up sphere
        self.sphere_model = sphere.CMNIcoSphere(subdivisionTimes=2)
        self.index_buffer = gloo.IndexBuffer(self.sphere_model.indices)
        self.position_buffer = gloo.VertexBuffer(np.float32(self.sphere_model.a_position))
        Isize = self.sphere_model.indices.size

        self.tile_center_q = Geometry.qn(self.sphere_model.tile_center)
        self.tile_hori_dir = Geometry.qn(np.real(self.sphere_model.tile_orientation)).normalize[:, None]
        self.tile_vert_dir = Geometry.qn(np.imag(self.sphere_model.tile_orientation)).normalize[:, None]
        startpoint = Geometry.cen2tri(np.random.rand(Isize // 3), np.random.rand(Isize // 3), .1)
        self.motmat = None

        self._texcoord = np.float32(startpoint.reshape([-1, 2]))
        self.sphere_program['a_position'] = self.position_buffer
        self.sphere_program['a_texcoord'] = self._texcoord

    def initialize(self):
        #self.p_rot_elv.data[0] = 90
        pass
    def gen_motmat(self):
        center = self.tile_center_q[:, None]

        trans_direction = Geometry.qn([self.p_trans_azi.data[0] / 180 * np.pi, self.p_trans_elv.data[0] / 180 * np.pi])
        trans_motmat = Geometry.qcross(center, trans_direction) / 30000 * self.p_trans_speed.data[0]

        rot_direction = Geometry.qn([self.p_rot_azi.data[0] / 180 * np.pi, self.p_rot_elv.data[0] / 180 * np.pi])
        rot_motmat = Geometry.projection(center, rot_direction) / 30000 * self.p_rot_speed.data[0]
        motion_matrix = trans_motmat + rot_motmat
        self.motmat = np.repeat(Geometry.qdot(self.tile_hori_dir, motion_matrix) - 1.j
                                * Geometry.qdot(self.tile_vert_dir, motion_matrix), 3, axis=0)

    def render(self, dt):
        speed_ms = vxattribute.read_attribute('translational_speed')[2]
        speed_ms = speed_ms[0, 0]
        rot_rads = vxattribute.read_attribute('angular_speed')[2] * 180/np.pi
        rot_rads = rot_rads[0,0]
        #print('cms', speed_cms)
        #print('rot ',rot_rads)
        self.p_trans_speed.data[0] = speed_ms * 3000 # to convert it to mm/s this gave a more accurate speed
        # change the sign
        self.p_rot_speed.data[0] = - rot_rads


        self.gen_motmat()
        self._texcoord += np.array([np.real(self.motmat), np.imag(self.motmat)]).T[0, ...]
        self.sphere_program['a_texcoord'] = self._texcoord * 10 ** self.p_tex_scale.data[0]
        self.sphere_program.draw('triangles', self.index_buffer)
