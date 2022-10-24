"""
vxpy ./visuals/spherical/grating.py
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import signal
from vispy import gloo
import vxpy.utils.geometry as Geometry

from vxpy.core import visual
from vxpy.utils import sphere


class SphGlobalFlow(visual.SphericalVisual):

    time = visual.FloatParameter('time', internal=True)
    p_trans_azi = visual.FloatParameter('p_trans_azi', default=0, limits=(-180, +180), step_size=1)
    p_trans_elv = visual.FloatParameter('p_trans_elv', default=0, limits=(-90, +90), step_size=1)
    p_trans_speed = visual.FloatParameter('p_trans_speed', default=0, limits=(-100, +100), step_size=1)
    p_rot_azi = visual.FloatParameter('p_rot_azi', default=0, limits=(-180, +180), step_size=1)
    p_rot_elv = visual.FloatParameter('p_rot_elv', default=0, limits=(-90, +90), step_size=1)
    p_rot_speed = visual.FloatParameter('p_rot_speed', default=0, limits=(-100, +100), step_size=1)
    p_tex_scale = visual.FloatParameter('p_tex_scale', default=0, limits=(-2, +2), step_size=0.01)

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)
        vert = self.load_vertex_shader('./sph_CMN.vert')
        frag = self.load_shader('./sph_CMN.frag')
        self.sphere_program = gloo.Program(vert, frag)
        self.sphere_program['u_texture'] = np.uint8(
            np.random.randint(0, 2, [100, 100, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.sphere_program['u_texture'].wrapping = "repeat"

        # Set up sphere
        self.sphere_model = sphere.CMNIcoSphere(subdivisionTimes=2)
        self.index_buffer = gloo.IndexBuffer(self.sphere_model.indices)
        self.position_buffer = gloo.VertexBuffer(np.float32(self.sphere_model.a_position))
        Isize = self.sphere_model.indices.size

        self.tile_center_q = Geometry.qn(self.sphere_model.tile_center)
        self.tile_hori_dir = Geometry.qn(np.real(self.sphere_model.tile_orientation)).normalize[:, None]
        self.tile_vert_dir = Geometry.qn(np.imag(self.sphere_model.tile_orientation)).normalize[:, None]
        startpoint = Geometry.cen2tri(np.random.rand(np.int(Isize / 3)), np.random.rand(np.int(Isize / 3)), .1)
        self.motmat = None

        self._texcoord = np.float32(startpoint.reshape([-1, 2]))
        self.sphere_program['a_position'] = self.position_buffer
        self.sphere_program['a_texcoord'] = self._texcoord

    def initialize(self):
        self.sphere_model = sphere.CMNIcoSphere(subdivisionTimes=3)
        self.index_buffer = gloo.IndexBuffer(self.sphere_model.indices)
        self.position_buffer = gloo.VertexBuffer(np.float32(self.sphere_model.a_position))
        Isize = self.sphere_model.indices.size

        self.tile_center_q = Geometry.qn(self.sphere_model.tile_center)
        self.tile_hori_dir = Geometry.qn(np.real(self.sphere_model.tile_orientation)).normalize[:, None]
        self.tile_vert_dir = Geometry.qn(np.imag(self.sphere_model.tile_orientation)).normalize[:, None]
        startpoint = Geometry.cen2tri(np.random.rand(np.int(Isize / 3)), np.random.rand(np.int(Isize / 3)), .1)
        self.motmat = None

        self._texcoord = np.float32(startpoint.reshape([-1, 2]))
        self.sphere_program['a_position'] = self.position_buffer
        self.sphere_program['a_texcoord'] = self._texcoord

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
        self.gen_motmat()
        self.apply_transform(self.sphere_program)
        self._texcoord += np.array([np.real(self.motmat), np.imag(self.motmat)]).T[0, ...]
        self.sphere_program['a_texcoord'] = self._texcoord * 10**self.p_tex_scale.data[0]
        self.sphere_program.draw('triangles', self.index_buffer)
        

class SphCMN(visual.SphericalVisual):

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)
        vert = self.load_vertex_shader('./sph_CMN.vert')
        frag = self.load_shader('./sph_CMN.frag')
        self.sphere_program = gloo.Program(vert, frag)
        # Set up sphere
        self.sphere_model = sphere.CMNIcoSphere(subdivisionTimes=2)
        self.index_buffer = gloo.IndexBuffer(self.sphere_model.indices)
        self.position_buffer = gloo.VertexBuffer(np.float32(self.sphere_model.a_position))
        Isize = self.sphere_model.indices.size
        sp_sigma = 1  # spatial CR
        tp_sigma = 20  # temporal CR
        spkernel = np.exp(-(self.sphere_model.intertile_distance ** 2) / (2 * sp_sigma ** 2))
        spkernel *= spkernel > .001
        tp_min_length = np.int (np.ceil (np.sqrt (-2 * tp_sigma ** 2 * np.log (.01 * tp_sigma * np.sqrt (2 * np.pi)))))
        tpkernel = np.linspace (-tp_min_length, tp_min_length, num=2 * tp_min_length + 1)
        tpkernel = 1 / (tp_sigma * np.sqrt (2 * np.pi)) * np.exp (-tpkernel ** 2 / (2 * tp_sigma ** 2))
        tpkernel *= tpkernel > .0001

        flowvec = np.random.normal (size=[np.int (Isize / 3), 500, 3])  # Random white noise motion vector
        flowvec /= Geometry.vecNorm (flowvec)[:, :, None]
        tpsmooth_x = signal.convolve (flowvec[:, :, 0], tpkernel[np.newaxis, :], mode='same')
        tpsmooth_y = signal.convolve (flowvec[:, :, 1], tpkernel[np.newaxis, :], mode='same')
        tpsmooth_z = signal.convolve (flowvec[:, :, 2], tpkernel[np.newaxis, :], mode='same')
        spsmooth_x = np.dot (spkernel, tpsmooth_x)
        spsmooth_y = np.dot (spkernel, tpsmooth_y)
        spsmooth_z = np.dot (spkernel, tpsmooth_z)  #
        spsmooth_Q = Geometry.qn(np.array([spsmooth_x, spsmooth_y, spsmooth_z]).transpose ([1, 2, 0]))

        tileCen_Q = Geometry.qn (self.sphere_model.tile_center)
        tileOri_Q1 = Geometry.qn (np.real (self.sphere_model.tile_orientation)).normalize[:, None]
        tileOri_Q2 = Geometry.qn (np.imag (self.sphere_model.tile_orientation)).normalize[:, None]
        projected_motmat = Geometry.projection (tileCen_Q[:, None], spsmooth_Q)
        self.motmatFull = Geometry.qdot(tileOri_Q1, projected_motmat) - 1.j * Geometry.qdot (tileOri_Q2, projected_motmat)
        startpoint = Geometry.cen2tri(np.random.rand (np.int (Isize / 3)), np.random.rand (np.int (Isize / 3)), .1)
        self._texcoord = np.float32(startpoint.reshape([-1, 2]) / 2)
        self.sphere_program['a_position'] = self.position_buffer
        self.sphere_program['a_texcoord'] = gloo.VertexBuffer(self._texcoord)
        self.sphere_program['u_texture']= np.uint8(np.random.randint(0, 2, [100, 100, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.sphere_program['u_texture'].wrapping = "repeat"
        self.i = 0

    def initialize(self, **params):
        pass

    def render(self, dt):
        tidx = np.mod(self.i,499)
        motmat = np.repeat(self.motmatFull[:,tidx],3,axis = 0)
        self.apply_transform(self.sphere_program)
        self._texcoord += np.array([np.real(motmat), np.imag(motmat)]).T / 1000
        self.sphere_program['a_texcoord'] = self._texcoord
        self.i += 1
        self.sphere_program.draw('triangles', self.index_buffer)
