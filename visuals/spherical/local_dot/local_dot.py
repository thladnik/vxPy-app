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
from vispy import gloo
from vispy import scene

from vxpy.api import camera_rpc
from vxpy.core import visual
from vxpy.utils import sphere, geometry
from vxpy.routines.camera.zf_tracking import EyePositionDetection


class IcoDot(visual.SphericalVisual):

    p_interval = 'p_interval'

    def __init__(self, *args):
        visual.SphericalVisual.__init__(self, *args)

        # Set up sphere
        self.ico_sphere = sphere.IcosahedronSphere(subdiv_lvl=3)
        self.index_buffer = gloo.IndexBuffer(self.ico_sphere.get_indices())
        self.vertices = self.ico_sphere.get_vertices()
        self.position_buffer = gloo.VertexBuffer(self.vertices)
        self.states = np.ascontiguousarray(np.ones((self.vertices.shape[0],)), dtype=np.float32)
        self.state_buffer = gloo.VertexBuffer(self.states)

        # Set up programs
        VERT = self.load_vertex_shader('./local_dot.vert')
        FRAG = self.load_shader('./local_dot.frag')
        self.dot = gloo.Program(VERT, FRAG)
        self.dot.bind(self.position_buffer)
        self.dot.bind(self.state_buffer)
        self.dot['a_position'] = self.position_buffer
        self.dot['a_state'] = self.state_buffer

        self.avail_vertex_idcs = np.where(self.vertices[:,-1] < .5)[0]
        np.random.seed(1)
        self.avail_vertex_idcs = np.random.permutation(self.avail_vertex_idcs)
        print(f'Available vertices {len(self.avail_vertex_idcs)}')

        self.baseline_level = 0.05
        self.group_size = 5
        self.idx = None

    def initialize(self, **params):
        self.dot['u_time'] = 0.0
        self.idx = 0
        self.states[:] = self.baseline_level
        self.state_buffer[:] = self.states

    def render(self, dt):
        self.dot['u_time'] += dt

        interval = self.parameters.get(self.p_interval) / 1000

        if interval is None:
            return

        cur_time = self.dot['u_time'][0]
        tau = 2.
        self.states = self.states[:] - dt * 1./tau * (self.states[:] - self.baseline_level)
        self.state_buffer[:] = self.states

        if np.floor(cur_time/interval) > self.idx / self.group_size:

            choose_random = False

            if not choose_random:
                # Loop around
                if self.idx >= self.avail_vertex_idcs.shape[0]:
                    self.idx = 0

                idcs = (self.avail_vertex_idcs[(self.idx):(self.idx+self.group_size)]).astype(np.int64)
                self.states[idcs] = 1.
                self.state_buffer[:] = self.states


            else:
                while True:
                    i = np.random.randint(self.states.shape[0])
                    v = self.vertices[i,:]
                    if v[-1] < .5:
                        print(f'bling {i}')
                        self.states[i] = 1.
                        self.state_buffer[i] = 1.
                        # self.states[:] = .5
                        # self.states[i] = 1.
                        # self.states[i] = np.array([1.], dtype=np.float32)
                        # self.state_buffer.set_subdata(np.array([1.], dtype=np.float32), offset=i)
                        # self.states[:] = .9
                        # self.canvas.update()
                        break
                    else:
                        print(f'no for {i} [{v}]')

            self.idx += self.group_size

        # Draw dots
        self.apply_transform(self.dot)
        self.dot.draw('triangles', self.index_buffer)

    interface = [
        (p_interval, 1000, 100, 5000, dict(step_size=100)),
        ('Reinitialize', initialize)
    ]
