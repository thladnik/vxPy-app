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

from vxpy.core import visual, logger
from vxpy.utils import sphere
from vxpy.utils import geometry

log = logger.getLogger(__name__)


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

        # Fetch available indices (up to elevation of +45deg)
        elev_max = np.pi/4
        v_max = geometry.sph2cart(0., elev_max, 1.)
        self.avail_vertex_idcs = np.where(self.vertices[:,-1] < v_max[-1])[0]
        np.random.seed(1)
        self.avail_vertex_idcs = np.random.permutation(self.avail_vertex_idcs)
        log.info('Using {len(self.avail_vertex_idcs)} vertices '
                      + 'up to elevation of {:.1f}'.format(180 / np.pi * elev_max))
        self.baseline_level = 0.0
        self.group_size = 5
        self.idx = None
        self.tau = 2
        self.end_on_next = False

    def initialize(self, **params):
        self.dot['u_time'] = 0.0
        self.idx = 0
        self.states[:] = self.baseline_level
        self.state_buffer[:] = self.states
        self.end_on_next = False

    def render(self, dt):
        self.dot['u_time'] += dt

        interval = self.parameters.get(self.p_interval) / 1000

        if interval is None:
            return

        cur_time = self.dot['u_time'][0]
        self.states = self.states[:] - dt * 1./self.tau * (self.states[:] - self.baseline_level)
        self.state_buffer[:] = self.states

        # Write to parameters
        self.parameters.update(decay_tau=self.tau)

        if np.floor(cur_time/interval) > self.idx / self.group_size:

            choose_random = False

            if not choose_random:

                max_idx = self.avail_vertex_idcs.shape[0]
                end_idx = self.idx + self.group_size

                if end_idx > max_idx:
                    if self.end_on_next:
                        self.end()
                    else:
                        end_idx = max_idx
                        self.end_on_next = True

                idcs = (self.avail_vertex_idcs[self.idx:end_idx]).astype(np.int64)

                lendiff = self.group_size - (end_idx-self.idx)
                if lendiff > 0:
                    idcs = np.append(idcs, self.avail_vertex_idcs[:lendiff].astype(np.int64))

                self.states[idcs] = 1.
                self.state_buffer[:] = self.states

                # Write to parameters
                self.parameters.update(active_vertex_indices=idcs, active_vertices=self.vertices[idcs])

            self.idx += self.group_size

        # Draw dots
        self.apply_transform(self.dot)
        # self.dot.draw('triangles', self.index_buffer)
        self.dot.draw('points')

    interface = [
        (p_interval, 1000, 100, 5000, dict(step_size=100)),
        ('Reinitialize', initialize)
    ]
