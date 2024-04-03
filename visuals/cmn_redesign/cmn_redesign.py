import os
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import scipy
import vispy
from scipy import signal
from vispy import gloo
from vispy import app
import numpy as np
from vispy.util import transforms
import quaternionic as qt


def sph2cart1(theta, phi, r):
    rcos_phi = r * np.cos(phi)
    x = np.sin(theta) * rcos_phi
    y = np.cos(theta) * rcos_phi
    z = r * np.sin(phi)
    return np.array([x, y, z])


def cart2sph1(cx, cy, cz):
    cxy = cx + cy * 1.j
    azi = np.angle(cxy)
    elv = np.angle(np.abs(cxy)+cz * 1.j)
    return azi, elv


def crossproduct(v1, v2):
    return np.cross(v1 / np.linalg.norm(v1), v2)


class IcosahedronSphere:
    gr = 1.61803398874989484820

    corners = [
        [-1, gr,  0],
        [1,  gr,  0],
        [-1, -gr, 0],
        [1,  -gr, 0],
        [0,  -1,  gr],
        [0,  1,   gr],
        [0,  -1,  -gr],
        [0,  1,   -gr],
        [gr, 0,   -1],
        [gr, 0,   1],
        [-gr, 0,  -1],
        [-gr, 0,  1],
    ]

    _faces = [

        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],

        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],

        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],

        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def __init__(self, subdiv_lvl, **kwargs):

        # Calculate initial vertices
        self._vertices = [self._vertex(*v) for v in self.corners]
        self._vertex_lvls = [0] * len(self.corners)

        # Subdivide faces
        self._cache = None
        self.subdiv_lvl = subdiv_lvl
        self._subdivide()

    def get_indices(self):
        return np.ascontiguousarray(np.array(self._faces, dtype=np.uint32)).flatten()

    def get_vertices(self):
        return np.ascontiguousarray(np.array(self._vertices, dtype=np.float32))

    def get_vertex_levels(self):
        return np.ascontiguousarray(np.array(self._vertex_lvls, dtype=np.int32))

    def get_spherical_coordinates(self):
        _vertices = self.get_vertices()
        az, el = np.array(cart2sph1(_vertices[0, :], _vertices[1, :], _vertices[2, :]))
        return np.ascontiguousarray(az), np.ascontiguousarray(el)

    def _vertex(self, x, y, z):
        vlen = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return [i/vlen for i in (x, y, z)]

    def _midpoint(self, p1, p2, v_lvl):
        key = '%i/%i' % (min(p1, p2), max(p1, p2))

        if key in self._cache:
            return self._cache[key]

        v1 = self._vertices[p1]
        v2 = self._vertices[p2]
        middle = [sum(i)/2 for i in zip(v1, v2)]

        self._vertices.append(self._vertex(*middle))
        self._vertex_lvls.append(v_lvl)
        index = len(self._vertices) - 1

        self._cache[key] = index

        return index

    def _subdivide(self):
        self._cache = {}
        for i in range(self.subdiv_lvl):
            new_faces = []
            for face in self._faces:
                v = [self._midpoint(face[0], face[1], i+1),
                     self._midpoint(face[1], face[2], i+1),
                     self._midpoint(face[2], face[0], i+1)]

                new_faces.append([face[0], v[0], v[2]])
                new_faces.append([face[1], v[1], v[0]])
                new_faces.append([face[2], v[2], v[1]])
                new_faces.append([v[0], v[1], v[2]])

            self._faces = new_faces


def create_motion_matrix(positions: np.ndarray,
                         frame_num: int,  # [frames]
                         tp_sigma: int = 20,  # [frames]
                         sp_sigma: float = 30  # [deg]
                         ) -> np.ndarray:
    """Create a motion matrix of given parameters
    """

    # Create flow vectors for each face and frame and normalize them
    flow_vec = np.random.normal(size=(positions.shape[0], frame_num, 3))  # Random white noise motion vector
    flow_vec /= np.linalg.norm(flow_vec, axis=-1)[:, :, None]

    # Calculate euclidean position/position distances and convert from chord to angle
    # see https://en.wikipedia.org/wiki/Chord_(geometry)#In_trigonometry
    eucl_dists = scipy.spatial.distance.cdist(positions, positions)
    eucl_dists[eucl_dists > 2.] = 2. # Fix rounding errors (no def. for arcsin(x) if x > 1)
    interpos_distances = 2 * np.arcsin(eucl_dists/2) * 180 / np.pi

    # Temporal smoothing
    tp_min_length = np.int64(np.ceil(np.sqrt(-2 * tp_sigma ** 2 * np.log(.01 * tp_sigma * np.sqrt(2 * np.pi)))))
    tp_kernel = np.linspace(-tp_min_length, tp_min_length, num=2 * tp_min_length + 1)
    tp_kernel = 1 / (tp_sigma * np.sqrt(2 * np.pi)) * np.exp(-tp_kernel ** 2 / (1/3 * tp_sigma ** 2))
    tp_kernel *= tp_kernel > .0001
    tp_smooth_x = signal.convolve(flow_vec[:, :, 0], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_y = signal.convolve(flow_vec[:, :, 1], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_z = signal.convolve(flow_vec[:, :, 2], tp_kernel[np.newaxis, :], mode='same')

    # Spatial smoothing
    sp_kernel = np.exp(-(interpos_distances ** 2) / (1/3 * sp_sigma ** 2))
    sp_kernel *= sp_kernel > .001

    sp_smooth_x = np.dot(sp_kernel, tp_smooth_x)
    sp_smooth_y = np.dot(sp_kernel, tp_smooth_y)
    sp_smooth_z = np.dot(sp_kernel, tp_smooth_z)

    return np.ascontiguousarray(np.array([sp_smooth_x, sp_smooth_y, sp_smooth_z]).transpose(), dtype=np.float32)


class Canvas(app.Canvas):

    frame_num = 200
    sp_sigma = 60.  # spatial contiguity radius [deg]
    tp_sigma = 1.  # temporal contiguity radius [s]
    fps = 20
    nominal_velocity = 30  # deg/s

    def __init__(self):
        app.Canvas.__init__(self, keys='interactive')

        sphere = IcosahedronSphere(subdiv_lvl=2)
        vertices = sphere.get_vertices()
        indices = sphere.get_indices()

        # Calculate face centers and save individual vertex copies for each face
        face_idcs = indices.reshape((-1, 3))
        self.face_num = face_idcs.shape[0]
        self.faces = np.zeros_like(face_idcs)
        self.centers = np.zeros((self.face_num, 3), dtype=np.float32)
        self.vertices = np.zeros((self.face_num*3, 3), dtype=np.float32)
        for i, face in enumerate(face_idcs):
            verts = vertices[face]
            for j, v in enumerate(verts):
                v_idx = i * 3 + j
                self.faces[i, j] = v_idx
                self.vertices[v_idx] = v
            center = np.mean(verts, axis=0)
            self.centers[i] = center / np.linalg.norm(center)

        self.indices = np.ascontiguousarray(self.faces.flatten(), dtype=np.uint32)
        self.index_buffer = gloo.IndexBuffer(self.indices)

        # Create motion vectors for face centers
        self.motion_vectors = create_motion_matrix(self.centers, frame_num=self.frame_num,
                                                   sp_sigma=self.sp_sigma, tp_sigma=int(self.tp_sigma * self.fps))

        # Add bias (for testing):
        # print(np.min(self.motion_vectors), np.max(self.motion_vectors))
        # self.motion_vectors += np.array([-2., 0., 0.])[None,None,:]

        # Create array for rotation matrices
        self.rotation_matrices = np.zeros((self.frame_num, self.face_num, 4, 4), dtype=np.float32)

        self.local_motion_vectors = np.zeros((self.frame_num, self.face_num, 3), dtype=np.float32)

        for tidx, motion_vectors in enumerate(self.motion_vectors):
            for iidx, (center, motvec) in enumerate(zip(self.centers, motion_vectors)):

                # Calculate local motion vector at center positions
                motvec_local = motvec - center * np.dot(motvec, center)
                self.local_motion_vectors[tidx, iidx] = motvec_local

                # Get axis perpendicular to center and local motion vector
                axis = crossproduct(motvec_local / np.linalg.norm(motvec_local), center)
                # Set rotation angle based on length of local motion vector
                angle = np.arctan(np.linalg.norm(motvec_local))
                # Calculate rotation matrix
                rotmat = transforms.rotate(angle / np.pi * self.nominal_velocity / self.fps, axis)
                self.rotation_matrices[tidx, iidx] = rotmat

        # plt.figure()
        # plt.hist(np.linalg.norm(self.local_motion_vectors, axis=2).flatten(), bins=50) # Nash's way
        # plt.hist(np.arctan(np.linalg.norm(self.local_motion_vectors, axis=2).flatten()), bins=50)  # major circle arcs
        # plt.show()

        # Array to save latest rotation for each face
        self.current_rotations = (np.ones((self.centers.shape[0], 4, 4), dtype=np.float32)
                                  * np.eye(4)[None,:,:])

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = self.vertices
        self.program['u_view'] = transforms.translate((0, 0, -2))
        self.model = np.eye(4)

        self.start_time = time.perf_counter()
        self.last_time = time.perf_counter()
        self.last_tidx = -1

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        self.theta = 0
        self.phi = 0

        self.show()

    def _stack_attribute(self, data: np.ndarray) -> np.ndarray:
        return np.repeat(data[:, :, None], 3, axis=0).reshape((-1, 4))

    def on_timer(self, event):

        self.theta += .0#1
        # self.phi += .5
        self.model = np.dot(transforms.rotate(self.theta, (0, 1, 0)),
                            transforms.rotate(self.phi, (0, 0, 1)))

        self.update()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.physical_size[0], event.physical_size[1])
        self.projection = transforms.perspective(70.0, event.size[0] / float(event.size[1]), 1.0, 10.0)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True, stencil=True)

        current_time = time.perf_counter()
        print(f'{(current_time-self.last_time)*1000:.1f}ms')
        self.last_time = current_time

        # Fetch motion vectors for this frame
        tidx = int((current_time - self.start_time) * self.fps) % self.frame_num

        # print(tidx)

        # Reset last index
        if tidx < self.last_tidx:
            self.last_tidx = -1

        self.program['u_projection'] = self.projection
        self.program['u_model'] = self.model

        if tidx > self.last_tidx:
            for i, (current_rot, new_rot) in enumerate(zip(self.current_rotations, self.rotation_matrices[tidx])):

                # Multiply current rotation with previous one
                rotmat_new = np.dot(new_rot, current_rot)

                self.current_rotations[i] = rotmat_new

            # Convert
            quats = qt.array.from_rotation_matrix(self.current_rotations)
            # Stack
            stacked_quats = np.ascontiguousarray(np.repeat(quats[:, None], 3, axis=0).reshape((-1, 4)), dtype=np.float32)
            # Write
            self.program['a_rotation'] = stacked_quats

            self.last_tidx = tidx

        self.program.draw('triangles', self.index_buffer)


if __name__ == '__main__':

    # vispy.use('PySide6')

    with open('./cmn_redesign.vert', 'r') as f:
        VERT_SHADER = f.read()

    with open('./cmn_redesign.frag', 'r') as f:
        FRAG_SHADER = f.read()

    canvas = Canvas()
    if sys.flags.interactive != 1:
        app.run()
