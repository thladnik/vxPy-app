import sys
import time

import scipy
from scipy import signal
from vispy import gloo
from vispy import app
import numpy as np
from vispy.util import transforms
import quaternionic as qt

import vxpy.core.visual as vxvisual
import vxpy.core.logger as vxlogger
import vxpy.core.container as vxcontainer
from vxpy.utils import sphere


log = vxlogger.getLogger(__name__)


def crossproduct(v1, v2):
    """Just workaround for known return value bug in numpy"""
    return np.cross(v1 / np.linalg.norm(v1), v2)


def create_3d_motion_matrix(positions: np.ndarray,
                            frame_num: int,  # [frames]
                            tp_cr: int = 20,  # [frames]
                            sp_cr: float = 30  # [deg]
                            ) -> np.ndarray:
    """Create a motion matrix of given parameters
    """

    # Create flow vectors for each face and frame and normalize them
    flow_vec = np.random.normal(size=(positions.shape[0], frame_num, 3))  # Random white noise motion vector
    flow_vec /= np.linalg.norm(flow_vec, axis=-1)[:, :, None]

    # Calculate euclidean position/position distances and convert from chord to angle
    # see https://en.wikipedia.org/wiki/Chord_(geometry)#In_trigonometry
    eucl_dists = scipy.spatial.distance.cdist(positions, positions)
    eucl_dists[eucl_dists > 2.] = 2.  # Fix rounding errors (no def. for arcsin(x) if x > 1)
    interpos_distances = 2 * np.arcsin(eucl_dists / 2) * 180 / np.pi

    # Temporal smoothing
    tp_sigma = tp_cr / (2 * np.sqrt(-np.log(0.1)))
    tp_min_length = np.int64(np.ceil(np.sqrt(-2 * tp_sigma ** 2 * np.log(.01 * tp_sigma * np.sqrt(2 * np.pi)))))
    tp_range = np.linspace(-tp_min_length, tp_min_length, num=2 * tp_min_length + 1)
    tp_kernel = 1 / (tp_sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * tp_range ** 2 / tp_sigma ** 2)
    tp_kernel *= tp_kernel > .001
    tp_smooth_x = signal.convolve(flow_vec[:, :, 0], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_y = signal.convolve(flow_vec[:, :, 1], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_z = signal.convolve(flow_vec[:, :, 2], tp_kernel[np.newaxis, :], mode='same')

    # Spatial smoothing
    sp_sigma = sp_cr / (2 * np.sqrt(-np.log(0.1)))
    sp_kernel = 1 / (sp_sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (interpos_distances ** 2) / (sp_sigma ** 2))
    sp_kernel *= sp_kernel > .001

    sp_smooth_x = np.dot(sp_kernel, tp_smooth_x)
    sp_smooth_y = np.dot(sp_kernel, tp_smooth_y)
    sp_smooth_z = np.dot(sp_kernel, tp_smooth_z)

    return np.ascontiguousarray(np.array([sp_smooth_x, sp_smooth_y, sp_smooth_z]).transpose(), dtype=np.float32)


class ContiguousMotionNoise3D(vxvisual.SphericalVisual):

    subdivision_level = 2
    frame_num = 200
    sp_cr = 60.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)

    def __init__(self, *args, **kwargs):
        super().__init__(self , *args, **kwargs)

        np.random.seed(1)

        # Define name for temporary dump
        save_group_name = self.__class__.__name__

        # If temp data already exists for this implementation of the visual, use it
        if vxcontainer.temporary_group_exists(save_group_name):
            saved_data = vxcontainer.temporary_load_group(save_group_name)

            self.centers = saved_data['centers']
            self.vertices = saved_data['vertices']
            self.indices = saved_data['indices']
            self.motion_vectors = saved_data['motion_vectors']
            self.rotation_quats = saved_data['rotation_quats']
            self.local_motion_vectors = saved_data['local_motion_vectors']

        # If no temp data exists, create all visual data here
        else:

            sph = sphere.IcosahedronSphere(subdiv_lvl=self.subdivision_level)
            vertices = sph.get_vertices()
            indices = sph.get_indices()

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

            # Create motion vectors for face centers
            self.motion_vectors = create_3d_motion_matrix(self.centers,
                                                          frame_num=self.frame_num,
                                                          sp_cr=self.sp_cr,
                                                          tp_cr=int(self.tp_cr * self.fps))

            # Add bias (for testing):
            self.motion_vectors += self.motion_vector_bias[None,None,:]

            # Create array for rotation matrices
            self.rotation_quats = np.zeros((self.frame_num, self.face_num, 4), dtype=np.float32)
            self.local_motion_vectors = np.zeros((self.frame_num, self.face_num, 3), dtype=np.float32)
            for tidx, motion_vectors in enumerate(self.motion_vectors):
                for iidx, (center, motvec) in enumerate(zip(self.centers, motion_vectors)):

                    # Calculate local motion vector at center positions
                    motvec_local = motvec - center * np.dot(motvec, center)
                    self.local_motion_vectors[tidx, iidx] = motvec_local

                    # Get axis perpendicular to center and local motion vector
                    axis = crossproduct(motvec_local / np.linalg.norm(motvec_local), center)
                    # Set rotation angle based on length of local motion vector
                    angle = -np.arctan(np.linalg.norm(motvec_local))
                    # Calculate rotation
                    self.rotation_quats[tidx, iidx] = qt.array.from_axis_angle(axis*angle*self.nominal_velocity/360)

            # Save visual data to temporary files
            data_dump = dict(centers=self.centers, vertices=self.vertices, indices=self.indices,
                             motion_vectors=self.motion_vectors, local_motion_vectors=self.local_motion_vectors,
                             rotation_quats=self.rotation_quats)

            vxcontainer.temporary_dump_group(save_group_name, data_dump)

        # Convert to quaternion for speed
        self.rotation_quats = qt.array(self.rotation_quats)

        # Arrays to save latest rotation for each face
        self.current_rotations = np.ones((self.centers.shape[0], 4, 4), dtype=np.float32) * np.eye(4)[None,:,:]
        self.current_quats = qt.array.from_rotation_matrix(self.current_rotations)

        # Create program
        self.VERT_SHADER = self.load_vertex_shader('./cmn_redesign.vert')
        self.FRAG_SHADER = self.load_shader('./cmn_redesign.frag')
        self.program = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self.program['a_position'] = self.vertices
        self.index_buffer = gloo.IndexBuffer(self.indices)

        self.time = 0.
        self.last_tidx = -1

    def initialize(self, **kwargs):
        self.time = 0.
        self.last_tidx = -1

    def render(self, dt):

        # Update time
        self.time += dt

        # Get correct time index
        tidx = int(self.time * self.fps) % self.frame_num

        # Reset last index if frame number is exceeded
        if tidx < self.last_tidx:
            log.warning('Maximum frame number for visual exceeded. Starting from beginning.')
            self.last_tidx = -1

        if tidx > self.last_tidx:

            # Apply new rotation to current one to update current
            self.current_quats = self.rotation_quats[tidx] * self.current_quats
            # Stack for attribute upload
            stacked_quats = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)
            # Write to attribute
            self.program['a_rotation'] = stacked_quats
            # Update last index
            self.last_tidx = tidx

        # Render
        self.program.draw('triangles', self.index_buffer)


class CMN3D20240404(ContiguousMotionNoise3D):

    subdivision_level = 3
    frame_num = 2000
    sp_cr = 60.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)


if __name__ == '__main__':

    pass
