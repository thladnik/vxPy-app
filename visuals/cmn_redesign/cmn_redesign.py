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


def create_motion_matrix(positions: np.ndarray,
                         frame_num: int,  # [frames]
                         tp_cr: int = 20,  # [frames]
                         sp_cr: float = 30  # [deg]
                         ) -> np.ndarray:
    """Create a motion matrix of given parameters
    """
    start_t = time.perf_counter()
    print('Create motion matrix')

    tp_sigma = tp_cr / (2 * np.sqrt(-np.log(0.1)))
    sp_sigma = sp_cr / (2 * np.sqrt(-np.log(0.1)))

    print(f'TP sigma: {tp_sigma:.2f} frames')
    print(f'SP sigma: {sp_sigma:.2f} deg')

    # Create flow vectors for each face and frame and normalize them
    flow_vec = np.random.normal(size=(positions.shape[0], frame_num, 3))  # Random white noise motion vector
    flow_vec /= np.linalg.norm(flow_vec, axis=-1)[:, :, None]

    # Temporal smoothing
    # tp_min_length = int(np.ceil(np.sqrt(-2 * tp_sigma ** 2 * np.log(.01 * tp_sigma * np.sqrt(2 * np.pi)))))
    # tp_range = np.linspace(-tp_min_length, tp_min_length, num=2 * tp_min_length + 1)
    tp_range = np.linspace(-2 * tp_cr, 2 * tp_cr, num=4 * tp_cr + 1)
    tp_kernel = 1 / (tp_sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (tp_range / tp_sigma) ** 2)
    tp_smooth_x = signal.convolve(flow_vec[:, :, 0], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_y = signal.convolve(flow_vec[:, :, 1], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_z = signal.convolve(flow_vec[:, :, 2], tp_kernel[np.newaxis, :], mode='same')

    # Spatial smoothing
    # Calculate euclidean position/position distances and convert from chord to angle
    # see https://en.wikipedia.org/wiki/Chord_(geometry)#In_trigonometry
    eucl_dists = scipy.spatial.distance.cdist(positions, positions)
    eucl_dists[eucl_dists > 2.] = 2.  # Fix rounding errors (no def. for arcsin(x) if x > 1)
    interpos_distances = 2 * np.arcsin(eucl_dists / 2) * 180 / np.pi
    sp_kernel = 1 / (sp_sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * (interpos_distances ** 2) / (sp_sigma ** 2))
    sp_kernel *= sp_kernel > .001

    sp_smooth_x = np.dot(sp_kernel, tp_smooth_x)
    sp_smooth_y = np.dot(sp_kernel, tp_smooth_y)
    sp_smooth_z = np.dot(sp_kernel, tp_smooth_z)

    print(f'Generation time: {time.perf_counter() - start_t:.1f}')

    return np.ascontiguousarray(np.array([sp_smooth_x, sp_smooth_y, sp_smooth_z]).transpose(), dtype=np.float32)


def project_motion_vectors(centers: np.ndarray, motion_vectors: np.ndarray):
    print('Project motion vectors')

    start_t = time.perf_counter()

    # Create output arrays
    rotation_quats = np.zeros((*motion_vectors.shape[:2], 4), dtype=np.float32)
    local_motion_vectors = np.zeros((*motion_vectors.shape[:2], 3), dtype=np.float32)
    angular_velocities = np.zeros(motion_vectors.shape[:2], dtype=np.float32)

    for tidx, motion_vecs in enumerate(motion_vectors):
        motvecs_local = motion_vecs - centers * np.sum(motion_vecs * centers, axis=1)[:, None]
        local_motion_vectors[tidx] = motvecs_local

        axes = crossproduct(motvecs_local / np.linalg.norm(motvecs_local, axis=1)[:, None], centers)
        axes /= np.linalg.norm(axes, axis=1)[:, None]

        angles = -np.linalg.norm(motvecs_local, axis=1)
        angular_velocities[tidx, :] = angles

        # Calculate quaternion
        rotation_quats[tidx, :, 0] = np.cos(angles / 2)
        rotation_quats[tidx, :, 1] = axes[:, 0] * np.sin(angles / 2)
        rotation_quats[tidx, :, 2] = axes[:, 1] * np.sin(angles / 2)
        rotation_quats[tidx, :, 3] = axes[:, 2] * np.sin(angles / 2)

    print(f'Projection time: {time.perf_counter() - start_t:.1f}')

    return local_motion_vectors, angular_velocities, rotation_quats


class ContiguousMotionNoise3D(vxvisual.SphericalVisual):
    subdivision_level = 3
    frame_num = 10000
    sp_cr = 57.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 30  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)

    time = vxvisual.FloatParameter('time', internal=True)
    frame_index = vxvisual.IntParameter('frame_index', internal=True)
    noise_scale = vxvisual.FloatParameter('noise_scale', default=5., limits=(.01, 50.), step_size=.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        print(f'Running at {self.fps} fps')
        print(f'Frame number {self.frame_num} frames')
        print(f'Spatial CR {self.sp_cr} deg')
        print(f'Temporal CR {self.tp_cr} s')
        print(f'Nominal velocity {self.nominal_velocity} deg/s')

        np.random.seed(1)

        # Define name for temporary dump
        save_group_name = self.__class__.__name__

        # # If temp data already exists for this implementation of the visual, use it
        # if vxcontainer.temporary_group_exists(save_group_name):
        #     saved_data = vxcontainer.temporary_load_group(save_group_name)
        #
        #     self.centers = saved_data['centers']
        #     self.vertices = saved_data['vertices']
        #     self.indices = saved_data['indices']
        #     self.motion_vectors = saved_data['motion_vectors']
        #     self.rotation_quats = saved_data['rotation_quats']
        #     self.local_motion_vectors = saved_data['local_motion_vectors']
        #
        # # If no temp data exists, create all visual data here
        # else:

        sph = sphere.IcosahedronSphere(subdiv_lvl=self.subdivision_level)
        vertices = sph.get_vertices()
        indices = sph.get_indices()

        # Calculate face centers and save individual vertex copies for each face
        face_idcs = indices.reshape((-1, 3))
        self.face_num = face_idcs.shape[0]
        self.faces = np.zeros_like(face_idcs)
        self.centers = np.zeros((self.face_num, 3), dtype=np.float32)
        self.vertices = np.zeros((self.face_num * 3, 3), dtype=np.float32)
        for i, face in enumerate(face_idcs):
            verts = vertices[face]
            for j, v in enumerate(verts):
                v_idx = i * 3 + j
                self.faces[i, j] = v_idx
                self.vertices[v_idx] = v
            center = np.mean(verts, axis=0)
            self.centers[i] = center / np.linalg.norm(center)

        self.indices = np.ascontiguousarray(self.faces.flatten(), dtype=np.uint32)

        # Create motion vectors
        self.motion_vectors = create_motion_matrix(self.centers,
                                                   frame_num=self.frame_num,
                                                   sp_cr=self.sp_cr,
                                                   tp_cr=int(self.tp_cr * self.fps))

        # Add bias (for testing):
        self.motion_vectors += self.motion_vector_bias[None, None, :]

        # Normalize velocity
        motion_norms = np.linalg.norm(self.motion_vectors, axis=-1)
        self.motion_vectors *= self.nominal_velocity / motion_norms.max() * np.pi / 180

        # Project local motion vectors and calculate unit quaternions
        t = time.perf_counter()
        self.local_motion_vectors, self.angular_velocities, self.rotation_quats = project_motion_vectors(self.centers,
                                                                                                         self.motion_vectors)

        # Convert to quaternion array
        self.rotation_quats = qt.array(self.rotation_quats).normalized

        # Show velocity distribution
        # plt.figure()
        # plt.hist(np.linalg.norm(self.local_motion_vectors, axis=2).flatten() * 180 / np.pi, bins=50)
        # plt.xlabel('Velocities [deg/s]')
        # plt.show()

        # # Save visual data to temporary files
        # data_dump = dict(centers=self.centers, vertices=self.vertices, indices=self.indices,
        #                  motion_vectors=self.motion_vectors, local_motion_vectors=self.local_motion_vectors,
        #                  rotation_quats=self.rotation_quats)
        #
        # vxcontainer.temporary_dump_group(save_group_name, data_dump)

        # Arrays to save latest rotation for each face
        start_rotations = np.ones((self.centers.shape[0], 4, 4), dtype=np.float32) * np.eye(4)[None, :, :]
        self.current_quats = qt.array.from_rotation_matrix(start_rotations)

        # Create program
        self.VERT_SHADER = self.load_vertex_shader('./cmn_redesign.vert')
        self.FRAG_SHADER = self.load_shader('./cmn_redesign.frag')
        self.program = gloo.Program(self.VERT_SHADER, self.FRAG_SHADER)
        self.program['a_position'] = self.vertices
        self.index_buffer = gloo.IndexBuffer(self.indices)
        self.noise_scale.connect(self.program)

        data_dump = dict(centers=self.centers, vertices=self.vertices, indices=self.indices,
                         motion_vectors=self.motion_vectors, local_motion_vectors=self.local_motion_vectors,
                         rotation_quats=self.rotation_quats)

        vxcontainer.dump(data_dump, group=self.__class__.__name__)

    def initialize(self, **kwargs):
        self.time.data = 0.
        self.frame_index.data = -1

    def render(self, dt):

        # Update time
        self.time.data += dt

        # Get correct time index
        tidx = int(self.time.data[0] * self.fps) % self.frame_num
        self.frame_index.data = tidx

        # Calculate future rotations (for t += 1 second)
        future_quats = self.rotation_quats[tidx] * self.current_quats

        # SLERP current rotations
        self.current_quats = qt.slerp(self.current_quats, future_quats, dt)

        # Stack for write
        stacked_quats = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)

        # Write to program
        self.program['a_rotation'] = stacked_quats

        # Render
        self.program.draw('triangles', self.index_buffer)


class CMNUp(ContiguousMotionNoise3D):
    subdivision_level = 2
    frame_num = 500
    sp_cr = 30.  # spatial contiguity radius [deg]
    tp_cr = 3.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 50  # mean local velocity [deg/s]
    motion_vector_bias = np.array([.0, .0, .2])


class CMNForward(ContiguousMotionNoise3D):
    subdivision_level = 2
    frame_num = 500
    sp_cr = 30.  # spatial contiguity radius [deg]
    tp_cr = 3.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 50  # mean local velocity [deg/s]
    motion_vector_bias = np.array([.2, .0, .0])


class CMNLeft(ContiguousMotionNoise3D):
    subdivision_level = 2
    frame_num = 500
    sp_cr = 30.  # spatial contiguity radius [deg]
    tp_cr = 3.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 50  # mean local velocity [deg/s]
    motion_vector_bias = np.array([.0, .2, .0])


class CMN3D20240404(ContiguousMotionNoise3D):
    subdivision_level = 2
    frame_num = 500
    sp_cr = 30.  # spatial contiguity radius [deg]
    tp_cr = 3.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 300  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)


class CMN3D20240410(ContiguousMotionNoise3D):
    """Stimulus parameters do not seem to work"""
    subdivision_level = 3
    frame_num = 30_000
    sp_cr = 57.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)


class CMN3D20240411(ContiguousMotionNoise3D):
    subdivision_level = 2
    frame_num = 30_000
    sp_cr = 57.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 20  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)


if __name__ == '__main__':
    pass
