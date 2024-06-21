import math
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
from vxpy.utils import sphere, geometry


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


class CMN_FORE_BACK(vxvisual.SphericalVisual):
    subdivision_level = 2
    frame_num = 10000
    sp_cr = 90.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 30  # [frames/s]
    nominal_velocity = 120  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)

    time = vxvisual.FloatParameter('time', internal=True)
    past_time_position: float = None
    time_mod: float = None
    frame_index = vxvisual.IntParameter('frame_index', internal=True)
    noise_scale = vxvisual.FloatParameter('noise_scale', default=2.5, limits=(.01, 50.), step_size=.01)
    phase_num: int = None
    pause_num: int = None
    duration: int = None
    direction: float = None
    Fore_CMN_phase: int = None
    Repeat_times: int = None
    initiation: int = None
    back_and_forth_num: int = None
    VERT_PATH = ''
    FRAG_PATH = ''
    fore_rotate: float = None
    fore_stationary: float = None

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        print(f'Running at {self.fps} fps')
        print(f'Frame number {self.frame_num} frames')
        print(f'Spatial CR {self.sp_cr} deg')
        print(f'Temporal CR {self.tp_cr} s')
        print(f'Nominal velocity {self.nominal_velocity} deg/s')

        np.random.seed(1)

        self.sphere = (sphere.IcosahedronSphere_cmn_for_back(subdiv_lvl=self.subdivision_level))
        vertices = self.sphere.get_vertices()
        indices = self.sphere.get_indices()
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
            self.centers[i] = (center / np.linalg.norm(center))
        self.indices = np.ascontiguousarray(self.faces.flatten(), dtype=np.uint32)
        # self.centers = np.ascontiguousarray(np.repeat(self.centers, 3, 0), dtype=np.float32)
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

        start_rotations = np.ones((self.centers.shape[0], 4, 4), dtype=np.float32) * np.eye(4)[None, :, :]
        self.current_quats = qt.array.from_rotation_matrix(start_rotations)
        self.start_rotation = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)

        # Create program
        self.program = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        self.index_buffer = gloo.IndexBuffer(self.indices)
        self.program['a_position'] = self.vertices
        # self.program['a_start_rotation'] = self.start_rotation
        self.program['a_triangle_center_az'] = self.sphere.triangle_center_az
        self.program['a_triangle_center_el'] = self.sphere.triangle_center_el
        self.program['a_past_time_pos'] = self.past_time_position
        self.noise_scale.connect(self.program)
        self.time.connect(self.program)

        # self.last_phase_stacked_quats = qt.array.from_rotation_matrix(start_rotations)
        # stacked_quats_currentframe = qt.array.from_rotation_matrix(start_rotations)
        stacked_quats_background = qt.array.from_rotation_matrix(start_rotations)
        # stacked_quats_foreground = qt.array.from_rotation_matrix(start_rotations)
        stacked_quats_initiation = qt.array.from_rotation_matrix(start_rotations)
        self.current_quats = qt.array.from_rotation_matrix(start_rotations)
        self.stacked_quats_initiation = qt.array.from_rotation_matrix(start_rotations)
        self.foreground_cmn_current_quats = qt.array.from_rotation_matrix(start_rotations)
        # initiation
        for initiation in range(self.initiation):
            for i in range(5 * self.fps):  # cmn_duration = 5
                stacked_quats_initiation_nextframe = self.rotation_quats[i] * stacked_quats_initiation
                stacked_quats_initiation = qt.slerp(stacked_quats_initiation, stacked_quats_initiation_nextframe,
                                                    1 / self.fps)
        self.stacked_quats_initiation = stacked_quats_initiation
        # stacked_quats_background = self.stacked_quats_initiation
        # self.current_quats = self.stacked_quats_initiation
        self.current_after_station_quats = self.stacked_quats_initiation
        # self.foreground_cmn_current_quats = self.stacked_quats_initiation

        # Foreground
        stacked_quats_foreground = self.stacked_quats_initiation
        self.stacked_quats_foreground_start = np.repeat(stacked_quats_foreground[:, None], 3,
                                                        axis=0).reshape((-1, 4)).astype(np.float32)
        self.program['a_foreground_rotation_start'] = self.stacked_quats_foreground_start

        # Background
        for i in range(50 * self.fps):  # cmn_background_moving = 45
            stacked_quats_backmove_nextframe = self.rotation_quats[i] * stacked_quats_background
            stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_backmove_nextframe,
                                                1 / self.fps)
        self.last_phase_stacked_quats = stacked_quats_background

        stacked_quats_stationary = np.repeat(self.last_phase_stacked_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(
            np.float32)
        self.program['a_rotation_stationary'] = stacked_quats_stationary

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
        if 50 <= math.fmod(self.time.data, 180) <= 135:
            self.current_quats = self.last_phase_stacked_quats
        else:
            self.current_quats = self.current_quats
            future_quats = self.rotation_quats[tidx] * self.current_quats
            self.current_quats = qt.slerp(self.current_quats, future_quats, dt)
        current_time = self.time.data
        if 10 <= math.fmod(self.time.data, 180) <= 25:
            self.time_mod = math.fmod((current_time - 10), 15)
        elif 30 <= math.fmod(self.time.data, 180) <= 45:
            self.time_mod = math.fmod((current_time - 30), 15)
        elif 55 <= math.fmod(self.time.data, 180) <= 70:
            self.time_mod = math.fmod((current_time - 55), 15)
        elif 75 <= math.fmod(self.time.data, 180) <= 90:
            self.time_mod = math.fmod((current_time - 75), 15)
        elif 95 <= math.fmod(self.time.data, 180) <= 110:
            self.time_mod = math.fmod((current_time - 95), 15)
        elif 115 <= math.fmod(self.time.data, 180) <= 130:
            self.time_mod = math.fmod((current_time - 115), 15)
        elif 140 <= math.fmod(self.time.data, 180) <= 155:
            self.time_mod = math.fmod((current_time - 140), 15)
        elif 160 <= math.fmod(self.time.data, 180) < 175:
            self.time_mod = math.fmod((current_time - 160), 15)
        else:
            self.time_mod = 0
        self.program['time_mod'] = self.time_mod

        # foreground_timing for direction
        if 10 <= math.fmod(self.time.data, 180) <= 25 or 75 <= math.fmod(self.time.data, 180) <= 90 or 95 <= math.fmod(
                self.time.data, 180) <= 110 or 160 <= math.fmod(self.time.data, 180) <= 175:
            self.direction = 1
        elif 30 <= math.fmod(self.time.data, 180) <= 45 or 55 <= math.fmod(self.time.data,
                                                                           180) <= 70 or 115 <= math.fmod(
                self.time.data, 180) <= 130 or 140 <= math.fmod(self.time.data, 180) <= 155:
            self.direction = -1
        else:
            self.direction = 0
        self.program['a_fore_direct'] = self.direction

        # foreground_timing for position
        if 25 <= math.fmod(self.time.data, 180) < 45 or 110 <= math.fmod(self.time.data, 180) < 130:
            self.past_time_position = math.pi * 15 / 6
        elif 70 <= math.fmod(self.time.data, 180) < 90 or 155 <= math.fmod(self.time.data, 180) < 175:
            self.past_time_position = -15 * math.pi / 6
        else:
            self.past_time_position = 0.
        self.program['a_past_time_pos'] = self.past_time_position

        # foreground_timing for rotation or stationary
        if 0 <= math.fmod(self.time.data, 180) <= 5:
            self.fore_rotate = 0
            self.fore_stationary = 1
        else:
            self.fore_rotate = 1
            self.fore_stationary = 0
        self.program['a_fore_rotate'] = self.fore_rotate
        self.program['a_fore_stationary'] = self.fore_stationary

        # Calculate future rotations (for t += 1 second)
        fore_cmn_future_quats = self.rotation_quats[tidx] * self.foreground_cmn_current_quats
        # SLERP current rotations
        self.foreground_cmn_current_quats = qt.slerp(self.foreground_cmn_current_quats, fore_cmn_future_quats, dt)

        # Stack for write
        stacked_quats = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)
        foreground_cmn_stacked_quats = np.repeat(self.foreground_cmn_current_quats[:, None], 3, axis=0).reshape(
            (-1, 4)).astype(np.float32)

        # Write to program
        if 50 <= math.fmod(self.time.data, 170) <= 135:
            self.program['a_rotation'] = self.program['a_rotation_stationary']
        else:
            self.program['a_rotation'] = stacked_quats
        self.program['a_foreground_cmn_rotation'] = foreground_cmn_stacked_quats
        # Render
        self.program.draw('triangles', self.index_buffer)


class FORE_NON_CMN(CMN_FORE_BACK):  # pause1
    VERT_PATH = 'main.vert'
    FRAG_PATH = 'main.frag'
    phase_num: int = 0
    pause_num: int = 0
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 0
