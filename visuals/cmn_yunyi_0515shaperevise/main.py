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
    sp_cr = 57.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 30  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)

    time = vxvisual.FloatParameter('time', internal=True)
    frame_index = vxvisual.IntParameter('frame_index', internal=True)
    noise_scale = vxvisual.FloatParameter('noise_scale', default=2.5, limits=(.01, 50.), step_size=.01)
    phase_num: int = None
    pause_num: int = None
    duration: int = None
    Fore_CMN_phase: int = None
    Repeat_times: int = None
    initiation: int = None
    back_and_forth_num: int = None
    VERT_PATH = ''
    FRAG_PATH = ''

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
        self.noise_scale.connect(self.program)
        self.time.connect(self.program)

        # self.last_phase_stacked_quats = qt.array.from_rotation_matrix(start_rotations)
        # stacked_quats_currentframe = qt.array.from_rotation_matrix(start_rotations)
        stacked_quats_background = qt.array.from_rotation_matrix(start_rotations)
        stacked_quats_foreground = qt.array.from_rotation_matrix(start_rotations)
        stacked_quats_initiation = qt.array.from_rotation_matrix(start_rotations)
        self.stacked_quats_initiation = qt.array.from_rotation_matrix(start_rotations)

        # initiation
        for initiation in range(self.initiation):
            for i in range(5 * self.fps):  # cmn_duration = 5
                stacked_quats_initiation_nextframe = self.rotation_quats[i] * stacked_quats_initiation
                stacked_quats_initiation = qt.slerp(stacked_quats_initiation, stacked_quats_initiation_nextframe,
                                                    1 / self.fps)
        self.stacked_quats_initiation = stacked_quats_initiation

        # Foreground
        stacked_quats_foreground = self.stacked_quats_initiation
        self.stacked_quats_foreground_start = np.repeat(stacked_quats_foreground[:, None], 3,
                                                        axis=0).reshape((-1, 4)).astype(np.float32)
        self.program['a_foreground_rotation_start'] = self.stacked_quats_foreground_start

        # Background
        stacked_quats_background = self.stacked_quats_initiation

        if self.back_and_forth_num == 1:
            if self.pause_num == 1:
                for pause_num in range(self.pause_num * 5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(self.phase_num * 15 * self.fps):  # phase_duration = 15
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
            if self.pause_num == 2:
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(15 * self.fps):  # phase_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                if self.phase_num == 2:
                    for pause_num in range(15 * self.fps):  # pause_duration = 5
                        stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                        stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                            1 / self.fps)
            if self.pause_num == 3:
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(15 * self.fps):  # phase_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(15 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
        if self.back_and_forth_num == 2:
            for pause_num in range(5 * self.fps):  # pause_duration = 5
                stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                    1 / self.fps)
            for phase_num in range(15 * self.fps):  # phase_duration = 5
                stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                    1 / self.fps)
            for pause_num in range(5 * self.fps):  # pause_duration = 5
                stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                    1 / self.fps)
            for pause_num in range(15 * self.fps):  # pause_duration = 5
                stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                    1 / self.fps)
            for pause_num in range(5 * self.fps):  # pause_duration = 5
                stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                    1 / self.fps)
            if self.pause_num == 1:
                for pause_num in range(self.pause_num * 5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(self.phase_num * 15 * self.fps):  # phase_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
            if self.pause_num == 2:
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(15 * self.fps):  # phase_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                if self.phase_num == 2:
                    for pause_num in range(15 * self.fps):  # pause_duration = 5
                        stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                        stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                            1 / self.fps)
            if self.pause_num == 3:
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for phase_num in range(15 * self.fps):  # phase_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[phase_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(15 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
                for pause_num in range(5 * self.fps):  # pause_duration = 5
                    stacked_quats_nextframe = self.rotation_quats[pause_num] * stacked_quats_background
                    stacked_quats_background = qt.slerp(stacked_quats_background, stacked_quats_nextframe,
                                                        1 / self.fps)
        self.last_phase_stacked_quats = stacked_quats_background
        self.foreground_cmn_current_quats = self.stacked_quats_initiation
        self.current_quats = self.last_phase_stacked_quats
        stacked_quats_stationary = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)
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

        # Calculate future rotations (for t += 1 second)
        future_quats = self.rotation_quats[tidx] * self.current_quats
        fore_cmn_future_quats = self.rotation_quats[tidx] * self.foreground_cmn_current_quats
        # SLERP current rotations
        self.current_quats = qt.slerp(self.current_quats, future_quats, dt)
        self.foreground_cmn_current_quats = qt.slerp(self.foreground_cmn_current_quats, fore_cmn_future_quats, dt)

        # Stack for write
        stacked_quats = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)
        foreground_cmn_stacked_quats = np.repeat(self.foreground_cmn_current_quats[:, None], 3, axis=0).reshape(
            (-1, 4)).astype(np.float32)

        # Write to program
        self.program['a_rotation'] = stacked_quats
        self.program['a_foreground_cmn_rotation'] = foreground_cmn_stacked_quats
        # Render
        self.program.draw('triangles', self.index_buffer)


class CMN(CMN_FORE_BACK):
    VERT_PATH = 'CMN.vert'
    FRAG_PATH = 'CMN.frag'
    phase_num: int = 0
    duration: int = 0
    CMN_phase: int = 0
    initiation = 0
    back_and_forth_num = 0


class ForegroundStationary(CMN_FORE_BACK):  # pause1
    VERT_PATH = 'foreground_stationary.vert'
    FRAG_PATH = 'foreground_stationary.frag'
    phase_num: int = 0
    pause_num: int = 0
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class ForegroundMovingForward(CMN_FORE_BACK):
    VERT_PATH = 'foreground_moving_forward.vert'
    FRAG_PATH = 'foreground_moving_forward.frag'
    phase_num: int = 0
    pause_num: int = 1
    duration: int = 15  # no use
    CMN_phase: int = 1  # no use
    initiation = 1
    back_and_forth_num = 1


class Pause2(CMN_FORE_BACK):
    VERT_PATH = 'pause2.vert'
    FRAG_PATH = 'pause2.frag'
    phase_num: int = 1
    pause_num: int = 1
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class ForegroundMovingBackward(CMN_FORE_BACK):
    VERT_PATH = 'foreground_moving_backward.vert'
    FRAG_PATH = 'foreground_moving_backward.frag'
    phase_num: int = 1
    pause_num: int = 2
    duration: int = 15
    CMN_phase: int = 1
    initiation = 1
    back_and_forth_num = 1


class Pause3(CMN_FORE_BACK):
    VERT_PATH = 'foreground_stationary.vert'
    FRAG_PATH = 'foreground_stationary.frag'
    phase_num: int = 2
    pause_num: int = 2
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class Pause4(CMN_FORE_BACK):
    VERT_PATH = 'stationary_reverse.vert'
    FRAG_PATH = 'stationary_reverse.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class BackgroundStationaryBackward(CMN_FORE_BACK):
    VERT_PATH = 'background_stationary_backward.vert'
    FRAG_PATH = 'background_stationary_backward.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class Pause5(CMN_FORE_BACK):
    VERT_PATH = 'pause5.vert'
    FRAG_PATH = 'pause5.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class BackgroundStationaryForward(CMN_FORE_BACK):
    VERT_PATH = 'background_stationary_foreward.vert'
    FRAG_PATH = 'background_stationary_foreward.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 15
    CMN_phase: int = 1
    initiation = 1
    back_and_forth_num = 1


class Pause6(CMN_FORE_BACK):
    VERT_PATH = 'pause6.vert'
    FRAG_PATH = 'pause6.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class BackgroundStationaryBackwardReverse(CMN_FORE_BACK):
    VERT_PATH = 'background_stationary_backward_reverse.vert'
    FRAG_PATH = 'background_stationary_backward_reverse.frag'
    phase_num: int = 3
    pause_num: int = 3
    duration: int = 15
    CMN_phase: int = 1
    initiation = 1
    back_and_forth_num = 1


class Pause7(CMN_FORE_BACK):
    VERT_PATH = 'stationary_reverse.vert'
    FRAG_PATH = 'stationary_reverse.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 1


class Pause8(CMN_FORE_BACK):
    VERT_PATH = 'foreground_stationary.vert'
    FRAG_PATH = 'foreground_stationary.frag'
    phase_num: int = 0
    pause_num: int = 0
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2


class ForegroundMovingBackwardReverse(CMN_FORE_BACK):
    VERT_PATH = 'foreground_moving_backward_reverse.vert'
    FRAG_PATH = 'foreground_moving_backward_reverse.frag'
    phase_num: int = 0
    pause_num: int = 1
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2


class Pause9(CMN_FORE_BACK):
    VERT_PATH = 'pause9.vert'
    FRAG_PATH = 'pause9.frag'
    phase_num: int = 1
    pause_num: int = 1
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2


class ForegroundMovingForwardReverse(CMN_FORE_BACK):
    VERT_PATH = 'foreground_moving_forward_reverse.vert'
    FRAG_PATH = 'foreground_moving_forward_reverse.frag'
    phase_num: int = 1
    pause_num: int = 2
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2


class ForegroundStationaryReverse(CMN_FORE_BACK):  # pause 10
    VERT_PATH = 'foreground_stationary.vert'
    FRAG_PATH = 'foreground_stationary.frag'
    phase_num: int = 2
    pause_num: int = 2
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2


class CMNReverse(CMN_FORE_BACK):
    VERT_PATH = 'CMNReverse.vert'
    FRAG_PATH = 'CMNReverse.frag'
    phase_num: int = 2
    pause_num: int = 3
    duration: int = 0
    CMN_phase: int = 0
    initiation = 1
    back_and_forth_num = 2

# class StationaryReverse(CMN_FORE_BACK):
# VERT_PATH = 'stationary_reverse.vert'
# FRAG_PATH = 'stationary_reverse.frag'
# phase_num: int = 6
# duration: int = 15
# CMN_phase: int = 1
# stationary phase = phase_num: int = 6;duration: int = 15;CMN_phase: int = 1 ; extra_cmn_phase =1; must in oder
