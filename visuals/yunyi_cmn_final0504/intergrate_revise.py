import os.path
from typing import List, Tuple
from vispy import gloo
import vxpy.utils.geometry as Geometry
import vxpy.core.container as vxcontainer
import vxpy.core.logger as vxlogger
import numpy as np
from scipy import signal
import vxpy.core.visual as vxvisual
from vxpy.utils import sphere
import time
import scipy
import quaternionic as qt

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


class CMN_ForegroundBackground(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    # Paths to shaders
    VERT_PATH = ''
    FRAG_PATH = ''
    frame_index = vxvisual.IntParameter('frame_index', internal=True)
    motion_frame = vxvisual.Parameter('motion_matrix', shape=(320,), dtype=np.complex_, internal=True)

    frame_num = 1_000
    tp_sigma = 10
    sp_sigma = 0.1
    stimulus_fps = 20
    norm_speed = 0.01
    stimulus_diretion = 1
    phase_num: int = None
    duration: int = None,

    sp_cr = 57.  # spatial contiguity radius [deg]
    tp_cr = 1.  # temporal contiguity radius [s]
    fps = 30  # [frames/s]
    nominal_velocity = 67  # mean local velocity [deg/s]
    motion_vector_bias = np.array([0., 0., 0.])  # Bias motion vectors (for testing)
    noise_scale = vxvisual.FloatParameter('noise_scale', default=5., limits=(.01, 50.), step_size=.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        np.random.seed(1)

        # Set up 3d model of sphere
        self.cmn_parameters = {'tp_sigma': self.tp_sigma,
                               'sp_sigma': self.sp_sigma,
                               'frame_num': self.frame_num}
        self.sphere = sphere.IcoSphere_UVsphere_new(subdivisionTimes=2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)
        self.tile_center_x_buffer = gloo.VertexBuffer(self.sphere.triangle_center_x)
        self.tile_center_y_buffer = gloo.VertexBuffer(self.sphere.triangle_center_y)
        self.tile_center_inshader = gloo.VertexBuffer(self.sphere.tile_center_inshader)
        self.triangle_num_buffer = gloo.VertexBuffer(self.sphere.triangle_num)
        self.triangle_angle_buffer = gloo.VertexBuffer(self.sphere.triangle_angle)
        # self.a_texture_foreground_cord_buffer = gloo.VertexBuffer(self.sphere.a_texture_foreground_cord)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        self.time.connect(self.rotating_dot)

        # Connect parameters (this makes them be automatically updated in the connected programs)
        face_num = self.sphere.indices.size // 3
        # self.texture_start_coords = Geometry.cen2tri(np.random.rand(face_num), np.random.rand(face_num), .1)
        self.texture_start_coords = self.sphere.a_texture_foreground_cordinate2D
        self.texture_start_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        self.binary_texture = np.uint8(np.random.randint(0, 2, [50, 50, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.rotating_dot['u_texture'] = self.binary_texture
        self.rotating_dot['u_texture'].wrapping = 'mirrored_repeat'
        self.a_texture_foreground_cordinate2D = np.float32(
            self.sphere.a_texture_foreground_cordinate2D.reshape([-1, 2]) / 2)
        # Set texture coordinates
        self.texture_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        # self.rotating_dot['a_texcoord'] = gloo.VertexBuffer(self.texture_coords)
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer
        self.rotating_dot['a_tile_center_x'] = self.tile_center_x_buffer
        self.rotating_dot['a_tile_center_y'] = self.tile_center_y_buffer
        self.rotating_dot['a_tile_center_inshader'] = gloo.VertexBuffer(self.sphere.tile_center_inshader)
        self.rotating_dot['a_texture_start_coords'] = self.texture_start_coords
        self.rotating_dot['a_triangle_num'] = self.triangle_num_buffer
        self.rotating_dot['a_triangle_angle'] = self.triangle_angle_buffer
        self.rotating_dot['a_texture_foreground_cordinate2D'] = self.a_texture_foreground_cordinate2D
        # self.rotating_dot['a_texture_foreground_cord'] = self.a_texture_foreground_cord_buffer

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
        start_rotations = np.ones((self.sphere.tile_center_inshader.shape[0], 4, 4), dtype=np.float32) * np.eye(4)[None,
                                                                                                         :, :]
        self.current_quats = qt.array.from_rotation_matrix(start_rotations)

        self.texture_coords_lastphase = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        for a in range(self.phase_num):
            for i in range(1, self.duration * self.stimulus_fps + 1):
                motmat_eachframe = np.repeat(start_rotations[:, i], 3, axis=0)
                self.texture_coords_lastphase += np.array(
                    [np.real(motmat_eachframe), np.imag(motmat_eachframe)]).T * self.norm_speed
                # self.motmat_sum = np.repeat(self.rotated_motion_matrix[:, 30*self.stimulus_fps], 3, axis=0)
        self.noise_scale.connect(self.rotating_dot)
        # Dump info to recording file
        data_dump = dict(centers=self.tile_center_inshader, vertices=self.position_buffer, indices=self.index_buffer,
                         motion_vectors=self.motion_vectors, local_motion_vectors=self.local_motion_vectors,
                         rotation_quats=self.rotation_quats)

        vxcontainer.dump(data_dump, group=self.__class__.__name__)

    def initialize(self, **params):
        # Reset time
        self.time.data = 0.0
        self.frame_index.data = 0
        # Reset texture coordinates
        self.texture_coords = self.texture_coords_lastphase
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer
        self.rotating_dot['a_texcoord_fore'] = self.texture_start_coords
        self.rotating_dot['a_tile_center_x'] = self.tile_center_x_buffer
        self.rotating_dot['a_tile_center_y'] = self.tile_center_y_buffer
        self.rotating_dot['a_tile_center_inshader'] = gloo.VertexBuffer(self.sphere.tile_center_inshader)
        self.rotating_dot['a_texture_start_coords'] = self.texture_start_coords
        self.rotating_dot['a_triangle_num'] = self.triangle_num_buffer
        self.rotating_dot['a_triangle_angle'] = self.triangle_angle_buffer
        # self.rotating_dot['a_texture_foreground_cord'] = self.a_texture_foreground_cord_buffer
        self.rotating_dot['a_texture_foreground_cordinate2D'] = self.a_texture_foreground_cordinate2D
        self.time.connect(self.rotating_dot)

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        frame_idx = int(self.time.data * self.stimulus_fps) % (self.frame_num - 1)

        self.frame_index.data = frame_idx

        # Calculate future rotations (for t += 1 second)
        future_quats = self.rotation_quats[frame_idx] * self.current_quats

        # SLERP current rotations
        self.current_quats = qt.slerp(self.current_quats, future_quats, dt)

        # Stack for write
        stacked_quats = np.repeat(self.current_quats[:, None], 3, axis=0).reshape((-1, 4)).astype(np.float32)

        # Write to program
        self.rotating_dot['a_rotation'] = stacked_quats

        # Update index


        # Apply default transforms to the program for mapping according to hardware calibration
        # self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)


class Stationary(CMN_ForegroundBackground):
    VERT_PATH = 'stationary.vert'
    FRAG_PATH = 'stationary.frag'
    phase_num: int = 0
    duration: int = 0

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        self.rotating_dot['a_texcoord'] = self.texture_coords

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)


class ForegroundStationary(CMN_ForegroundBackground):
    VERT_PATH = 'foreground_stationary.vert'
    FRAG_PATH = 'foreground_stationary.frag'
    phase_num: int = 0
    duration: int = 0


class ForegroundMovingForward(CMN_ForegroundBackground):
    VERT_PATH = 'foreground_moving_forward.vert'
    FRAG_PATH = 'foreground_moving_forward.frag'
    phase_num: int = 1
    duration: int = 15


class ForegroundMovingBackward(CMN_ForegroundBackground):
    VERT_PATH = 'foreground_moving_backward.vert'
    FRAG_PATH = 'foreground_moving_backward.frag'
    phase_num: int = 2
    duration: int = 15


class BackgroundStationaryBackward(CMN_ForegroundBackground):
    VERT_PATH = 'background_stationary_backward.vert'
    FRAG_PATH = 'background_stationary_backward.frag'
    phase_num: int = 3
    duration: int = 15

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        self.rotating_dot['a_texcoord'] = self.texture_coords

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)


class BackgroundStationaryForward(CMN_ForegroundBackground):
    VERT_PATH = 'background_stationary_foreward.vert'
    FRAG_PATH = 'background_stationary_foreward.frag'
    phase_num: int = 3
    duration: int = 15

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        self.rotating_dot['a_texcoord'] = self.texture_coords

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)
