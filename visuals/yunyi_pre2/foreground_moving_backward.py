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

log = vxlogger.getLogger(__name__)


def create_motion_matrix(tile_centers: np.ndarray,
                         intertile_distance: float,
                         frame_num: int,
                         tile_orientations: np.ndarray = None,
                         tp_sigma: int = 20,
                         sp_sigma: float = 0.1) -> List[np.ndarray]:
    """Create a motion matrix of given parameters
    """

    # Define temp keys
    save_name = f'cmn_ico_{frame_num}f_{tp_sigma}tp_{sp_sigma}sp'
    save_keys = [f'{save_name}_motion_matrix', f'{save_name}_rotated_motion_matrix']

    if vxcontainer.temporary_exists(*save_keys):
        return vxcontainer.temporary_load(*save_keys)

    # Create flow vectors for each face and frame and normalize them
    flow_vec = np.random.normal(size=(len(tile_centers), frame_num, 3))  # Random white noise motion vector
    flow_vec /= Geometry.vecNorm(flow_vec)[:, :, np.newaxis]

    # Temporal smoothing
    tp_min_length = np.int64(np.ceil(np.sqrt(-2 * tp_sigma ** 2 * np.log(.01 * tp_sigma * np.sqrt(2 * np.pi)))))
    tp_kernel = np.linspace(-tp_min_length, tp_min_length, num=2 * tp_min_length + 1)
    tp_kernel = 1 / (tp_sigma * np.sqrt(2 * np.pi)) * np.exp(-tp_kernel ** 2 / (2 * tp_sigma ** 2))
    tp_kernel *= tp_kernel > .0001
    tp_smooth_x = signal.convolve(flow_vec[:, :, 0], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_y = signal.convolve(flow_vec[:, :, 1], tp_kernel[np.newaxis, :], mode='same')
    tp_smooth_z = signal.convolve(flow_vec[:, :, 2], tp_kernel[np.newaxis, :], mode='same')

    # Spatial smoothing
    sp_kernel = np.exp(-(intertile_distance ** 2) / (2 * sp_sigma ** 2))
    sp_kernel *= sp_kernel > .001

    sp_smooth_x = np.dot(sp_kernel, tp_smooth_x)
    sp_smooth_y = np.dot(sp_kernel, tp_smooth_y)
    sp_smooth_z = np.dot(sp_kernel, tp_smooth_z)
    sp_smooth_q = Geometry.qn(np.array([sp_smooth_x, sp_smooth_y, sp_smooth_z]).transpose([1, 2, 0]))

    tile_cen_q = Geometry.qn(tile_centers)

    # Take face/tile orientation into account (for actual stimulus display)
    if tile_orientations is not None:
        tile_ori_q1 = Geometry.qn(np.real(tile_orientations)).normalize[:, None]
        tile_ori_q2 = Geometry.qn(np.imag(tile_orientations)).normalize[:, None]
        projected_motmat = Geometry.projection(tile_cen_q[:, None], sp_smooth_q)
        rotated_motion_matrix = Geometry.qdot(tile_ori_q1, projected_motmat) \
                                - 1.j * Geometry.qdot(tile_ori_q2, projected_motmat)

    # Map to horizontal/vertical axes in tile plane (for analysis and illustration)
    projected_motmat = Geometry.projection(tile_cen_q[:, np.newaxis], sp_smooth_q)
    tile_up_vec = Geometry.projection(tile_cen_q, Geometry.qn([0, 0, 1])).normalize
    tile_hori_vec = Geometry.qcross(tile_cen_q, tile_up_vec).normalize
    motion_matrix = Geometry.qdot(tile_up_vec[:, np.newaxis], projected_motmat) \
                    - 1.j * Geometry.qdot(tile_hori_vec[:, np.newaxis], projected_motmat)
    if tile_orientations is None:
        rotated_motion_matrix = np.copy(motion_matrix)

    # Plot reference directions for motion matrix projection:
    # fig = plt.figure()
    # ax = plt.subplot(projection='3d')
    # ax.quiver(*centers.T, *tile_up_vec.matrixform[:, 1:].T / 5, color='green', label='vertical')
    # ax.quiver(*centers.T, *tile_hori_vec.matrixform[:, 1:].T / 5, color='red', label='horizontal')
    # ax.quiver(*centers.T, *sp_smooth_q.matrixform[:,50, 1:].T / 5, color='black', label='motion')
    # fig.legend()
    # plt.show()

    vxcontainer.temporary_dump(**{save_keys[0]: motion_matrix, save_keys[1]: rotated_motion_matrix})

    return [motion_matrix, rotated_motion_matrix]


class ForegroundMovingBackward2(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    # Paths to shaders
    VERT_PATH = 'foreground_moving_backward.vert'
    FRAG_PATH = 'foreground_moving_backward.frag'
    frame_index = vxvisual.IntParameter('frame_index', internal=True)
    motion_frame = vxvisual.Parameter('motion_matrix', shape=(320,), dtype=np.complex_, internal=True)

    frame_num = 1_000
    tp_sigma = 10
    sp_sigma = 0.1
    stimulus_fps = 20
    norm_speed = 0.01
    stimulus_diretion = 1

    # center = geometry.sph2cart(5 * np.pi / 9, -np.pi / 6, 1.)
    # distance_verticle_pos = geometry.sph2cart(5 * np.pi / 9, 0, 1.)
    # distance_horizontal_pos = geometry.sph2cart(13 * np.pi / 18, -np.pi / 6, 1.)
    # distance_verticle = distance_verticle_pos[2] - center[2]
    # distance_horizontal = distance_horizontal_pos[1] - center[1]

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

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        self.time.connect(self.rotating_dot)

        # Connect parameters (this makes them be automatically updated in the connected programs)
        face_num = self.sphere.indices.size // 3
        self.texture_start_coords = Geometry.cen2tri(np.random.rand(face_num), np.random.rand(face_num), .1)
        self.texture_start_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        # num = self.sphere.azim_lvls * self.sphere.elev_lvls
        # self.texture_coords = Geometry.cen2tri(np.random.rand(num), np.random.rand(num))
        # self.texture_start_coords = self.sphere.get_uv_coordinates()
        # self.texture_start_coords = Geometry.cen2tri(np.random.rand(num//3), np.random.rand(num//3),.1)
        # self.texture_start_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        # self.time.connect(self.rotating_dot)
        # self.distance_verticle.connect(self.rotating_dot)
        # self.distance_horizontal.connect(self.rotating_dot)
        # self.center.connect(self.rotating_dot)
        # Set texture
        self.binary_texture = np.uint8(np.random.randint(0, 2, [75, 75, 1]) * np.array([[[1, 1, 1]]]) * 255)
        # self.binary_texture = np.uint8(np.random.randint(0, 2, [100, 100, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.rotating_dot['u_texture'] = self.binary_texture
        self.rotating_dot['u_texture'].wrapping = 'repeat'

        # Set texture coordinates
        self.texture_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        # self.rotating_dot['a_texcoord'] = gloo.VertexBuffer(self.texture_coords)
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer

        # Create motion matrix to save to file for analysis
        self.motion_matrix, self.rotated_motion_matrix = create_motion_matrix(tile_centers=self.sphere.tile_center,
                                                                              intertile_distance=self.sphere.intertile_distance,
                                                                              tile_orientations=self.sphere.tile_orientation,
                                                                              **self.cmn_parameters)

        # Apply direction
        self.motion_matrix = self.motion_matrix[:, ::self.stimulus_diretion]
        self.rotated_motion_matrix = self.rotated_motion_matrix[:, ::self.stimulus_diretion]

        # Dump info to recording file
        vxcontainer.dump(
            dict(motion_matrix=self.motion_matrix, rotated_motion_matrix=self.rotated_motion_matrix,
                 positions=self.sphere.a_position, texture_start_coords=self.texture_start_coords,
                 frame_num=self.frame_num, tp_sigma=self.tp_sigma, sp_sigma=self.sp_sigma,
                 stimulus_fps=self.stimulus_fps, norm_speed=self.norm_speed,
                 stimulus_diretion=self.stimulus_diretion, binary_texture=self.binary_texture),
            group=self.__class__.__name__
        )

    def initialize(self, **params):
        # Reset time
        self.time.data = 0.0
        self.frame_index.data = 0
        # Reset texture coordinates
        self.texture_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        # Iout = np.arange(self.sphere.azim_lvls * self.sphere.elev_lvls, dtype=np.uint32)
        # self.texture_coords = Geometry.cen2tri(np.random.rand(int(Iout.size / 3)), np.random.rand(int(Iout.size / 3)),
        # .1).reshape([Iout.size, 2])
        # Set positions with buffers
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer
        self.rotating_dot['a_texcoord_fore'] = self.texture_start_coords
        self.time.connect(self.rotating_dot)
    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        frame_idx = int(self.time.data * self.stimulus_fps) % (self.frame_num - 1)

        # Only move texture coordinate if this motion matrix frame wasn't used yet
        if frame_idx > self.frame_index.data[0]:
            # Save un-rotated version
            self.motion_frame.data = self.motion_matrix[:, frame_idx]

            # Update program based on motion matrix
            motmat = np.repeat(self.rotated_motion_matrix[:, frame_idx], 3, axis=0)
            # num = self.sphere.azim_lvls * self.sphere.elev_lvls
            self.texture_coords += np.array([np.real(motmat), np.imag(motmat)]).T * self.norm_speed
            self.rotating_dot['a_texcoord'] = self.texture_coords

        # Update index
        self.frame_index.data = frame_idx

        # Apply default transforms to the program for mapping according to hardware calibration
        # self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)
