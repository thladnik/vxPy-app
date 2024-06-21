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

    # TypeError: temporary_dump() got an unexpected keyword argument 'cmn_ico_1000f_10tp_0.1sp_motion_matrix'
    # vxcontainer.temporary_dump(**{save_keys[0]: motion_matrix, save_keys[1]: rotated_motion_matrix})

    return [motion_matrix, rotated_motion_matrix]


def texture_stripe_h(n):
    t_list = []
    for r in range(n):
        if r%2 == 0:
            r_list = [0 ] * n
        else:
            # r_list = [1] * n
            r_list = [1 if j%2 == 0 else 0 for j in range(n)]
        t_list.append(r_list)

    t_arr = np.array(t_list)
    t_arr = t_arr[:, :, np.newaxis]
    t_arr = np.uint8(t_arr * np.array([[[1, 1, 1]]]) * 255)
    return t_arr


def texture_stripe_v(n):
    t_arr = texture_stripe_h(n)
    return t_arr.transpose([1, 0, 2])


class CMN_ForegroundBackground(vxvisual.SphericalVisual):
    time = vxvisual.FloatParameter('time', internal=True)
    # Paths to shaders
    VERT_PATH = ''
    FRAG_PATH = ''
    frame_index = vxvisual.IntParameter('frame_index', internal=True)

    # subdivision = 0
    # num_vertices = 20

    # subdivision = 1
    # num_vertices = 80

    subdivision = 2
    num_vertices = 320

    motion_frame = vxvisual.Parameter('motion_matrix', shape=(num_vertices,), dtype=np.complex_, internal=True)

    frame_num = 1_000
    tp_sigma = 10
    sp_sigma = 0.1
    stimulus_fps = 20
    norm_speed = 0.01
    stimulus_diretion = 1
    phase_num: int = None
    duration: int = None,

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        print("__init__ =======================")
        np.random.seed(1)

        # Set up 3d model of sphere
        self.cmn_parameters = {'tp_sigma': self.tp_sigma,
                               'sp_sigma': self.sp_sigma,
                               'frame_num': self.frame_num}
        self.sphere = sphere.IcoSphere_UVsphere_new(subdivisionTimes=self.subdivision)
        # print(f'{self.sphere.a_azimuth=}')
        # print(f'{self.sphere.a_elevation=}')
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)
        self.tile_center_x_buffer = gloo.VertexBuffer(self.sphere.triangle_center_x)
        self.tile_center_y_buffer = gloo.VertexBuffer(self.sphere.triangle_center_y)
        self.tile_center_inshader = gloo.VertexBuffer(self.sphere.tile_center_inshader)
        self.triangle_num_buffer = gloo.VertexBuffer(self.sphere.triangle_num)

        # cju
        # cju_position = [(-0.5, -0.5, 0.0), (-0.5, +0.5, 0.0), (+0.5, -0.5, 0.0)]
        # cju_index = [0, 1, 2]
        # self.position_buffer = gloo.VertexBuffer(cju_position)
        # self.index_buffer = gloo.IndexBuffer(cju_index)

        # Set up program
        self.rotating_dot = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))
        self.time.connect(self.rotating_dot)

        # Connect parameters (this makes them be automatically updated in the connected programs)
        face_num = self.sphere.indices.size // 3
        self.texture_start_coords = Geometry.cen2tri(np.random.rand(face_num), np.random.rand(face_num), .1)
        self.texture_start_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        self.binary_texture = np.uint8(np.random.randint(0, 2, [75, 75, 1]) * np.array([[[1, 1, 1]]]) * 255)
	# cju use fixed texture to debug
        # self.binary_texture = texture_stripe_h(75)
        self.rotating_dot['u_texture'] = self.binary_texture
        self.rotating_dot['u_texture'].wrapping = 'repeat'
        print(f'{self.binary_texture=}')




        # Set texture coordinates
        self.texture_coords = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        self.rotating_dot['a_texcoord'] = gloo.VertexBuffer(self.texture_coords)
        self.rotating_dot['a_position'] = self.position_buffer
        self.rotating_dot['a_azimuth'] = self.azimuth_buffer
        self.rotating_dot['a_elevation'] = self.elevation_buffer
        self.rotating_dot['a_tile_center_x'] = self.tile_center_x_buffer
        self.rotating_dot['a_tile_center_y'] = self.tile_center_y_buffer
        self.rotating_dot['a_tile_center_inshader'] = gloo.VertexBuffer(self.sphere.tile_center_inshader)
        self.rotating_dot['a_texture_start_coords'] = self.texture_start_coords
        self.rotating_dot['a_triangle_num'] = self.triangle_num_buffer

        self.motion_matrix, self.rotated_motion_matrix = create_motion_matrix(tile_centers=self.sphere.tile_center,
                                                                              intertile_distance=self.sphere.intertile_distance,
                                                                              tile_orientations=self.sphere.tile_orientation,
                                                                              **self.cmn_parameters)

        # Apply direction
        self.motion_matrix = self.motion_matrix[:, ::self.stimulus_diretion]
        self.rotated_motion_matrix = self.rotated_motion_matrix[:, ::self.stimulus_diretion]
        self.texture_coords_lastphase = np.float32(self.texture_start_coords.reshape([-1, 2]) / 2)
        for _ in range(self.phase_num):
            for i in range(1, self.duration * self.stimulus_fps + 1):
                motmat_eachframe = np.repeat(self.rotated_motion_matrix[:, i], 3, axis=0)
                # cju
		# fix the frame to debug
                # motmat_eachframe = np.repeat(self.rotated_motion_matrix[:, 1], 3, axis=0)
                self.texture_coords_lastphase += np.array(
                    [np.real(motmat_eachframe), np.imag(motmat_eachframe)]).T * self.norm_speed
                # self.motmat_sum = np.repeat(self.rotated_motion_matrix[:, 30*self.stimulus_fps], 3, axis=0)

        # Dump info to recording file
        # vxcontainer.dump(
        #     dict(motion_matrix=self.motion_matrix, rotated_motion_matrix=self.rotated_motion_matrix,
        #          positions=self.sphere.a_position, texture_start_coords=self.texture_start_coords,
        #          frame_num=self.frame_num, tp_sigma=self.tp_sigma, sp_sigma=self.sp_sigma,
        #          stimulus_fps=self.stimulus_fps, norm_speed=self.norm_speed,
        #          stimulus_diretion=self.stimulus_diretion, binary_texture=self.binary_texture),
        #     group=self.__class__.__name__
        # )

    def initialize(self, **params):
        print("initialize =======================")
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
        self.time.connect(self.rotating_dot)

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

	# FIX add the following line to pass time to shader
        self.time.connect(self.rotating_dot)
        print(f"=====0.1*time: {0.1*self.time.data}")

        frame_idx = int(self.time.data * self.stimulus_fps) % (self.frame_num - 1)
        # print(f"render frame_idx={frame_idx}, {self.stimulus_fps=}=======================")

        # Only move texture coordinate if this motion matrix frame wasn't used yet
        if frame_idx > self.frame_index.data[0]:
            # Save un-rotated version
            self.motion_frame.data = self.motion_matrix[:, frame_idx]

            # Update program based on motion matrix
            motmat = np.repeat(self.rotated_motion_matrix[:, frame_idx], 3, axis=0)
            self.texture_coords += np.array([np.real(motmat), np.imag(motmat)]).T * self.norm_speed
            self.rotating_dot['a_texcoord'] = self.texture_coords

        # Update index
        self.frame_index.data = frame_idx

        # Apply default transforms to the program for mapping according to hardware calibration
        # self.apply_transform(self.rotating_dot)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.rotating_dot.draw('triangles', self.index_buffer)


# class Stationary(CMN_ForegroundBackground):
#     VERT_PATH = 'stationary.vert'
#     FRAG_PATH = 'stationary.frag'
#     phase_num: int = 0
#     duration: int = 0
#
#     def render(self, dt):
#         # Add elapsed time to u_time
#         self.time.data += dt
#
#         self.rotating_dot['a_texcoord'] = self.texture_coords
#
#         # Draw the actual visual stimulus using the indices of the  triangular faces
#         self.rotating_dot.draw('triangles', self.index_buffer)


# class ForegroundStationary(CMN_ForegroundBackground):
#     VERT_PATH = 'foreground_stationary.vert'
#     FRAG_PATH = 'foreground_stationary.frag'
#     phase_num: int = 0
#     duration: int = 0
#
#
class ForegroundMovingForward(CMN_ForegroundBackground):
    VERT_PATH = 'foreground_moving_forward.vert'
    FRAG_PATH = 'foreground_moving_forward_unk.frag'
    # VERT_PATH = 'cju_foreground_moving_forward.vert'
    # FRAG_PATH = 'cju_foreground_moving_forward.frag'
    phase_num: int = 1
    duration: int = 15
#
#
# class ForegroundMovingBackward(CMN_ForegroundBackground):
#     VERT_PATH = 'foreground_moving_backward.vert'
#     FRAG_PATH = 'foreground_moving_backward.frag'
#     phase_num: int = 2
#     duration: int = 15
#
#
# class BackgroundStationaryBackward(CMN_ForegroundBackground):
#     VERT_PATH = 'background_stationary_backward.vert'
#     FRAG_PATH = 'background_stationary_backward.frag'
#     phase_num: int = 3
#     duration: int = 15
#
#     def render(self, dt):
#         # Add elapsed time to u_time
#         self.time.data += dt
#
#         self.rotating_dot['a_texcoord'] = self.texture_coords
#
#         # Draw the actual visual stimulus using the indices of the  triangular faces
#         self.rotating_dot.draw('triangles', self.index_buffer)
#
#
# class BackgroundStationaryForward(CMN_ForegroundBackground):
#     VERT_PATH = 'background_stationary_foreward.vert'
#     FRAG_PATH = 'background_stationary_foreward.frag'
#     phase_num: int = 3
#     duration: int = 15
#
#     def render(self, dt):
#         # Add elapsed time to u_time
#         self.time.data += dt
#
#         self.rotating_dot['a_texcoord'] = self.texture_coords
#
#         # Draw the actual visual stimulus using the indices of the  triangular faces
#         self.rotating_dot.draw('triangles', self.index_buffer)


if __name__ == '__main__':
    texture = texture_stripe_h(5)
    print(texture)
    # s = sphere.IcoSphere_UVsphere_new(subdivisionTimes=2)
    # print(f'{np.max(s.a_azimuth)=}')
    # print(f'{np.max(s.a_elevation)=}')
    # print(f'{np.min(s.a_azimuth)=}')
    # print(f'{np.min(s.a_elevation)=}')
