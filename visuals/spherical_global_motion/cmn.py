"""Global contiguous motion noise on spherical surface
Original author: Yue Zhang
"""
import os.path
from typing import List, Tuple

import numpy as np
from scipy import signal
from vispy import gloo
import vxpy.utils.geometry as Geometry

import vxpy.core.container as vxcontainer
import vxpy.core.logger as vxlogger
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
    tile_ori_q1 = Geometry.qn(np.real(tile_orientations)).normalize[:, None]
    tile_ori_q2 = Geometry.qn(np.imag(tile_orientations)).normalize[:, None]
    projected_motmat = Geometry.projection(tile_cen_q[:, None], sp_smooth_q)
    rotated_motion_matrix = Geometry.qdot(tile_ori_q1, projected_motmat) \
                            - 1.j * Geometry.qdot(tile_ori_q2,
                                                  projected_motmat)

    # Map to horizontal/vertical axes in tile plane (for analysis and illustration)
    projected_motmat = Geometry.projection(tile_cen_q[:, np.newaxis], sp_smooth_q)
    tile_up_vec = Geometry.projection(tile_cen_q, Geometry.qn([0, 0, 1])).normalize
    tile_hori_vec = Geometry.qcross(tile_cen_q, tile_up_vec).normalize
    motion_matrix = Geometry.qdot(tile_up_vec[:, np.newaxis], projected_motmat) \
                    - 1.j * Geometry.qdot(tile_hori_vec[:, np.newaxis], projected_motmat)

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


class ContiguousMotionNoise(vxvisual.SphericalVisual):
    """Spherical contiguous motion noise stimulus (originally created by Yue Zhang)
    """

    time = vxvisual.FloatParameter('time', internal=True)
    frame_index = vxvisual.IntParameter('frame_index', internal=True)

    frame_num = 1_000
    tp_sigma = 10
    sp_sigma = 0.1
    stimulus_fps = 20
    norm_speed = 0.01
    stimulus_diretion = 1

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args)

        np.random.seed(1)

        self.cmn_parameters = {'tp_sigma': self.tp_sigma,
                               'sp_sigma': self.sp_sigma,
                               'frame_num': self.frame_num}

        # Create program
        vert = self.load_vertex_shader('./position_transform.vert')
        frag = self.load_shader('./texture_map.frag')
        self.sphere_program = gloo.Program(vert, frag)

        # Create sphere
        self.sphere = sphere.CMNIcoSphere(subdivisionTimes=2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(np.float32(self.sphere.a_position))

        face_num = self.sphere.indices.size // 3
        self._texcoord: np.ndarray = None
        self.startpoints = Geometry.cen2tri(np.random.rand(face_num), np.random.rand(face_num), .1)
        self.sphere_program['a_position'] = self.position_buffer

        # Set texture
        self.binary_texture = np.uint8(np.random.randint(0, 2, [100, 100, 1]) * np.array([[[1, 1, 1]]]) * 255)
        self.sphere_program['u_texture'] = self.binary_texture
        self.sphere_program['u_texture'].wrapping = 'repeat'

        # Set texture coordinates
        self._texcoord = np.float32(self.startpoints.reshape([-1, 2]) / 2)
        self.sphere_program['a_texcoord'] = gloo.VertexBuffer(self._texcoord)

        # Create motion matrix to save to file for analysis
        self.motion_matrix, self.rotated_motion_matrix = create_motion_matrix(tile_centers=self.sphere.tile_center,
                                                                              intertile_distance=self.sphere.intertile_distance,
                                                                              tile_orientations=self.sphere.tile_orientation,
                                                                              **self.cmn_parameters)

        # Apply direction
        self.motion_matrix = self.motion_matrix[:, ::self.stimulus_diretion]
        self.rotated_motion_matrix = self.rotated_motion_matrix[:, ::self.stimulus_diretion]

        # Dump to recording file
        vxcontainer.dump(motion_matrix=self.motion_matrix, rotated_motion_matrix=self.rotated_motion_matrix)

    def initialize(self, **params):
        # Reset time
        self.time.data = 0.0
        self.frame_index.data = 0

        # Reset texture coordinates
        self._texcoord = np.float32(self.startpoints.reshape([-1, 2]) / 2)
        # self.sphere_program['a_texcoord'] = gloo.VertexBuffer(self._texcoord)
        self.sphere_program['a_texcoord'] = self._texcoord

    def render(self, dt):
        self.time.data = self.time.data + dt
        frame_idx = int(self.time.data * self.stimulus_fps) % (self.frame_num - 1)

        # Only move texture coordinate if this motion matrix frame wasn't used yet
        if frame_idx > self.frame_index.data[0]:
            motmat = np.repeat(self.rotated_motion_matrix[:, frame_idx], 3, axis=0)
            self._texcoord += np.array([np.real(motmat), np.imag(motmat)]).T * self.norm_speed
            self.sphere_program['a_texcoord'] = self._texcoord

        # Update index
        self.frame_index.data = frame_idx

        # Render
        self.sphere_program.draw('triangles', self.index_buffer)


class CMN_100_000f_20fps_10tp_0p1sp(ContiguousMotionNoise):
    frame_num = 100_000
    tp_sigma = 10
    sp_sigma = 0.1
    stimulus_fps = 20
    norm_speed = 0.01
    stimulus_direction = 1


class CMN_15_000f_15fps_10tp_0p1sp_varns(ContiguousMotionNoise):
    frame_num = 15_000
    tp_sigma = 15
    sp_sigma = 0.1
    stimulus_fps = 15
    stimulus_diretion = 1
    normalized_speed = vxvisual.FloatParameter('normalized_speed', limits=(0., 1.), default=0.02, step_size=0.001)

    def render(self, dt):
        self.norm_speed = self.normalized_speed.data[0]
        ContiguousMotionNoise.render(self, dt)


class CMN_15_000f_15fps_10tp_0p1sp_0p03ns(ContiguousMotionNoise):
    frame_num = 15_000
    tp_sigma = 15
    sp_sigma = 0.1
    stimulus_fps = 15
    norm_speed = 0.03
    stimulus_diretion = 1




class CMN_15_000f_15fps_10tp_0p1sp_0p03ns_inv(CMN_15_000f_15fps_10tp_0p1sp_0p03ns):
    stimulus_direction = -1


if __name__ == '__main__':

    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.mplot3d import Axes3D

    from vxpy.utils import geometry

    np.random.seed(1)

    create_vector_video = True
    create_polar_histogram = False

    frame_num = 2_000
    tp_sigma = 10
    sp_sigma = 0.1


    def despine(axis, spines=None, hide_ticks=True):
        def hide_spine(spine):
            spine.set_visible(False)

        for spine in axis.spines.keys():
            if spines is not None:
                if spine in spines:
                    hide_spine(axis.spines[spine])
            else:
                hide_spine(axis.spines[spine])

        if hide_ticks:
            axis.xaxis.set_ticks([])
            axis.yaxis.set_ticks([])


    # Create sphere model
    sphere_model = sphere.CMNIcoSphere(subdivisionTimes=2)
    centers = sphere_model.tile_center
    tile_orientations = sphere_model.tile_orientation
    intertile_distance = sphere_model.intertile_distance

    # Create motion matrix
    motmat = create_motion_matrix(centers, intertile_distance,
                                  frame_num=frame_num, tp_sigma=tp_sigma, sp_sigma=sp_sigma)

    # Calculate 2d mapped positions
    azims, elevs, r = geometry.cart2sph(*centers.T)

    # Calculate phase angles
    angles = np.angle(motmat)

    # 2D direction vectors
    U, V = np.real(motmat), np.imag(motmat)

    if create_vector_video:
        print('Create vector field animation')


        def scale_uv_vectors(u, v):
            return u, v


        def animate(tidx, qr):
            if tidx % 100 == 0:
                print(f'{tidx}/{motmat.shape[1]}')

            qr.set_UVC(*scale_uv_vectors(U[:, tidx], V[:, tidx]))


        fig = plt.figure(figsize=(10, 5), dpi=200)
        ax = plt.subplot()
        ax.set_title(f'frames: {frame_num}, sp_sigma: {sp_sigma}, tp_sigma: {tp_sigma}')
        ax.set_aspect(1)
        qr = ax.quiver(azims, elevs, *scale_uv_vectors(U[:, 0], V[:, 0]),
                       pivot='middle', scale=15, width=0.0015, headlength=4.5)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels([-180, -90, 0, 90, 180])
        ax.set_xlabel('azimuth [deg]')
        ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
        ax.set_yticklabels([-90, -45, 0, 45, 90])
        ax.set_ylabel('elevation [deg]')

        fig.tight_layout()

        ani = animation.FuncAnimation(fig, animate, fargs=(qr,), interval=50, blit=False, frames=motmat.shape[1])
        ani.save(f'./motion_vectors_f{frame_num}_sp{sp_sigma}_tp{tp_sigma}.mp4', writer='ffmpeg')

    if create_polar_histogram:
        print('Create polar histograms')

        inset_size = 0.30  # inch

        fig3d1 = plt.figure()
        ax3d1 = plt.subplot(projection='3d')
        ax3d1.scatter(*centers.T)

        fig2 = plt.figure(figsize=(14, 7))
        ax2 = plt.subplot()
        ax2.set_facecolor('black')
        ax2.set_aspect(1)
        ax2.scatter(azims, elevs, s=5)
        fig2.tight_layout()

        inaxes = []
        for i, (az, el) in enumerate(zip(azims, elevs)):
            axins = inset_axes(ax2, width=inset_size, height=inset_size, loc='center',
                               bbox_to_anchor=(az, el, 0., 0.),
                               bbox_transform=ax2.transData, borderpad=0,
                               axes_class=matplotlib.projections.get_projection_class('polar'))

            inaxes.append(axins)
            axins.set_rticks([])
            despine(axins, hide_ticks=False)
            axins.set_xticks([])
            axins.set_facecolor('black')

            # Plot polar hist
            counts, bins = np.histogram(angles[i, :], bins=20)
            props = counts / counts.max()
            bin_centers = bins[1:] - (bins[1] - bins[0])
            bars = axins.bar(bin_centers, props, width=0.3)

            # Use custom colors and opacity
            for r, bar in zip(props, bars):
                bar.set_facecolor(plt.cm.viridis(r))
                bar.set_alpha(0.5)

    plt.show()
