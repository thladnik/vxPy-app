from __future__ import annotations
import h5py
import numpy as np
import scipy.io
from vispy import gloo
from vispy.util import transforms
from vxpy.core import visual
from vxpy.utils import sphere

import vxpy.core.visual as vxvisual


def _convert_mat_texture_to_hdf(mat_path: str, hdf_path: str):
    d = scipy.io.loadmat(mat_path)
    x, y, z = d['grid']['x'][0][0], \
              d['grid']['y'][0][0], \
              d['grid']['z'][0][0]

    v = np.array([x.flatten(), y.flatten(), z.flatten()])
    vertices = np.ascontiguousarray(v.T, dtype=np.float32)

    idcs = list()
    azim_lvls = x.shape[1]
    elev_lvls = x.shape[0]
    for i in np.arange(elev_lvls):
        for j in np.arange(azim_lvls):
            idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
            idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
    indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)
    indices = indices[indices < azim_lvls * elev_lvls]

    states = np.ascontiguousarray(d['totalintensity'].flatten(), dtype=np.float32)

    with h5py.File(hdf_path, 'w') as f:
        f.create_dataset('vertices', data=vertices)
        f.create_dataset('indices', data=indices)
        f.create_dataset('intensity', data=states)


def _import_texture_from_hdf(texture_file):
    with h5py.File(texture_file, 'r') as f:
        vertices = f['vertices'][:]
        indices = f['indices'][:]
        intensity = f['intensity'][:]

    return vertices, indices, intensity


class MotionAxis(visual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        visual.Mat4Parameter.__init__(self, *args, **kwargs)

        self.value_map = {'forward': self._rotate_forward,
                          'sideways': self._rotate_sideways,
                          'vertical': np.eye(4)}

    @staticmethod
    def _rotate_forward():
        return transforms.rotate(90, (0, 1, 0))

    @staticmethod
    def _rotate_sideways():
        return transforms.rotate(90, (1, 0, 0))


class RotatingTexture2000(vxvisual.SphericalVisual):

    VERT_LOC = './ml_texture.vert'
    FRAG_LOC = './ml_rotating_texture.frag'

    # define general parameters
    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # define texture displacement parameters
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    texture_default = vxvisual.Attribute('texture_default', static=True)
    luminance = vxvisual.FloatParameter('luminance', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    rotation_start_time = vxvisual.IntParameter('rotation_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    rotation_duration = vxvisual.IntParameter('rotation_duration', static=True, default=10000, limits=(20, 200), step_size=10)  # ms
    rotation_target_angle = vxvisual.FloatParameter('rotation_target_angle', static=True, default=300., limits=(-90.0, 90.0), step_size=0.01)  # deg

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'


    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.simu_sacc = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.simu_sacc['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters to rendering program
        self.time.connect(self.simu_sacc)
        self.rotation.connect(self.simu_sacc)
        self.luminance.connect(self.simu_sacc)
        self.contrast.connect(self.simu_sacc)
        self.texture_default.connect(self.simu_sacc)

        # Global azimuth angle
        self.protocol.global_visual_props['azim_angle'] = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        rot_start_time = self.rotation_start_time.data[0]
        rot_duration = self.rotation_duration.data[0]
        rot_target_angle = self.rotation_target_angle.data[0]

        time_in_rotation = time - rot_start_time
        if 0.0 < time_in_rotation <= rot_duration:
            # Calculate rotation
            angle = rot_target_angle * 1000 * dt / rot_duration
            self.protocol.global_visual_props['azim_angle'] += angle


        # Set rotation
        self.rotation.data = transforms.rotate(self.protocol.global_visual_props['azim_angle'], (0, 0, 1))

        # Set luminance
        baseline_lum = self.luminance.data[0]
        self.luminance.data = baseline_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)
