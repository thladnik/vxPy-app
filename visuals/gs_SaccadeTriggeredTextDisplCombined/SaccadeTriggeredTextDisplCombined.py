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


class Luminance(vxvisual.FloatParameter):
    def __init__(self, *args, **kwargs):
        vxvisual.FloatParameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):

        time = TextureRotationCosineFlash.time.data[0]
        lum_amp = TextureRotationCosineFlash.flash_amp.data[0]
        sine_freq = TextureRotationCosineFlash.flash_freq.data[0]
        lum_mean = TextureRotationCosineFlash.baseline_lum.data[0]

        lum = lum_mean + np.sin(sine_freq * time * 2.0 * np.pi) * lum_amp
        self.data = lum


class TextureRotationCosineFlash(vxvisual.SphericalVisual):     # for coarse texture only!

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_rotation_and_cosflash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    flash_start_time = vxvisual.IntParameter('flash_start_time', static=True, default=4000, limits=(0.0, 5000), step_size=100)  # ms
    flash_duration = vxvisual.IntParameter('flash_duration', static=True, default=500, limits=(0.0, 5000), step_size=100)  # ms
    flash_amp = vxvisual.FloatParameter('flash_amp', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    flash_freq = vxvisual.FloatParameter('flash_freq', static=True, default=2.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    rotation_start_time = vxvisual.IntParameter('rotation_start_time', static=True, default=4000, limits=(0.0, 5000), step_size=100)  # ms
    rotation_duration = vxvisual.IntParameter('rotation_duration', static=True, default=100, limits=(20, 200), step_size=10)  # ms
    rotation_amplitude = vxvisual.FloatParameter('rotation_amplitude', static=True, default=30, limits=(0, 360), step_size=1)  # deg
    rotation_direction = vxvisual.IntParameter('rotation_direction', static=True, default=1, limits=(-1, 1),
                                                 step_size=1)  # deg

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

        # Connect update connections
        # self.time.add_downstream_link(self.rotation)
        # self.time.add_downstream_link(self.luminance)
        self.protocol.global_visual_props['azim_angle'] = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        rot_start_time = self.rotation_start_time.data[0]
        roation_duration = self.rotation_duration.data[0]
        rotation_amplitude = self.rotation_amplitude.data[0]
        rotation_direction = self.rotation_direction.data[0]

        time_in_saccade = time - rot_start_time
        if 0.0 < time_in_saccade <= roation_duration:
            # Calculate rotation
            angle = rotation_direction * rotation_amplitude * 1000 * dt / roation_duration
            self.protocol.global_visual_props['azim_angle'] += angle

        # Set rotation (to keep track of azimuth position of texture across phases)
        self.rotation.data = transforms.rotate(self.protocol.global_visual_props['azim_angle'], (0, 0, 1))

        # Sine "flash"
        flash_start_time = self.flash_start_time.data[0]
        flash_duration = self.flash_duration.data[0]
        flash_freq = self.flash_freq.data[0]
        flash_amp = self.flash_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_flash = time - flash_start_time
        if 0.0 < time_in_flash <= flash_duration:
            # current_lum = baseline_lum + np.sin(flash_freq * time_in_flash / 1000 * 2.0 * np.pi) * flash_amp / 2.0
            current_lum = (baseline_lum - (flash_amp / 2)) + np.cos(
                flash_freq * time_in_flash / 1000 * 2.0 * np.pi) * flash_amp / 2.0
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)
