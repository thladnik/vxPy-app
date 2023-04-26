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


################
# TEXTURE ROTATING IN AZIMUTH

# class SaccadicRotation(vxvisual.Mat4Parameter):
#
#     def __init__(self, *args, **kwargs):
#         vxvisual.Mat4Parameter.__init__(self, *args, **kwargs)
#
#     def upstream_updated(self):
#
#         rotation = self.data.copy()
#
#         self.data = rotation
#
#         # Get times
#         time = SimuSaccadeWithSineFlash.time.data[0] * 1000  # s -> ms
#         sacc_start_time = SimuSaccadeWithSineFlash.saccade_start_time.data[0]
#
#         # Has saccade started?
#         time_in_saccade = time - sacc_start_time
#         if time_in_saccade <= 0.0:
#             return
#
#         saccade_duration = SimuSaccadeWithSineFlash.saccade_duration.data[0]
#         saccade_target_angle = SimuSaccadeWithSineFlash.saccade_target_angle.data[0]
#
#         # Has saccade ended?
#         if saccade_duration < time_in_saccade:
#             return
#
#         # Calculate rotation
#         angle = saccade_target_angle * time_in_saccade / saccade_duration
#         print(angle)
#         rotation = transforms.rotate(angle, (0, 0, 1))
#
#         # Set rotation
#         self.data = rotation

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

        time = SimuSaccadeWithSineFlash1000.time.data[0]
        lum_amp = SimuSaccadeWithSineFlash1000.sine_amp.data[0]
        sine_freq = SimuSaccadeWithSineFlash1000.sine_freq.data[0]
        lum_mean = SimuSaccadeWithSineFlash1000.baseline_lum.data[0]

        lum = lum_mean + np.sin(sine_freq * time * 2.0 * np.pi) * lum_amp
        self.data = lum


class SimuSaccadeWithSineFlash1000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000), step_size=100)  # ms
    sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000), step_size=100)  # ms
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=1.0, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200), step_size=10)  # ms
    saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15., limits=(-90.0, 90.0), step_size=0.01)  # deg

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_1000_blobs.hdf5'

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
        self.azim_angle = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        sacc_start_time = self.saccade_start_time.data[0]
        saccade_duration = self.saccade_duration.data[0]
        saccade_target_angle = self.saccade_target_angle.data[0]

        time_in_saccade = time - sacc_start_time
        if 0.0 < time_in_saccade <= saccade_duration:
            # Calculate rotation
            angle = saccade_target_angle * 1000 * dt / saccade_duration
            self.azim_angle += angle

        # Set rotation
        self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

        # Sine "flash"
        sine_start_time = self.sine_start_time.data[0]
        sine_duration = self.sine_duration.data[0]
        sine_freq = self.sine_freq.data[0]
        sine_amp = self.sine_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_sine = time - sine_start_time
        if 0.0 < time_in_sine <= sine_duration:
            current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)

        class SimuSaccadeWithSineFlash1000(vxvisual.SphericalVisual):

            VERT_LOC = './gs_texture.vert'
            FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

            time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

            # Varying parameters
            luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
            # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
            rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

            # Static (set in program)
            texture_default = vxvisual.Attribute('texture_default', static=True)

            # Static (not set in rendering program)
            baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0),
                                                   step_size=0.01)
            # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
            contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                               step_size=0.01)  # Absolute contrast
            sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000),
                                                    step_size=100)  # ms
            sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000),
                                                  step_size=100)  # ms
            sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0),
                                               step_size=0.01)  # total lum range
            sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0),
                                                step_size=0.1)  # Hz
            saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500,
                                                       limits=(0.0, 5000), step_size=100)  # ms
            saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200),
                                                     step_size=10)  # ms
            saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15.,
                                                           limits=(-90.0, 90.0), step_size=0.01)  # deg

            texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_1000_blobs.hdf5'

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
                self.azim_angle = 0.

            def initialize(self, **kwargs):
                self.time.data = 0.

            def render(self, dt):
                self.time.data += dt

                # Get times
                time = self.time.data[0] * 1000  # s -> ms

                # Saccade
                sacc_start_time = self.saccade_start_time.data[0]
                saccade_duration = self.saccade_duration.data[0]
                saccade_target_angle = self.saccade_target_angle.data[0]

                time_in_saccade = time - sacc_start_time
                if 0.0 < time_in_saccade <= saccade_duration:
                    # Calculate rotation
                    angle = saccade_target_angle * 1000 * dt / saccade_duration
                    self.azim_angle += angle

                # Set rotation
                self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

                # Sine "flash"
                sine_start_time = self.sine_start_time.data[0]
                sine_duration = self.sine_duration.data[0]
                sine_freq = self.sine_freq.data[0]
                sine_amp = self.sine_amp.data[0]
                baseline_lum = self.baseline_lum.data[0]

                time_in_sine = time - sine_start_time
                if 0.0 < time_in_sine <= sine_duration:
                    #current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                    current_lum = (baseline_lum - (sine_amp/2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                else:
                    current_lum = baseline_lum

                # Set luminance
                self.luminance.data = current_lum

                self.apply_transform(self.simu_sacc)
                self.simu_sacc.draw('triangles', indices=self.index_buffer)


class SimuSaccadeWithSineFlash2000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000), step_size=100)  # ms
    sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000), step_size=100)  # ms
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=1.0, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200), step_size=10)  # ms
    saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15., limits=(-90.0, 90.0), step_size=0.01)  # deg

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
        self.azim_angle = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        sacc_start_time = self.saccade_start_time.data[0]
        saccade_duration = self.saccade_duration.data[0]
        saccade_target_angle = self.saccade_target_angle.data[0]

        time_in_saccade = time - sacc_start_time
        if 0.0 < time_in_saccade <= saccade_duration:
            # Calculate rotation
            angle = saccade_target_angle * 1000 * dt / saccade_duration
            self.protocol.global_visual_props['azim_angle'] += angle

        # Set rotation
        self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

        # Sine "flash"
        sine_start_time = self.sine_start_time.data[0]
        sine_duration = self.sine_duration.data[0]
        sine_freq = self.sine_freq.data[0]
        sine_amp = self.sine_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_sine = time - sine_start_time
        if 0.0 < time_in_sine <= sine_duration:
            #current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = (baseline_lum - (sine_amp/2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)

        class SimuSaccadeWithSineFlash2000(vxvisual.SphericalVisual):

            VERT_LOC = './gs_texture.vert'
            FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

            time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

            # Varying parameters
            luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
            # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
            rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

            # Static (set in program)
            texture_default = vxvisual.Attribute('texture_default', static=True)

            # Static (not set in rendering program)
            baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0),
                                                   step_size=0.01)
            # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
            contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                               step_size=0.01)  # Absolute contrast
            sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000),
                                                    step_size=100)  # ms
            sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000),
                                                  step_size=100)  # ms
            sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0),
                                               step_size=0.01)  # total lum range
            sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0),
                                                step_size=0.1)  # Hz
            saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500,
                                                       limits=(0.0, 5000), step_size=100)  # ms
            saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200),
                                                     step_size=10)  # ms
            saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15.,
                                                           limits=(-90.0, 90.0), step_size=0.01)  # deg

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
                self.azim_angle = 0.

            def initialize(self, **kwargs):
                self.time.data = 0.

            def render(self, dt):
                self.time.data += dt

                # Get times
                time = self.time.data[0] * 1000  # s -> ms

                # Saccade
                sacc_start_time = self.saccade_start_time.data[0]
                saccade_duration = self.saccade_duration.data[0]
                saccade_target_angle = self.saccade_target_angle.data[0]

                time_in_saccade = time - sacc_start_time
                if 0.0 < time_in_saccade <= saccade_duration:
                    # Calculate rotation
                    angle = saccade_target_angle * 1000 * dt / saccade_duration
                    self.azim_angle += angle

                # Set rotation
                self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

                # Sine "flash"
                sine_start_time = self.sine_start_time.data[0]
                sine_duration = self.sine_duration.data[0]
                sine_freq = self.sine_freq.data[0]
                sine_amp = self.sine_amp.data[0]
                baseline_lum = self.baseline_lum.data[0]

                time_in_sine = time - sine_start_time
                if 0.0 < time_in_sine <= sine_duration:
                    # current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                    current_lum = (baseline_lum - (sine_amp / 2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                else:
                    current_lum = baseline_lum

                # Set luminance
                self.luminance.data = current_lum

                self.apply_transform(self.simu_sacc)
                self.simu_sacc.draw('triangles', indices=self.index_buffer)


class SimuSaccadeWithSineFlash4000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000), step_size=100)  # ms
    sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000), step_size=100)  # ms
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200), step_size=10)  # ms
    saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15., limits=(-90.0, 90.0), step_size=0.01)  # deg

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

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
        self.azim_angle = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        sacc_start_time = self.saccade_start_time.data[0]
        saccade_duration = self.saccade_duration.data[0]
        saccade_target_angle = self.saccade_target_angle.data[0]

        time_in_saccade = time - sacc_start_time
        if 0.0 < time_in_saccade <= saccade_duration:
            # Calculate rotation
            angle = saccade_target_angle * 1000 * dt / saccade_duration
            self.azim_angle += angle

        # Set rotation
        self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

        # Sine "flash"
        sine_start_time = self.sine_start_time.data[0]
        sine_duration = self.sine_duration.data[0]
        sine_freq = self.sine_freq.data[0]
        sine_amp = self.sine_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_sine = time - sine_start_time
        if 0.0 < time_in_sine <= sine_duration:
            #current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = (baseline_lum - (sine_amp/2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = (baseline_lum - (sine_amp/2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)

        class SimuSaccadeWithSineFlash4000(vxvisual.SphericalVisual):

            VERT_LOC = './gs_texture.vert'
            FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

            time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

            # Varying parameters
            luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
            # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
            rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

            # Static (set in program)
            texture_default = vxvisual.Attribute('texture_default', static=True)

            # Static (not set in rendering program)
            baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0),
                                                   step_size=0.01)
            # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
            contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                               step_size=0.01)  # Absolute contrast
            sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000),
                                                    step_size=100)  # ms
            sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000),
                                                  step_size=100)  # ms
            sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0),
                                               step_size=0.01)  # total lum range
            sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0),
                                                step_size=0.1)  # Hz
            saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500,
                                                       limits=(0.0, 5000), step_size=100)  # ms
            saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200),
                                                     step_size=10)  # ms
            saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15.,
                                                           limits=(-90.0, 90.0), step_size=0.01)  # deg

            texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

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
                self.azim_angle = 0.

            def initialize(self, **kwargs):
                self.time.data = 0.

            def render(self, dt):
                self.time.data += dt

                # Get times
                time = self.time.data[0] * 1000  # s -> ms

                # Saccade
                sacc_start_time = self.saccade_start_time.data[0]
                saccade_duration = self.saccade_duration.data[0]
                saccade_target_angle = self.saccade_target_angle.data[0]

                time_in_saccade = time - sacc_start_time
                if 0.0 < time_in_saccade <= saccade_duration:
                    # Calculate rotation
                    angle = saccade_target_angle * 1000 * dt / saccade_duration
                    self.azim_angle += angle

                # Set rotation
                self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

                # Sine "flash"
                sine_start_time = self.sine_start_time.data[0]
                sine_duration = self.sine_duration.data[0]
                sine_freq = self.sine_freq.data[0]
                sine_amp = self.sine_amp.data[0]
                baseline_lum = self.baseline_lum.data[0]

                time_in_sine = time - sine_start_time
                if 0.0 < time_in_sine <= sine_duration:
                    # current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                    current_lum = (baseline_lum - (sine_amp / 2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                else:
                    current_lum = baseline_lum

                # Set luminance
                self.luminance.data = current_lum

                self.apply_transform(self.simu_sacc)
                self.simu_sacc.draw('triangles', indices=self.index_buffer)



class SimuSaccadeWithStepFlash2000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000), step_size=100)  # ms
    sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000), step_size=100)  # ms
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=1.0, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200), step_size=10)  # ms
    saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15., limits=(-90.0, 90.0), step_size=0.01)  # deg

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
        self.azim_angle = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        sacc_start_time = self.saccade_start_time.data[0]
        saccade_duration = self.saccade_duration.data[0]
        saccade_target_angle = self.saccade_target_angle.data[0]

        time_in_saccade = time - sacc_start_time
        if 0.0 < time_in_saccade <= saccade_duration:
            # Calculate rotation
            angle = saccade_target_angle * 1000 * dt / saccade_duration
            self.azim_angle += angle

        # Set rotation
        self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

        # Sine "flash"
        sine_start_time = self.sine_start_time.data[0]
        sine_duration = self.sine_duration.data[0]
        sine_freq = self.sine_freq.data[0]
        sine_amp = self.sine_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_sine = time - sine_start_time
        if 0.0 < time_in_sine <= sine_duration:
            #current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = 0.25
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)

        class SimuSaccadeWithSineFlash2000(vxvisual.SphericalVisual):

            VERT_LOC = './gs_texture.vert'
            FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

            time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

            # Varying parameters
            luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
            # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
            rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

            # Static (set in program)
            texture_default = vxvisual.Attribute('texture_default', static=True)

            # Static (not set in rendering program)
            baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0),
                                                   step_size=0.01)
            # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
            contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                               step_size=0.01)  # Absolute contrast
            sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000),
                                                    step_size=100)  # ms
            sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000),
                                                  step_size=100)  # ms
            sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0),
                                               step_size=0.01)  # total lum range
            sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0),
                                                step_size=0.1)  # Hz
            saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500,
                                                       limits=(0.0, 5000), step_size=100)  # ms
            saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200),
                                                     step_size=10)  # ms
            saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15.,
                                                           limits=(-90.0, 90.0), step_size=0.01)  # deg

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
                self.azim_angle = 0.

            def initialize(self, **kwargs):
                self.time.data = 0.

            def render(self, dt):
                self.time.data += dt

                # Get times
                time = self.time.data[0] * 1000  # s -> ms

                # Saccade
                sacc_start_time = self.saccade_start_time.data[0]
                saccade_duration = self.saccade_duration.data[0]
                saccade_target_angle = self.saccade_target_angle.data[0]

                time_in_saccade = time - sacc_start_time
                if 0.0 < time_in_saccade <= saccade_duration:
                    # Calculate rotation
                    angle = saccade_target_angle * 1000 * dt / saccade_duration
                    self.azim_angle += angle

                # Set rotation
                self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

                # Sine "flash"
                sine_start_time = self.sine_start_time.data[0]
                sine_duration = self.sine_duration.data[0]
                sine_freq = self.sine_freq.data[0]
                sine_amp = self.sine_amp.data[0]
                baseline_lum = self.baseline_lum.data[0]

                time_in_sine = time - sine_start_time
                if 0.0 < time_in_sine <= sine_duration:
                    # current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                    current_lum = (baseline_lum - (sine_amp / 2)) + np.cos(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
                else:
                    current_lum = baseline_lum

                # Set luminance
                self.luminance.data = current_lum

                self.apply_transform(self.simu_sacc)
                self.simu_sacc.draw('triangles', indices=self.index_buffer)


class SimuSaccadeWithStepFlash4000(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_simu_saccade_sine_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = Luminance('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # rotation = SaccadicRotation('rotation', default=0.0, limits=(0.0, 360.0), internal=True)
    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0), internal=True)

    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # Absolute contrast
    sine_start_time = vxvisual.IntParameter('sine_start_time', static=True, default=2000, limits=(0.0, 5000), step_size=100)  # ms
    sine_duration = vxvisual.IntParameter('sine_duration', static=True, default=1000, limits=(0.0, 5000), step_size=100)  # ms
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    sine_freq = vxvisual.FloatParameter('sine_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz
    saccade_start_time = vxvisual.IntParameter('saccade_start_time', static=True, default=1500, limits=(0.0, 5000), step_size=100)  # ms
    saccade_duration = vxvisual.IntParameter('saccade_duration', static=True, default=50, limits=(20, 200), step_size=10)  # ms
    saccade_target_angle = vxvisual.FloatParameter('saccade_target_angle', static=True, default=15., limits=(-90.0, 90.0), step_size=0.01)  # deg

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

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
        self.azim_angle = 0.

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] * 1000  # s -> ms

        # Saccade
        sacc_start_time = self.saccade_start_time.data[0]
        saccade_duration = self.saccade_duration.data[0]
        saccade_target_angle = self.saccade_target_angle.data[0]

        time_in_saccade = time - sacc_start_time
        if 0.0 < time_in_saccade <= saccade_duration:
            # Calculate rotation
            angle = saccade_target_angle * 1000 * dt / saccade_duration
            self.azim_angle += angle

        # Set rotation
        self.rotation.data = transforms.rotate(self.azim_angle, (0, 0, 1))

        # Sine "flash"
        sine_start_time = self.sine_start_time.data[0]
        sine_duration = self.sine_duration.data[0]
        sine_freq = self.sine_freq.data[0]
        sine_amp = self.sine_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_sine = time - sine_start_time
        if 0.0 < time_in_sine <= sine_duration:
            #current_lum = baseline_lum + np.sin(sine_freq * time_in_sine / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = 0.25
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.simu_sacc)
        self.simu_sacc.draw('triangles', indices=self.index_buffer)
