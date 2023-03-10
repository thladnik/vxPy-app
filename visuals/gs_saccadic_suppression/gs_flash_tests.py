from __future__ import annotations
import h5py
import numpy as np
import scipy.io
from vispy import gloo
from vispy.util import transforms

import vxpy.core.visual as vxvisual


def _import_texture_from_hdf(texture_file):
    with h5py.File(texture_file, 'r') as f:
        vertices = f['vertices'][:]
        indices = f['indices'][:]
        intensity = f['intensity'][:]

    return vertices, indices, intensity

# def _convert_mat_texture_to_hdf(path: str):
#     # Load model data
#     d = scipy.io.loadmat(path)
#     x, y, z = d['grid']['x'][0][0], \
#               d['grid']['y'][0][0], \
#               d['grid']['z'][0][0]
#
#     vertices = np.array([x.flatten(), y.flatten(), z.flatten()]).T
#
#     idcs = list()
#     azim_lvls = x.shape[1]
#     elev_lvls = x.shape[0]
#     for i in np.arange(elev_lvls):
#         for j in np.arange(azim_lvls):
#             idcs.append([i * azim_lvls + j, i * azim_lvls + j + 1, (i + 1) * azim_lvls + j + 1])
#             idcs.append([i * azim_lvls + j, (i + 1) * azim_lvls + j, (i + 1) * azim_lvls + j + 1])
#
#     indices = np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)
#     indices = indices[indices < azim_lvls * elev_lvls]
#
#     return


def _convert_mat_texture_to_hdf(mat_path: str, hdf_path: str):
    d = scipy.io.loadmat(mat_path)
    x, y, z = d['grid']['x'][0][0], \
              d['grid']['y'][0][0], \
              d['grid']['z'][0][0]

    # az, el = d['grid']['azim'][0][0].flatten(), \
    #          d['grid']['elev'][0][0].flatten()

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



class TextureModulateContrLum(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_meanlum_mod.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # Absolute contrast
    contrast = vxvisual.FloatParameter('contrast', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0))
    texture_default = vxvisual.Attribute('texture_default', static=True)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.time.remove_downstream_links()
        self.rotation.value_map = self._rotate

        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Rotation
        self.rotation.data = 0

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        self.time.connect(self.tex_mod)
        self.rotation.connect(self.tex_mod)
        self.luminance.connect(self.tex_mod)
        self.contrast.connect(self.tex_mod)
        self.texture_default.connect(self.tex_mod)

    @staticmethod
    def _rotate(angle):
        return transforms.rotate(angle, (0, 0, 1))

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        self.apply_transform(self.tex_mod)
        self.tex_mod.draw('triangles', indices=self.index_buffer)


################
# SINUS LUMINANCE MODULATION

class SinusAmplitude(vxvisual.FloatParameter):
    def __init__(self, *args, **kwargs):
        vxvisual.FloatParameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):

        time = TextureSinusLumModulation.time.data[0]
        amp = TextureSinusLumModulation.sine_luminance_amplitude.data[0]
        freq = TextureSinusLumModulation.sine_luminance_frequency.data[0]
        mean = TextureSinusLumModulation.sine_mean_luminance.data[0]

        lum = mean + np.sin(freq * time * 2.0 * np.pi) * amp / 2.0
        TextureSinusLumModulation.luminance.data = lum


class TextureSinusLumModulation(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_meanlum_mod.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # Absolute contrast
    contrast = vxvisual.FloatParameter('contrast', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0))
    texture_default = vxvisual.Attribute('texture_default', static=True)

    sine_luminance_amplitude = SinusAmplitude('sine_luminance_amplitude', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    sine_luminance_frequency = vxvisual.FloatParameter('sine_luminance_frequency', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)
    sine_mean_luminance = vxvisual.FloatParameter('sine_mean_luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.time.add_downstream_link(self.sine_luminance_amplitude)
        self.rotation.value_map = self._rotate

        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Rotation
        self.rotation.data = 0

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        self.time.connect(self.tex_mod)
        self.rotation.connect(self.tex_mod)
        self.luminance.connect(self.tex_mod)
        self.contrast.connect(self.tex_mod)
        self.texture_default.connect(self.tex_mod)

    @staticmethod
    def _rotate(angle):
        return transforms.rotate(angle, (0, 0, 1))

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        self.apply_transform(self.tex_mod)
        self.tex_mod.draw('triangles', indices=self.index_buffer)


################
# FLASH

class FlashLuminance(vxvisual.FloatParameter):
    def __init__(self, *args, **kwargs):
        vxvisual.FloatParameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        time = TextureFlash.time.data[0]
        start = TextureFlash.flash_start.data[0]
        dur = TextureFlash.flash_duration.data[0] / 1000

        # Get current luminance
        if start > time:
            lum = TextureFlash.start_luminance.data[0]
        elif time >= start > time - dur:
            lum = self.data[0]
        else:
            lum = TextureFlash.end_luminance.data[0]

        # Set luminance
        TextureFlash.luminance.data = lum


class TextureFlash(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_meanlum_mod.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # Absolute contrast
    contrast = vxvisual.FloatParameter('contrast', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0))
    texture_default = vxvisual.Attribute('texture_default', static=True)

    start_luminance = vxvisual.FloatParameter('start_luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)
    flash_start = vxvisual.FloatParameter('flash_start', static=True, default=1.0, limits=(0.0, 10.0), step_size=0.1)
    flash_duration = vxvisual.FloatParameter('flash_duration', static=True, default=500, limits=(0, 1000), step_size=10)
    flash_luminance = FlashLuminance('flash_luminance', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)
    end_luminance = vxvisual.FloatParameter('end_luminance', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.time.add_downstream_link(self.flash_luminance)
        self.rotation.value_map = self._rotate

        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Rotation
        self.rotation.data = 0

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        self.time.connect(self.tex_mod)
        self.rotation.connect(self.tex_mod)
        self.luminance.connect(self.tex_mod)
        self.contrast.connect(self.tex_mod)
        self.texture_default.connect(self.tex_mod)

    @staticmethod
    def _rotate(angle):
        return transforms.rotate(angle, (0, 0, 1))

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        self.apply_transform(self.tex_mod)
        self.tex_mod.draw('triangles', indices=self.index_buffer)


################
# TEXTURE ROTATING IN AZIMUTH

class AngularVelocity(vxvisual.FloatParameter):

    def __init__(self, *args, **kwargs):
        vxvisual.FloatParameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        time = TextureRotation.time.data[0]
        ang_vel = self.data[0]

        angle = time * ang_vel

        TextureRotation.rotation.data = angle


class TextureRotation(vxvisual.SphericalVisual):

    VERT_LOC = './gs_texture.vert'
    FRAG_LOC = './gs_texture_meanlum_mod.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    # Absolute contrast
    contrast = vxvisual.FloatParameter('contrast', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    rotation = vxvisual.Mat4Parameter('rotation', default=0.0, limits=(0.0, 360.0))
    texture_default = vxvisual.Attribute('texture_default', static=True)

    angular_velocity = AngularVelocity('angular_velocity', static=True, default=20.0, limits=(0.0, 360.0))

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        self.time.add_downstream_link(self.angular_velocity)
        self.rotation.value_map = self._rotate

        texture_file = 'visuals/sphere_simu_saccade/stimulus_data/blobstimtest.hdf5'

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(texture_file)

        # Rotation
        self.rotation.data = 0

        # Create index buffer
        self.index_buffer = gloo.IndexBuffer(indices)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.tex_mod = gloo.Program(VERT, FRAG, count=vertices.shape[0])
        self.tex_mod['a_position'] = vertices

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        self.time.connect(self.tex_mod)
        self.rotation.connect(self.tex_mod)
        self.luminance.connect(self.tex_mod)
        self.contrast.connect(self.tex_mod)
        self.texture_default.connect(self.tex_mod)

    @staticmethod
    def _rotate(angle):
        return transforms.rotate(angle, (0, 0, 1))

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        self.apply_transform(self.tex_mod)
        self.tex_mod.draw('triangles', indices=self.index_buffer)
