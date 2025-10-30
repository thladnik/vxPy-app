# NEW STIMULUS - START PHASES WHEN
import numpy as np    # generall inputs

# vispy stuff
from vispy import gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer             # program function and bufferes
from vispy.geometry import create_sphere

# vxpy stuff
from vxpy.core import visual as vxvisual
import h5py
import numpy as np
import scipy.io
import vxpy.core.ipc as vxipc

# visual stimulus for experiments presenting a dark flash after saccade, INF-Phase version of vxPy

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

# class of stimulus ()
class SaccadeFlash_SaccadeTriggerStimulation4000(vxvisual.SphericalVisual):
    """ Dark flash (with some delay) presented after saccade was detected """
    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Define parameters of variable phase
    time = vxvisual.FloatParameter('time', internal=True)
    base_luminance = vxvisual.FloatParameter('base_luminance', default=0.75, static=True)
    u_color = vxvisual.FloatParameter('u_color', internal=True)  # control checking presented lum for every frame
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast
    luminance = vxvisual.FloatParameter('luminance', internal=True)  # control checking presented lum for every frame
    flash_delay = vxvisual.FloatParameter('flash_delay', default=0.5, static=True)
    flash_duration = vxvisual.FloatParameter('flash_duration', default=0.5, static=True)
    flash_cos_amplitude = vxvisual.FloatParameter('flash_cos_amplitude', default=0.25, static=True)
    flash_cos_freq = vxvisual.FloatParameter('flash_cos_freq', default=2.0, static=True)  # in Hz

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_4000_blobs.hdf5'

    # path to vertex and fragement shader
    VERT_PATH = './gs_texture_no_rot.vert'
    FRAG_PATH = './gs_simu_saccade_sine_flash.frag'


    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        self.index_buffer = IndexBuffer(indices)
        self.position_buffer = VertexBuffer(vertices)

        # set up program
        VERT = self.load_vertex_shader(self.VERT_PATH)
        FRAG = self.load_shader(self.FRAG_PATH)
        self.progam = gloo.Program(VERT, FRAG, count=vertices.shape[0])

        # set position in program
        self.progam['a_position'] = self.position_buffer

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.progam)
        self.u_color.connect(self.progam)
        #self.phase_duration.connect(self.progam)
        self.luminance.connect(self.progam)
        self.contrast.connect(self.progam)
        #self.phase_end_buffer.connect(self.progam)
        self.texture_default.connect(self.progam)

    def flash_colorCurve(self, currentFrame_time):
        """Flash (dark/light) brightness should be modulated by a cos function.
           Shape: get darker and then again brighter (back to baseline Luminance).
            Cos function shifted to beginn at 0????? """
        flash_current_lum = (self.base_luminance.data[0] - self.flash_cos_amplitude.data) + \
                            np.cos(self.flash_cos_freq.data * currentFrame_time * 2.0 * np.pi) * \
                            self.flash_cos_amplitude.data
        flash_current_color = flash_current_lum
        return flash_current_color

    def initialize(self, **params):
        """Initialize running of skritp. Reset variables that need to have a specific value at each start of phase start"""
        self.time.data = 0.0
        self.contrast.data = 0.5
        self.luminance.data = 1
        # self.time.data = vxipc.get_time()  # get global time

        self.flash_start = self.time.data + self.flash_delay.data

    def render(self, dt):
        """ Render stimulus with previous defined stuff. Like on_draw method in vispy. """
        # Add elapsed time to u_time
        self.time.data += dt

        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=True)

        # 1. check if idle state or stimulus happening (by comparing time)
        if self.flash_start <= self.time.data < self.flash_start + self.flash_duration.data:
            #self.u_color.data = self.flash_colorCurve(self.time.data[0] - self.flash_start)  # record current lum
            self.luminance.data = self.flash_colorCurve(self.time.data[0] - self.flash_start)  # record current lum

        elif self.time.data[0] >= self.flash_start + self.flash_duration.data:
            # reset flash_start value to inf
            self.flash_start = np.inf

        else:
            # set color to fixed value (This is IDLE state)
            #self.u_color.data = self.base_luminance.data  # record current lum
            self.luminance.data = self.base_luminance.data  # record current lum

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.apply_transform(self.progam)
        self.progam.draw('triangles', self.index_buffer)

class SaccadeFlash_SaccadeTriggerStimulation2000(vxvisual.SphericalVisual):
    """ Dark flash (with some delay) presented after saccade was detected """
    # Static (set in program)
    texture_default = vxvisual.Attribute('texture_default', static=True)

    # Define parameters of variable phase
    time = vxvisual.FloatParameter('time', internal=True)
    base_luminance = vxvisual.FloatParameter('base_luminance', default=0.75, static=True)
    u_color = vxvisual.FloatParameter('u_color', internal=True)  # control checking presented lum for every frame
    contrast = vxvisual.FloatParameter('contrast', static=True, default=0.5, limits=(0.0, 1.0),
                                       step_size=0.01)  # Absolute contrast
    luminance = vxvisual.FloatParameter('luminance', internal=True)  # control checking presented lum for every frame
    flash_delay = vxvisual.FloatParameter('flash_delay', default=0.5, static=True)
    flash_duration = vxvisual.FloatParameter('flash_duration', default=0.5, static=True)
    flash_cos_amplitude = vxvisual.FloatParameter('flash_cos_amplitude', default=0.25, static=True)
    flash_cos_freq = vxvisual.FloatParameter('flash_cos_freq', default=2.0, static=True)  # in Hz

    texture_file = 'visuals/gs_saccadic_suppression/stimulus_data/texture_brightness_0_1_2000_blobs.hdf5'

    # path to vertex and fragement shader
    VERT_PATH = './gs_texture_no_rot.vert'
    FRAG_PATH = './gs_simu_saccade_sine_flash.frag'


    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # Load model and texture
        vertices, indices, intensity = _import_texture_from_hdf(self.texture_file)

        self.index_buffer = IndexBuffer(indices)
        self.position_buffer = VertexBuffer(vertices)

        # set up program
        VERT = self.load_vertex_shader(self.VERT_PATH)
        FRAG = self.load_shader(self.FRAG_PATH)
        self.progam = gloo.Program(VERT, FRAG, count=vertices.shape[0])

        # set position in program
        self.progam['a_position'] = self.position_buffer

        # Set normalized texture
        tex = np.ascontiguousarray((intensity - intensity.min()) / (intensity.max() - intensity.min()))
        self.texture_default.data = tex

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.progam)
        self.u_color.connect(self.progam)
        #self.phase_duration.connect(self.progam)
        self.luminance.connect(self.progam)
        self.contrast.connect(self.progam)
        #self.phase_end_buffer.connect(self.progam)
        self.texture_default.connect(self.progam)

    def flash_colorCurve(self, currentFrame_time):
        """Flash (dark/light) brightness should be modulated by a cos function.
           Shape: get darker and then again brighter (back to baseline Luminance).
            Cos function shifted to beginn at 0????? """
        flash_current_lum = (self.base_luminance.data[0] - self.flash_cos_amplitude.data) + \
                            np.cos(self.flash_cos_freq.data * currentFrame_time * 2.0 * np.pi) * \
                            self.flash_cos_amplitude.data
        flash_current_color = flash_current_lum
        return flash_current_color

    def initialize(self, **params):
        """Initialize running of skritp. Reset variables that need to have a specific value at each start of phase start"""
        self.time.data = 0.0
        self.contrast.data = 0.5
        self.luminance.data = 1
        # self.time.data = vxipc.get_time()  # get global time

        self.flash_start = self.time.data + self.flash_delay.data

    def render(self, dt):
        """ Render stimulus with previous defined stuff. Like on_draw method in vispy. """
        # Add elapsed time to u_time
        self.time.data += dt

        # gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=True)

        # 1. check if idle state or stimulus happening (by comparing time)
        if self.flash_start <= self.time.data < self.flash_start + self.flash_duration.data:
            #self.u_color.data = self.flash_colorCurve(self.time.data[0] - self.flash_start)  # record current lum
            self.luminance.data = self.flash_colorCurve(self.time.data[0] - self.flash_start)  # record current lum

        elif self.time.data[0] >= self.flash_start + self.flash_duration.data:
            # reset flash_start value to inf
            self.flash_start = np.inf

        else:
            # set color to fixed value (This is IDLE state)
            #self.u_color.data = self.base_luminance.data  # record current lum
            self.luminance.data = self.base_luminance.data  # record current lum

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.apply_transform(self.progam)
        self.progam.draw('triangles', self.index_buffer)
