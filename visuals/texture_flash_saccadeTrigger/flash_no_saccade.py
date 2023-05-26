import numpy as np    # generall inputs

# vispy stuff
from vispy import gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer             # program function and bufferes
from vispy.geometry import create_sphere

# vxpy stuff
from vxpy.core import visual as vxvisual
import vxpy.core.event as vxevent
from vxpy.extras.zf_eyeposition_tracking import EyePositionDetectionRoutine
import vxpy.core.ipc as vxipc

class SaccadeFlash_Flash_without_saccade(vxvisual.SphericalVisual):
    """Present random saccade flashes. Independent of when saccade happend"""

    # Define parameters (all parameters changeable in GUI or should be saved for analysis)
    time = vxvisual.FloatParameter('time', internal=True)
    base_luminance = vxvisual.FloatParameter('base_luminance', default=0.75, static=True)
    phase_duration = vxvisual.FloatParameter('phase_duration', static=True)
    luminance_overTime = vxvisual.FloatParameter('u_color', internal=True)  # control checking presented lum for every frame

    flash_delay = vxvisual.FloatParameter('flash_delay', default=0.5, static=True)
    flash_duration = vxvisual.FloatParameter('flash_duration', default=0.5, static=True)
    flash_cos_amplitude = vxvisual.FloatParameter('flash_cos_amplitude', default=0.25, static=True)
    flash_cos_freq = vxvisual.FloatParameter('flash_cos_freq', default=2.0, static=True)  # in Hz

    # path to vertex and fragement shader
    VERT_PATH = 'saccade_flash.vert'
    FRAG_PATH = 'saccade_flash.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # gerneate uv-sphere and put data in buffer
        m_sphere = create_sphere(method='ico')  # mesh of a uv-sphere
        V_sphere = m_sphere.get_vertices()  # get vertices
        I_sphere = m_sphere.get_faces()  # get indices
        self.index_buffer = IndexBuffer(I_sphere)
        self.position_buffer = VertexBuffer(V_sphere)

        # set up program
        self.progam = Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # set position in program
        self.progam['a_position'] = self.position_buffer

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.progam)
        self.base_luminance.connect(self.progam)
        self.phase_duration.connect(self.progam)
        self.luminance_overTime.connect(self.progam)

        self.flash_delay.connect(self.progam)
        self.flash_duration.connect(self.progam)
        self.flash_cos_amplitude.connect(self.progam)
        self.flash_cos_freq.connect(self.progam)

        self.saccade_trigger = vxevent.OnTrigger(EyePositionDetectionRoutine.sacc_trigger_name)

        self.flash_start = None
        self.luminance_overTime.data = 1

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
        """ Reset variables that need to have a specific value at each start of phase start"""
        # Reset u_time to 0 on each visual initialization
        # self.time.data = 0.0
        self.time.data = vxipc.get_time()
        self.flash_start = vxipc.get_time() + self.flash_delay.data[0] # start of phase + delay
        #print('---------- Start of phase ----------')

    def render(self, dt):
        """ Render stimulus with previous defined stuff. Like on_draw method in vispy. """
        # Add elapsed time to u_time
        self.time.data += dt

        gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=True)

        # 1. check if idle state or stimulus happening (by comparing time)
        if self.flash_start <= self.time.data < self.flash_start + self.flash_duration.data:
            self.luminance_overTime.data = self.flash_colorCurve(self.time.data[0] - self.flash_start)  # record current lum

        elif self.time.data[0] >= self.flash_start + self.flash_duration.data:
            # reset flash_start value to inf
            self.flash_start = np.inf

        else:
            # set color to fixed value (This is IDLE state)
            self.luminance_overTime.data = self.base_luminance.data  # record current lum

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.apply_transform(self.progam)
        self.progam.draw('triangles', self.index_buffer)