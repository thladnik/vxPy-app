import numpy as np    # generall inputs

# vispy stuff
from vispy import gloo, app
from vispy.gloo import Program, VertexBuffer, IndexBuffer             # program function and bufferes
from vispy.util.transforms import translate, perspective, rotate      # 3d objekte und rotation und so...
from vispy.geometry import create_sphere

# vxpy stuff
from vxpy.core import visual as vxvisual
from vxpy.core.attribute import read_attribute


def create_parameterset_freq2(flash_delay, phase_duration, saccade_waiting_time, phase_end_buffer):
    """ Function for generating Parameters for saccade_flash protocoll. """
    return {SaccadeFlash_PhaseStimulation.base_luminance: 0.75,
            SaccadeFlash_PhaseStimulation.phase_duration: phase_duration,
            SaccadeFlash_PhaseStimulation.saccade_t_between_saccades: saccade_waiting_time,
            SaccadeFlash_PhaseStimulation.phaseEnd_buffer: phase_end_buffer,

            SaccadeFlash_PhaseStimulation.flash_delay: flash_delay,
            SaccadeFlash_PhaseStimulation.flash_duration: 0.5,
            SaccadeFlash_PhaseStimulation.flash_cos_amplitude: 0.25,
            SaccadeFlash_PhaseStimulation.flash_cos_freq: 2.0}


# class of stimulus (not saccade triggered)
class SaccadeFlash_PhaseStimulation(vxvisual.SphericalVisual):
    """Dark flash (with some delay) after saccade was detected"""

    # Define parameters (all parameters changeable in GUI or should be saved for analysis)
    time = vxvisual.FloatParameter('time', internal=True)
    base_luminance = vxvisual.FloatParameter('base_luminance', default=0.75, static=True)
    phase_duration = vxvisual.FloatParameter('phase_duration', static=True)
    luminance_overTime = vxvisual.FloatParameter('luminance_overTime', internal=True)  # control checking presented lum for every frame
    phaseEnd_buffer = vxvisual.FloatParameter('phase_end_buffer', internal=True, static=True)
    phase_duration_pure = vxvisual.FloatParameter('phase_duration_pure', internal=True, static=True)

    saccade_start = vxvisual.FloatParameter('saccade_start', internal=True)
    saccade_start_save = vxvisual.FloatParameter('saccade_start_save', internal=True, static=True)
    saccade_t_between_saccades = vxvisual.FloatParameter('saccade_t_between_saccades', default= 8.0, static=True) # time in which no saccade should trigger
    saccade_ValidEventList = vxvisual.FloatParameter('saccade_ValidEventList', internal=True) # counter for all saccade events
    saccade_timeout_overhang = vxvisual.FloatParameter('saccade_timeout_overhang', static=True, internal=True) # overhang time from last stimulation pahse
    #saccade_detected_time = vxvisual.FloatParameter('saccade_detected_time', static=True, internal=True) # timepoint when saccade was detected

    flash_delay = vxvisual.FloatParameter('flash_delay', default=0.5, static=True)
    flash_start = vxvisual.FloatParameter('flash_start', internal=True)#, static=True)
    flash_duration = vxvisual.FloatParameter('flash_duration', default=0.5, static=True)
    flash_cos_amplitude = vxvisual.FloatParameter('flash_cos_amplitude', default=0.25, static=True)
    flash_cos_freq = vxvisual.FloatParameter('flash_cos_freq', default=2.0, static=True) # in Hz


    # path to vertex and fragement shader
    VERT_PATH = 'saccade_flash.vert'
    FRAG_PATH = 'saccade_flash.frag'

    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)

        # gerneate uv-sphere and put data in buffer
        m_sphere = create_sphere(method='ico')  # mesh of a uv-sphere
        V_sphere = m_sphere.get_vertices()      # get vertices
        I_sphere = m_sphere.get_faces()         # get indices
        self.index_buffer = IndexBuffer(I_sphere)
        self.position_buffer = VertexBuffer(V_sphere)

        # set up program
        self.progam = Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # set poition in program
        self.progam['a_position'] = self.position_buffer

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.progam)
        self.base_luminance.connect(self.progam)
        self.phase_duration.connect(self.progam)
        self.luminance_overTime.connect(self.progam)
        self.phaseEnd_buffer.connect(self.progam)
        self.phase_duration_pure.connect(self.progam)

        self.saccade_start.connect(self.progam)
        self.saccade_start_save.connect(self.progam)
        self.saccade_t_between_saccades.connect(self.progam)
        self.saccade_ValidEventList.connect(self.progam)
        self.saccade_timeout_overhang.connect(self.progam)
        #self.saccade_detected_time.connect(self.progam)

        self.flash_delay.connect(self.progam)
        self.flash_start.connect(self.progam)
        self.flash_duration.connect(self.progam)
        self.flash_cos_amplitude.connect(self.progam)
        self.flash_cos_freq.connect(self.progam)


        # initialize first parameters, that change between phases or change based on settings (but are static)
        self.saccade_idx = None                                                                # something from saccade detection.... ask Tim
        self.saccade_timeout_overhang.data = 0.01                                               # create self overlap initialisieren
        self.phase_duration_pure.data = self.phase_duration.data - self.phaseEnd_buffer.data
        self.saccade_detection_Flag = False                                                    # no saccades detected at beginning
        self.no_saccade_time = self.saccade_timeout_overhang.data                              # beginning every saccade should be detected
        self.saccade_start_save.data = np.inf
        self.saccade_start.data = np.inf
        self.saccade_ValidEventList.data = 0
        self.flash_start.data = 0

        print('saccade_timeout_overhang ', self.phase_duration_pure.data)
        print('phase_duration ', self.phase_duration.data)
        print('phase_end_buffer ', self.phaseEnd_buffer.data)
        print('saccade_t_between_saccades ', self.saccade_t_between_saccades.data)
        print('base_luminance', self.base_luminance.data)
        print('saccade_timeout_overhang', self.saccade_timeout_overhang.data)
        print('saccade_start_save', self.saccade_start_save.data)
        print('saccade_start', self.saccade_start.data)

        # self.react_to_rigger('saccade_trigger', self.saccade_happened)      # insert real saccade trigger function here!
        self.trigger_functions.append(self.saccade_happend) # test only stimulus:

    def check_saccade_happened(self):
        if self.saccade_idx is None:
            idx, times, values = read_attribute('eyeposdetect_saccade_trigger')
            self.saccade_idx = idx[-1] if len(idx) > 0 else None
            self.saccade_ValidEventList.data = 0  # set marker vor valid saccade

        else:
            idx, times, values = read_attribute('eyeposdetect_saccade_trigger', from_idx=self.saccade_idx)
            if sum(values) > 0:
                self.saccade_happend()
                self.saccade_ValidEventList.data = 1  # set marker vor valid saccade
                # enter reset here?
                #self.saccade_idx =None

    def saccade_happend(self):
        """ When saccade detected set start time of saccade detected """
        self.saccade_start.data = self.time.data[0]
        print('SACCADE HAPPEND')
        #print(self.phase_duration_pure.data)
        #print(self.phase_duration.data)
        #print(self.phase_end_buffer.data)
        #print(self.base_luminance.data)

        # check if detected saccade is not too close to end of phase
        if self.saccade_start.data <= self.phase_duration_pure.data:
            print('A - IN pure_phase')

            # check if detected saccade is outside of noFlash time and set saccade start and flash start
            if self.saccade_start.data > self.no_saccade_time and self.saccade_detection_Flag is False:
                self.flash_start.data = self.saccade_start.data + self.flash_delay.data
                self.saccade_start_save.data = self.saccade_start.data
                print('Saccade outside noFlash Time')

                #set flag for saccade detected (only one per phase)
                if self.phase_duration.data > 1.0: # only if not in stimulus testing mode
                    self.saccade_detection_Flag = True
                    #print('Not in Testing Mode')

                # define timeout time for next saccade and check for overhanging time in next phase
                self.no_saccade_time = self.saccade_start.data + self.saccade_t_between_saccades.data
                # if saccade_t_between_saccades larger than phase
                if self.no_saccade_time > self.phase_duration.data > 1: # >1 because stimulus testing case
                    self.saccade_timeout_overhang.data = self.no_saccade_time - self.phase_duration.data
                    self.saccade_t_between_saccades.data = self.phase_duration.data # set saccade timeout to length of phase
                    print('Overhang detected: ', self.saccade_timeout_overhang.data)

                else:
                    self.saccade_timeout_overhang.data = 0.0  # if no overhang, add nothing
                    print('no Overhang')

            else: # delete saccade start if within saccade timeout or already one valid saccade in phase
                self.saccade_start.data = np.inf
                self.saccade_start_save.data = np.inf
                if self.saccade_start.data < self.no_saccade_time and self.saccade_detection_Flag is False:
                    print('Saccade IN noFlash Time')
                if self.saccade_detection_Flag is True:
                    print('Saccade already HAPPEND')

        else: # delete saccade start if too close to end of phase (phase_pure_time)
            self.saccade_start.data = np.inf
            print('B - OUT pure phase')

    def flash_colorCurve(self, currentFrame_time):
        """Flash (dark/light) brightness should be modulated by a cos function.
           Shape: get darker and then again brighter (back to baseline Luminance).
            Cos function shifted to beginn at 0????? """
        flash_current_lum = (self.base_luminance.data[0] - self.flash_cos_amplitude.data) + \
                            np.cos(self.flash_cos_freq.data * currentFrame_time * 2.0 * np.pi) * \
                            self.flash_cos_amplitude.data
        flash_current_color = [flash_current_lum, flash_current_lum, flash_current_lum, 1]
        return flash_current_color

    def initialize(self, **params):
        """ Reset variables that need to have a specific value at each start of phase start"""
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0
        self.flash_start.data = np.inf
        self.saccade_start.data = np.inf
        self.saccade_detection_Flag = False                         # no saccades detected at beginning
        self.no_saccade_time = self.saccade_timeout_overhang.data   # beginning every saccade should be detected
        self.saccade_ValidEventList.data = 0
        self.phase_duration_pure.data = self.phase_duration.data - self.phaseEnd_buffer.data
        #print(self.phase_duration_pure.data)
        #print('no_saccade_time ',self.no_saccade_time)
        print('---------- Start of phase ----------')

    def render(self, dt):
        """ Render stimulus with previous defined stuff. Like on_draw method in vispy. """
        # Add elapsed time to u_time
        self.time.data += dt

        self.check_saccade_happened()

        gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=True)

        # 1. check if idle state or stimulus happening (by comparing time)
        if self.flash_start.data <= self.time.data < self.flash_start.data + self.flash_duration.data:
            currentColor = self.flash_colorCurve(self.time.data[0] - self.flash_start.data)
            self.progam['u_color'] = currentColor
            self.luminance_overTime.data = currentColor[0]  # record current lum

        elif self.time.data[0] >= self.flash_start.data + self.flash_duration.data:
            # reset flash_start value to inf
            self.flash_start.data = np.inf

        else:
            # set color to fixed value
            self.progam['u_color'] = [self.base_luminance.data, self.base_luminance.data, self.base_luminance.data, 1]
            self.luminance_overTime.data = self.base_luminance.data  # record current lum

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.apply_transform(self.progam)
        self.progam.draw('triangles', self.index_buffer)