from __future__ import annotations
import h5py
import numpy as np
import scipy.io
from vispy import gloo
from vispy.util import transforms
from vxpy.core import visual
from vxpy.utils import sphere

import vxpy.core.visual as vxvisual

class LogChirp(vxvisual.SphericalVisual):

    VERT_LOC = './chirp.vert'
    FRAG_LOC = './chirp.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)
    freq_t = vxvisual.FloatParameter('freq_t',default=30, limits=(0.5,60),step_size=0.5)    # Hz

    # Static parameters
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)
    sine_amp = vxvisual.FloatParameter('sine_amp', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    starting_freq = vxvisual.FloatParameter('starting_freq',static = True, default = 30, limits = (0.5,60), step_size=0.5)  # Hz
    final_freq = vxvisual.FloatParameter('final_freq',static = True, default = 1, limits = (0.5,60), step_size=0.5)  # Hz
    chirp_duration = vxvisual.FloatParameter('chirp_duration', static=True, default=20, limits = (0.5,60), step_size=0.5)   # sec

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.a_azimuth)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.a_elevation)

        # Set up program
        VERT = self.load_vertex_shader(self.VERT_LOC)
        FRAG = self.load_shader(self.FRAG_LOC)
        self.chirp = gloo.Program(VERT, FRAG)
        self.chirp['a_position'] = self.position_buffer

        # connect parameters
        self.time.connect(self.chirp)
        self.luminance.connect(self.chirp)
        self.freq_t.connect(self.chirp)
        self.baseline_lum.connect(self.chirp)
        self.sine_amp.connect(self.chirp)
        self.starting_freq.connect(self.chirp)
        self.final_freq.connect(self.chirp)
        self.chirp_duration.connect(self.chirp)

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] # in sec

        # Get static parameters
        baseline_lum = self.baseline_lum.data[0]
        sine_amp = self.sine_amp.data[0]
        starting_freq = self.starting_freq.data[0]
        final_freq = self.final_freq.data[0]
        chirp_duration = self.chirp_duration.data[0]

        current_freq = starting_freq * (final_freq/starting_freq)**(time/chirp_duration)
        #current_lum = baseline_lum + sine_amp * (np.sin(current_freq * time * 2.0 * np.pi)) / 2.0
        current_lum = baseline_lum + sine_amp * (scipy.signal.chirp(time,starting_freq,chirp_duration,1,'logarithmic'))

        self.luminance.data = current_lum
        self.freq_t.data = current_freq

        self.apply_transform(self.chirp)
        self.chirp.draw('triangles', indices=self.index_buffer)

