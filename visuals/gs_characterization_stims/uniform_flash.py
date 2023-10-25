from __future__ import annotations
import h5py
import numpy as np
import scipy.io
from vispy import gloo
from vispy.util import transforms
from vxpy.core import visual
from vxpy.utils import sphere

import vxpy.core.visual as vxvisual

class UniformFlashStep(vxvisual.SphericalVisual):     # for coarse texture only!

    VERT_LOC = './uniform_flash.vert'
    FRAG_LOC = './uniform_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    flash_start_time = vxvisual.IntParameter('flash_start_time', static=True, default=4000, limits=(0.0, 5000), step_size=100)  # ms
    flash_duration = vxvisual.IntParameter('flash_duration', static=True, default=500, limits=(0.0, 5000), step_size=100)  # ms
    flash_amp = vxvisual.FloatParameter('flash_amp', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # total lum range

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
        self.uni_flash = gloo.Program(VERT, FRAG)
        self.uni_flash['a_position'] = self.position_buffer


        # Connect parameters to rendering program
        self.time.connect(self.uni_flash)
        self.luminance.connect(self.uni_flash)

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] # in sec

        # Step Flash
        flash_start_time = self.flash_start_time.data[0]
        flash_duration = self.flash_duration.data[0]
        flash_amp = self.flash_amp.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_flash = time - flash_start_time
        if 0.0 < time_in_flash <= flash_duration:
            # current_lum = baseline_lum + np.sin(flash_freq * time_in_flash / 1000 * 2.0 * np.pi) * sine_amp / 2.0
            current_lum = baseline_lum + flash_amp
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.uni_flash)
        self.uni_flash.draw('triangles', indices=self.index_buffer)


class UniformFlashCos(vxvisual.SphericalVisual):     # for coarse texture only!

    VERT_LOC = './uniform_flash.vert'
    FRAG_LOC = './uniform_flash.frag'

    time = vxvisual.FloatParameter('time', default=0.0, limits=(0.0, 20.0))

    # Varying parameters
    luminance = vxvisual.FloatParameter('luminance', default=0.5, limits=(0.0, 1.0), step_size=0.01)

    # Static (not set in rendering program)
    baseline_lum = vxvisual.FloatParameter('baseline_lum', static=True, default=0.75, limits=(0.0, 1.0), step_size=0.01)    # lum_contrast = vxvisual.FloatParameter('lum_contrast', static=True, default=0.25, limits=(0.0, 1.0), step_size=0.01)
    flash_start_time = vxvisual.IntParameter('flash_start_time', static=True, default=4000, limits=(0.0, 5000), step_size=100)  # ms
    flash_duration = vxvisual.IntParameter('flash_duration', static=True, default=500, limits=(0.0, 5000), step_size=100)  # ms
    flash_amp = vxvisual.FloatParameter('flash_amp', static=True, default=0.5, limits=(0.0, 1.0), step_size=0.01)  # total lum range
    flash_freq = vxvisual.FloatParameter('flash_freq', static=True, default=1.0, limits=(0.0, 20.0), step_size=0.1)  # Hz

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
        self.uni_flash = gloo.Program(VERT, FRAG)
        self.uni_flash['a_position'] = self.position_buffer


        # Connect parameters to rendering program
        self.time.connect(self.uni_flash)
        self.luminance.connect(self.uni_flash)

    def initialize(self, **kwargs):
        self.time.data = 0.

    def render(self, dt):
        self.time.data += dt

        # Get times
        time = self.time.data[0] # in sec

        # Step Flash
        flash_start_time = self.flash_start_time.data[0]
        flash_duration = self.flash_duration.data[0]
        flash_amp = self.flash_amp.data[0]
        flash_freq = self.flash_freq.data[0]
        baseline_lum = self.baseline_lum.data[0]

        time_in_flash = time - flash_start_time
        if 0.0 < time_in_flash <= flash_duration:
            current_lum = baseline_lum + np.sin(flash_freq * time_in_flash * 2.0 * np.pi) * flash_amp / 2.0
            # current_lum = baseline_lum + flash_amp
        else:
            current_lum = baseline_lum

        # Set luminance
        self.luminance.data = current_lum

        self.apply_transform(self.uni_flash)
        self.uni_flash.draw('triangles', indices=self.index_buffer)
