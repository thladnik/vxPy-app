"""
vxPy_app ./visuals/spherical_grating/spherical_grating.py
Copyright (C) 2022 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
from vispy import gloo
from vispy.util import transforms
import numpy as np

from vxpy.core import visual
from vxpy.utils import sphere


class RFCenterLocation(visual.Vec3Parameter):
    def __init__(self, *args, **kwargs):
        visual.Vec3Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        azim = PartialSphericalGratingWithRoundRFs.rf_center_azimuth.data[0]
        azim = azim / 180 * np.pi
        elev = PartialSphericalGratingWithRoundRFs.rf_center_elevation.data[0]
        elev = elev / 180 * np.pi

        vec3 = np.array([np.sin(azim) * np.cos(elev),
                         np.cos(azim) * np.cos(elev),
                         np.sin(elev)])
        self.data = vec3

        print(azim, elev)


class PartialSphericalGratingWithRoundRFs(visual.SphericalVisual):
    # (optional) Add a short description
    description = ''

    # Define parameters
    time = visual.FloatParameter('time', internal=True)
    angular_velocity = visual.FloatParameter('angular_velocity', default=30, limits=(-150, 150), step_size=5, static=True)
    angular_period = visual.FloatParameter('angular_period', default=45, limits=(1, 360), step_size=5, static=True)
    rf_center_azimuth = visual.FloatParameter('rf_center_azimuth', default=0, limits=(-180, 180), step_size=1, static=True)
    rf_center_elevation = visual.FloatParameter('rf_center_elevation', default=0, limits=(-90, 90), step_size=1, static=True)
    rf_diameter = visual.FloatParameter('rf_diameter', default=30, limits=(1, 180), step_size=1, static=True)
    rf_center_location = RFCenterLocation('rf_center_location', internal=True, static=True)

    # Paths to shaders
    VERT_PATH = './partial_spherical_grating_round_rfs.vert'
    FRAG_PATH = './partial_spherical_grating_round_rfs.frag'

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree_zero_front_pos_cw)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.rf_diameter.connect(self.grating)
        self.rf_center_location.connect(self.grating)

        # Connect to location coordinates
        self.rf_center_azimuth.add_downstream_link(self.rf_center_location)
        self.rf_center_elevation.add_downstream_link(self.rf_center_location)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)


class LocalTranslationGrating(visual.SphericalVisual):
    # (optional) Add a short description
    description = ''

    # Define parameters
    time = visual.FloatParameter('time', internal=True)
    angular_velocity = visual.FloatParameter('angular_velocity', default=30, limits=(-150, 150), step_size=5, static=True)
    angular_period = visual.FloatParameter('angular_period', default=45, limits=(1, 360), step_size=5, static=True)
    rf_center_azimuth = visual.FloatParameter('rf_center_azimuth', default=0, limits=(-180, 180), step_size=1, static=True)
    rf_center_elevation = visual.FloatParameter('rf_center_elevation', default=0, limits=(-90, 90), step_size=1, static=True)
    rf_diameter = visual.FloatParameter('rf_diameter', default=0, limits=(1, 180), step_size=1, static=True)
    rf_center_location = RFCenterLocation('rf_center_location', internal=True, static=True)

    # Paths to shaders
    VERT_PATH = './partial_spherical_grating_round_rfs.vert'
    FRAG_PATH = './partial_spherical_grating_round_rfs.frag'

    def __init__(self, *args, **kwargs):
        visual.SphericalVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi / 2)
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)
        self.azimuth_buffer = gloo.VertexBuffer(self.sphere.azimuth_degree)
        self.elevation_buffer = gloo.VertexBuffer(self.sphere.elevation_degree)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.angular_velocity.connect(self.grating)
        self.angular_period.connect(self.grating)
        self.rf_diameter.connect(self.grating)
        self.rf_center_location.connect(self.grating)

        # Connect to location coordinates
        self.rf_center_azimuth.add_downstream_link(self.rf_center_location)
        self.rf_center_elevation.add_downstream_link(self.rf_center_location)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_azimuth'] = self.azimuth_buffer
        self.grating['a_elevation'] = self.elevation_buffer

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
