"""
vxPy_app ./visuals/local_translation_grating/local_translation_grating.py
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
import numpy as np
from vispy import gloo
from vispy.util import transforms
from vispy.geometry.generation import create_sphere

from vxpy.core import visual
from vxpy.utils import sphere


class GratingDirectionRotation(visual.Mat4Parameter):
    def __init__(self, *args, **kwargs):
        visual.Mat4Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        azim = LocalTranslationGrating_RoundArea.stimulus_patch_center_azimuth.data[0]
        azim_rad = azim / 180 * np.pi
        elev = LocalTranslationGrating_RoundArea.stimulus_patch_center_elevation.data[0]
        elev_rad = elev / 180 * np.pi

        vec_rf = np.array([np.sin(azim_rad) * np.cos(elev_rad),
                           np.cos(azim_rad) * np.cos(elev_rad),
                           np.sin(elev_rad)])

        pos2d = np.array([vec_rf[0], vec_rf[2]])
        dir2d = pos2d / np.linalg.norm(pos2d)
        if dir2d[0] < 0.0:
            angle = np.arcsin(dir2d[1])
        else:
            angle = -np.arcsin(dir2d[1])

        rot_mat = transforms.rotate(angle / np.pi * 180, (0, -1, 0))

        self.data = rot_mat


class StimulationPatchRotation(visual.Mat4Parameter):

    def __init__(self, *args, **kwargs):
        visual.Mat4Parameter.__init__(self, *args, **kwargs)

    def upstream_updated(self):
        azim = LocalTranslationGrating_RoundArea.stimulus_patch_center_azimuth.data[0]
        elev = LocalTranslationGrating_RoundArea.stimulus_patch_center_elevation.data[0]

        rot_mat = transforms.rotate(elev, (1, 0, 0)) @ transforms.rotate(azim, (0, 0, -1))

        self.data = rot_mat


class LocalTranslationGrating_RoundArea(visual.SphericalVisual):
# class LocalTranslationGrating_RoundArea(visual.BaseVisual):
    # (optional) Add a short description
    description = ''

    # Define parameters
    time = visual.FloatParameter('time', internal=True)
    grating_angular_velocity = visual.FloatParameter('grating_angular_velocity', default=30, limits=(-150, 150), step_size=5,
                                                     static=True)
    grating_angular_period = visual.FloatParameter('grating_angular_period', default=30, limits=(1, 360), step_size=5, static=True)
    stimulus_patch_center_azimuth = visual.FloatParameter('stimulus_patch_center_azimuth', default=0, limits=(-180, 180), step_size=1,
                                                          static=True)
    stimulus_patch_center_elevation = visual.FloatParameter('stimulus_patch_center_elevation', default=1, limits=(-90, 90), step_size=1,
                                                            static=True)
    stimulus_patch_diameter = visual.FloatParameter('stimulus_patch_diameter', default=30, limits=(1, 180), step_size=1, static=True)
    grating_direction_rotation = GratingDirectionRotation('grating_direction_rotation', internal=True, static=True)
    stimulus_patch_rotation = StimulationPatchRotation('stimulus_patch_rotation', internal=True, static=True)

    # Paths to shaders
    VERT_PATH = './local_translation_grating.vert'
    FRAG_PATH = './local_translation_grating_round_area.frag'

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
        self.grating_angular_velocity.connect(self.grating)
        self.grating_angular_period.connect(self.grating)
        self.stimulus_patch_diameter.connect(self.grating)
        self.grating_direction_rotation.connect(self.grating)
        self.stimulus_patch_rotation.connect(self.grating)

        # Connect to location coordinates
        self.stimulus_patch_center_azimuth.add_downstream_link(self.grating_direction_rotation)
        self.stimulus_patch_center_elevation.add_downstream_link(self.grating_direction_rotation)
        self.stimulus_patch_center_azimuth.add_downstream_link(self.stimulus_patch_rotation)
        self.stimulus_patch_center_elevation.add_downstream_link(self.stimulus_patch_rotation)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer

        # Set rf_location at the equator directly in front
        azim = 0
        elev = 0
        patch_center = np.array([np.sin(azim) * np.cos(elev),
                                 np.cos(azim) * np.cos(elev),
                                 np.sin(elev)])
        self.grating['stimulus_patch_center'] = patch_center

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)
