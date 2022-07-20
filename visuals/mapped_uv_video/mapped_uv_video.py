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
import os
import cv2
from vispy import gloo
import numpy as np

from vxpy.core import visual
from vxpy.utils import sphere
import vxpy.core.logger as vxlogger

log = vxlogger.getLogger(__name__)


class VideoTexture(visual.TextureUInt2D):

    def __init__(self, *args, **kwargs):
        visual.TextureUInt2D.__init__(self, *args, **kwargs)

    def load_video(self, input_video_path):

        # Load video
        # input_video_path = './visuals/mapped_uv_video/Aug 15_Insta1_Lower Kalambo River_T60.mp4'  # low contrast
        # input_video_path = './visuals/mapped_uv_video/Aug18_insta1_kalamboLodgeHarbor_R+50.mp4'  # higher contrast
        # input_video_path = './visuals/mapped_uv_video/Aug 19_Insta1_Isanga Bay_R+50.mp4'  #
        if not os.path.exists(input_video_path):
            log.warning(f'Video file {input_video_path} not found')
            return

        self.capture = cv2.VideoCapture(input_video_path)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.data = np.random.randint(0, 256, size=(self.height, self.width, 3))
        self.last_time = 0.0

    def upstream_updated(self):

        current_time = NaturalVideo.time.data[0]
        if current_time != 0.0 and current_time < self.last_time + 1./self.fps:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # Histogram equalization
        r = cv2.equalizeHist(frame[:, :, 0])
        g = cv2.equalizeHist(frame[:, :, 1])
        b = cv2.equalizeHist(frame[:, :, 2])
        equ_frame = np.stack([r, g, b], axis=-1)

        # Write
        # self.data = frame
        self.data = equ_frame

        # Save time for next frame
        self.last_time = current_time


# class NaturalVideo(visual.SphericalVisual):
class NaturalVideo(visual.BaseVisual):

    input_video_path = ''

    # Define parameters
    time = visual.FloatParameter('time', internal=True)
    # video_texture = visual.Texture2D('video_texture', internal=True, static=True)
    video_texture = VideoTexture('video_texture', internal=True, static=True)

    # Paths to shaders
    VERT_PATH = './sphere.vert'
    FRAG_PATH = './texture_sphere.frag'

    def __init__(self, *args, **kwargs):
        # visual.SphericalVisual.__init__(self, *args, **kwargs)
        visual.BaseVisual.__init__(self, *args, **kwargs)

        # Set up 3d model of sphere
        self.sphere = sphere.UVSphere(azim_lvls=60, elev_lvls=30, upper_elev=np.pi/2)
        self.uv_texture_coords = self.sphere.get_uv_coordinates()
        self.index_buffer = gloo.IndexBuffer(self.sphere.indices)
        self.position_buffer = gloo.VertexBuffer(self.sphere.a_position)

        self.video_texture.load_video(self.input_video_path)

        # Set up program
        self.grating = gloo.Program(self.load_vertex_shader(self.VERT_PATH), self.load_shader(self.FRAG_PATH))

        # Connect parameters (this makes them be automatically updated in the connected programs)
        self.time.connect(self.grating)
        self.video_texture.connect(self.grating)
        self.time.add_downstream_link(self.video_texture)

    def initialize(self, **params):
        # Reset u_time to 0 on each visual initialization
        self.time.data = 0.0

        # Set positions with buffers
        self.grating['a_position'] = self.position_buffer
        self.grating['a_texture_coord'] = np.ascontiguousarray(self.uv_texture_coords)

    def render(self, dt):
        # Add elapsed time to u_time
        self.time.data += dt

        # Apply default transforms to the program for mapping according to hardware calibration
        self.apply_transform(self.grating)

        # Draw the actual visual stimulus using the indices of the  triangular faces
        self.grating.draw('triangles', self.index_buffer)


class NaturalVideoRipplesOnSand(NaturalVideo):

    input_video_path = './visuals/mapped_uv_video/Aug 15_Insta1_Lower Kalambo River_T60.mp4'

    def __init__(self, *args, **kwargs):
        NaturalVideo.__init__(self, *args, **kwargs)


class NaturalVideoRipplesOnStones(NaturalVideo):

    input_video_path = './visuals/mapped_uv_video/Aug 19_Insta1_Isanga Bay_R+50.mp4'

    def __init__(self, *args, **kwargs):
        NaturalVideo.__init__(self, *args, **kwargs)


class NaturalVideoWithVegetation(NaturalVideo):

    input_video_path = './visuals/mapped_uv_video/Aug18_insta1_kalamboLodgeHarbor_R+50.mp4'

    def __init__(self, *args, **kwargs):
        NaturalVideo.__init__(self, *args, **kwargs)
