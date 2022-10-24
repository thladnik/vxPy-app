"""
./vxpy/setup/res/routines/camera/detect_particles.py
Copyright (C) 2021 Tim Hladnik

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

import vxpy.core.ui
from vxpy.api.attribute import ArrayAttribute, ArrayType
from vxpy.core.dependency import require_camera_device
from vxpy.api.routine import CameraRoutine


class CalculateControlCamPixelSum(CameraRoutine):

    camera_id = 'faraday_control_cam'

    def __init__(self, *args, **kwargs):
        CameraRoutine.__init__(self, *args, **kwargs)

        # (optional) Make sure right camera is configured (easier debugging)
        require_camera_device(self.camera_id)

    def setup(self):
        # Create an array attribute to store output image in
        self.camera_pixel_sum = ArrayAttribute('control_cam_pixel_sum', (1, ), ArrayType.uint64)

    def initialize(self):
        # Mark output array attribute as something to be written to file
        self.camera_pixel_sum.add_to_file()
        vxpy.core.gui.register_with_plotter('control_cam_pixel_sum')

    def main(self, *args, **frames):
        # Read frame
        frame = frames.get(self.camera_id)

        # Make sure there is a frame
        if frame is None:
            return
        # Write frame with drawn contours to attribute
        self.camera_pixel_sum.write(frame.sum())
