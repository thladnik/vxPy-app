"""Example routine implementation for reading frames from a camera device and calculating an output value
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
