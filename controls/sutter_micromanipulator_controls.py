import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer
from vxpy.devices.sutter_mp285 import SutterMP285
import vxpy.core.container as vxcontainer


class ControlSutter(vxcontrol.BaseControl):

    device: SutterMP285
    move_to_x: int = None
    move_to_y: int = None
    move_to_z: int = None

    #zero_position = 0

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_sutter')
        self.device.open()
        self.device.set_absolute_mode()
        self.device.set_velocity(200, 10)
        #self.zero_position = self.device.get_position()

    def initialize(self, **kwargs):

        start_xyz_position = self.device.get_position()
        self.device.update_panel()
        self.device.move_to_position(start_xyz_position + [self.move_to_x, self.move_to_y, self.move_to_z])
        self.device.update_panel()
        end_xyz_position = self.device.get_position()

        vxcontainer.add_phase_attributes({'sutter_start_pos_xyz': start_xyz_position, 'sutter_end_pos_xyz': end_xyz_position})

    def main(self, dt: float, **pins):
        pass

    def _end(self):
        print('-----END PHASE')





