import time

import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.routine as vxroutine
import vxpy.core.ui as vxui
import vxpy.core.devices.serial as vxserial


class KebabPositionTracker(vxroutine.IoRoutine):

    steps_to_full_rotation: int = 3200
    callback_is_set = False
    rotation_num = 0
    wait_for_position = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vxattribute.ArrayAttribute('stepper_position', (1,), vxattribute.ArrayType.int64)
        vxattribute.ArrayAttribute('stepper_rotation_number', (1,), vxattribute.ArrayType.uint64)
        vxattribute.ArrayAttribute('stepper_full_rotation_trigger', (1,), vxattribute.ArrayType.bool)

    def initialize(self):
        vxui.register_with_plotter('stepper_position', axis='stepper position')
        vxui.register_with_plotter('stepper_rotation_number')
        vxui.register_with_plotter('stepper_full_rotation_trigger')
        vxattribute.write_to_file(self, 'stepper_position')
        vxattribute.write_to_file(self, 'stepper_rotation_number')
        vxattribute.write_to_file(self, 'stepper_full_rotation_trigger')
        self.i = 0

    def main(self):
        self.device = vxserial.get_serial_device_by_id('Dev_kebab')
        # if self.i % 1 == 0:
        if not self.wait_for_position:
            self.device.board.stepper_get_current_position(self.device.motor, self._current_position_callback)
            self.wait_for_position = True

    def _current_position_callback(self, position: list):
        vxattribute.write_attribute('stepper_position', position[2])

        rotation_num = position[2] // self.steps_to_full_rotation

        vxattribute.write_attribute('stepper_full_rotation_trigger', rotation_num > self.rotation_num)
        self.rotation_num = rotation_num

        vxattribute.write_attribute('stepper_rotation_number', self.rotation_num)

        self.wait_for_position = False
