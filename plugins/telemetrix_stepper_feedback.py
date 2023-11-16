import time

import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.routine as vxroutine
import vxpy.core.ui as vxui
import vxpy.core.devices.serial as vxserial


class KebabPositionTracker(vxroutine.IoRoutine):

    steps_to_full_rotation: int = 3200
    callback_is_set = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vxattribute.ArrayAttribute('stepper_position', (1,), vxattribute.ArrayType.int64)
        vxattribute.ArrayAttribute('stepper_full_rotation_trigger', (1,), vxattribute.ArrayType.bool)

    def initialize(self):
        vxui.register_with_plotter('stepper_position', axis='stepper position')
        vxui.register_with_plotter('stepper_full_rotation_trigger')
        self.i = 0

    def main(self):
        # if not self.callback_is_set:
        self.device = vxserial.get_serial_device_by_id('Dev_kebab')
        if self.i % 10 == 0:
            self.device.board.stepper_get_current_position(self.device.motor, self._current_position_callback)

        if self.i % 1000 == 0:
            self.device.board.stepper_set_max_speed(self.device.motor, 1000)
            self.device.board.stepper_set_speed(self.device.motor, 50 * np.random.randint(10))
            self.device.board.stepper_run_speed(self.device.motor)
        self.i += 1
        # print('Muh')

    def _current_position_callback(self, position: list):
        print(position)
        vxattribute.write_attribute('stepper_position', position[2])
        vxattribute.write_attribute('stepper_full_rotation_trigger', position[2] % self.steps_to_full_rotation)
