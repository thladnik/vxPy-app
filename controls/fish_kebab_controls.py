import numpy as np

import vxpy.core.attribute as vxattribute
import vxpy.core.control as vxcontrol
import vxpy.core.devices.serial as vxserial
import vxpy.core.container as vxcontainer
from devices.telemetrix_stepper import TelemetrixStepperKebab
import vxpy.core.container as vxcontainer


class Control01(vxcontrol.BaseControl):

    device: TelemetrixStepperKebab
    velocity: int = None
    direction: int = None
    led_left_state = 0.
    led_right_state = 0.

    def __init__(self, *args, **kwargs):
        vxcontrol.BaseControl.__init__(self, *args, **kwargs)

        self.device = vxserial.get_serial_device_by_id('Dev_kebab')

    @staticmethod
    def _led_state_output(value: float) -> int:
        return int(value * 2**16)

    def initialize(self, **kwargs):
        self.device.board.stepper_set_max_speed(self.device.motor, 1000)
        self.device.board.stepper_set_speed(self.device.motor, 50 * self.velocity)
        self.device.board.stepper_run_speed(self.device.motor)

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        # Add datasets to be saved to file
        vxcontainer.add_phase_attributes({'velocity': self.velocity, 'direction': self.direction})
        vxcontainer.create_phase_dataset('led_left', (1,), np.float32)
        vxcontainer.create_phase_dataset('led_right', (1,), np.float32)

    def main(self, dt, **pins):

        self.device.board.analog_write(self.device.led_left, self._led_state_output(self.led_left_state))
        self.device.board.analog_write(self.device.led_right, self._led_state_output(self.led_right_state))

        vxcontainer.add_to_phase_dataset('led_left', self.led_left_state)
        vxcontainer.add_to_phase_dataset('led_right', self.led_right_state)
