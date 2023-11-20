from telemetrix import telemetrix

from vxpy.devices import telemetrix_device


class TelemetrixStepperKebab(telemetrix_device.Telemetrix):

    led_ir_back1 = 11
    led_ir_back2 = 6
    led_right = 10
    led_left = 9

    PULSE_PIN = 5
    DIRECTION_PIN = 2

    def _start(self) -> bool:

        self.board.set_pin_mode_digital_output(self.led_ir_back1)
        self.board.set_pin_mode_digital_output(self.led_ir_back2)
        self.board.set_pin_mode_analog_output(self.led_right)
        self.board.set_pin_mode_analog_output(self.led_left)

        self.motor = self.board.set_pin_mode_stepper(interface=1, pin1=self.PULSE_PIN, pin2=self.DIRECTION_PIN)
        self.board.stepper_stop(self.motor)
        self.board.digital_write(self.led_ir_back1, 1)
        self.board.digital_write(self.led_ir_back2, 1)
        self.board.analog_write(self.led_right, 0)
        self.board.analog_write(self.led_left, 0)

        return True

    def _end(self) -> bool:

        self.board.stepper_stop(self.motor)

        return True
