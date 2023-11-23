import vxpy.core.protocol as vxprotocol
from controls import fish_kebab_controls
import vxpy.core.event as vxevent


class VORTestTriggered(vxprotocol.TriggeredProtocol):

    rounds = 6
    direction = 1

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        self.set_phase_trigger(trigger)

        for cycle in range(2):

            # chill phase
            # chill phase
            chill_phase = vxprotocol.Phase(duration=20)
            chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                                    'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(chill_phase)

            # 90°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(90, self.rounds))
            phase.set_control(fish_kebab_controls.Control01, {'velocity': 90, 'direction': self.direction, 'rounds': self.rounds,
                                                              'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=20)
            chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                                    'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(chill_phase)

            # 45°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(45, self.rounds))
            phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': self.direction, 'rounds': self.rounds,
                                                              'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=20)
            chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                                    'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(chill_phase)

            # 22.5°/s
            phase = vxprotocol.Phase(duration=self._min_phase_duration(22.5, self.rounds))
            phase.set_control(fish_kebab_controls.Control01, {'velocity': 22.5, 'direction': self.direction, 'rounds': self.rounds,
                                                              'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(phase)

            # chill phase
            chill_phase = vxprotocol.Phase(duration=20)
            chill_phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                                    'led_left_state': 0., 'led_right_state': 0.})
            self.add_phase(chill_phase)

            self.direction *= -1


    def _min_phase_duration(self, velocity, rounds):
        return int(rounds * (360/velocity))



class Protocol01Static(vxprotocol.StaticProtocol):

    def create(self):

        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': 1, 'rounds': 3,
                                                          'led_left_state': 1., 'led_right_state': 0.})
        self.add_phase(phase)

        phase = vxprotocol.Phase(duration=30)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 45, 'direction': -1, 'rounds': 3,
                                                          'led_left_state': 1., 'led_right_state': 0.})
        self.add_phase(phase)


class Protocol01Triggered(vxprotocol.TriggeredProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.TriggeredProtocol.__init__(self, *args, **kwargs)

        # trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        trigger = vxevent.OnTrigger('stepper_full_rotation_trigger')
        self.set_phase_trigger(trigger)

        # phase 1
        phase = vxprotocol.Phase(duration=self._min_phase_duration(90, 3))
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 90, 'direction': 1, 'rounds': 3,
                                                          'led_left_state': 1., 'led_right_state': 0.})
        self.add_phase(phase)

        # phase 2
        phase = vxprotocol.Phase(duration=self._min_phase_duration(10, 1))
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 10, 'direction': -1, 'rounds': 1,
                                                          'led_left_state': 1., 'led_right_state': 1.})
        self.add_phase(phase)

        # phase 3
        phase = vxprotocol.Phase(duration=5)
        phase.set_control(fish_kebab_controls.Control01, {'velocity': 0, 'direction': 1, 'rounds': 0,
                                                          'led_left_state': 0., 'led_right_state': 0.})
        self.add_phase(phase)


    def _min_phase_duration(self, velocity, rounds):
        return int(rounds * (360/velocity))
