import vxprotocol
from vxvisuals import SphericalBlackWhiteGrating, SphereUniformBackground
from vxhardware import LedPWMControl

class OKR_StimulusWithUpdatedLightSequence(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(self):
        phase_duration = 8  # Dauer pro Phase (Sekunden)

        # Neue, erweiterte Lichtsequenz mit mehr Nullen
        light_sequence = [
            0, 1, 0, 0, 0.8, 0, 0, 1, 0, 0, 0.8, 0,
            0, 1, 0, 0, 0.6, 0, 0, 1, 0, 0, 0.6, 0,
            0, 1, 0, 0, 0.3, 0, 0, 1, 0, 0, 0.3, 0,
            0, 1, 0
        ]

        # OKR Stimulus-Parameter
        angular_period = 1 / 0.05
        velocity_A = 10
        velocity_B = -10

        def add_okr_phase(velocity, intensity):
            p = vxprotocol.Phase(duration=phase_duration)
            p.set_control(LedPWMControl, {'light_intensity': intensity})
            p.set_visual(SphericalBlackWhiteGrating, {
                SphericalBlackWhiteGrating.waveform: 'rectangular',
                SphericalBlackWhiteGrating.motion_axis: 'vertical',
                SphericalBlackWhiteGrating.motion_type: 'rotation',
                SphericalBlackWhiteGrating.angular_period: angular_period,
                SphericalBlackWhiteGrating.angular_velocity: velocity
            })
            self.add_phase(p)

        def add_pause():
            p = vxprotocol.Phase(duration=phase_duration)
            p.set_control(LedPWMControl, {'light_intensity': 0})
            p.set_visual(SphereUniformBackground, {
                SphereUniformBackground.u_color: (0., 0., 0.)
            })
            self.add_phase(p)

        # === Wiederhole gesamte Sequenz 3-mal ===
        for repeat in range(3):
            current_velocity = velocity_A  # Beginne jeden Durchlauf mit Richtung A

            i = 0
            while i + 2 < len(light_sequence):
                group = light_sequence[i:i+3]
                if group[0] == 0 and group[2] == 0:
                    # Gültiger Stimulusblock [0, x, 0]
                    for intensity in group:
                        add_okr_phase(current_velocity, intensity)
                    add_pause()
                    current_velocity *= -1  # Richtungswechsel
                    i += 3  # Weiter zur nächsten Gruppe
                else:
                    i += 1  # Überspringe, wenn keine gültige 3er-Gruppe

        self.add_phase(p)







