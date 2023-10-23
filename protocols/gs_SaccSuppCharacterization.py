import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.gs_characterization_stims.spherical_grating import SphericalBlackWhiteGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground

# define Parameter functions
def paramsMovGrat(waveform, motion_type, ang_vel, ang_period):
    return {
        SphericalBlackWhiteGrating.waveform: waveform,
        SphericalBlackWhiteGrating.motion_type: motion_type,
        SphericalBlackWhiteGrating.angular_velocity: ang_vel,
        SphericalBlackWhiteGrating.angular_period: ang_period
    }


class CharacterizationProtocol(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # set fixed parameters SF Tuning
        sf_waveform = 'rectangular'
        sf_motion_type = 'rotation'
        sf_motion_axis = 'vertical'
        sf_temp_freq = 4    # Hz
        sf_phase_dur = 6 # sec

        # Experimental Conditions:
        # 5 SFs [0.011, 0.022, 0.044, 0.088, 0.177 cyc/°] = angular periods [90, 45, 22.5, 11.25, 5.625 °/cyc]
        angular_periods = [90, 45, 22.5, 11.25, 5.625]

        # Add pre-phase (5 sec uniform grey)
        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.array([.5, .5, .5])})
        self.add_phase(p)

        for ang_per in angular_periods:
            self.global_visual_props['azim_angle'] = 0.
            # Static Phase, 0 pi phase shifted
            p = vxprotocol.Phase(sf_phase_dur)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type,0,
                                                                   ang_per))
            self.add_phase(p)

            # Moving Phase CW_1
            p = vxprotocol.Phase(sf_phase_dur + 0.0625) # produces +0.5 pi phase shift, by running 0.0625 sec longer (at 4Hz TF, 1/4 cycle = 0.0625s)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, ang_per*sf_temp_freq,
                                                                   ang_per))
            self.add_phase(p)

            # Static Phase, 0.5 pi phase shifted
            p = vxprotocol.Phase(sf_phase_dur)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, 0, ang_per))
            self.add_phase(p)

            # Moving Phase CCW_1
            p = vxprotocol.Phase(sf_phase_dur + 0.125) # produces -1 pi phase shift, by running 0.125 sec longer (at 4Hz TF, 1/2 cycle = 0.125s)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, -ang_per * sf_temp_freq,
                                                                   ang_per))
            self.add_phase(p)

            # Static Phase, -0.5 pi == 1.5 pi phase shifted
            p = vxprotocol.Phase(sf_phase_dur)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, 0, ang_per))
            self.add_phase(p)

            # Moving Phase CW_2
            p = vxprotocol.Phase(sf_phase_dur - 0.0625)  # produces -0.5 pi phase shift, by running 0.0625 sec shorter (at 4Hz TF, 1/4 cycle = 0.0625s)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, ang_per * sf_temp_freq,
                                                                   ang_per))
            self.add_phase(p)

            # Static Phase, -1 pi == 1 pi phase shifted
            p = vxprotocol.Phase(sf_phase_dur)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, 0, ang_per))
            self.add_phase(p)

            # Moving Phase CCW_2
            p = vxprotocol.Phase(sf_phase_dur + 0.125)  # produces -1 pi phase shift, by running 0.125 sec longer (at 4Hz TF, 1/2 cycle = 0.125s)
            p.set_visual(SphericalBlackWhiteGrating, paramsMovGrat(sf_waveform, sf_motion_type, -ang_per * sf_temp_freq,
                                                                   ang_per))
            self.add_phase(p)
