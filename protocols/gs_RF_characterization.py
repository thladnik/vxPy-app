import numpy as np
import vxpy.core.protocol as vxprotocol
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground
from visuals.jitter_noise import BinaryBlackWhiteJitterNoise30deg
from visuals.spherical_grating import SphericalBlackWhiteGrating
from visuals.gs_characterization_stims.chirp import LogChirp
from visuals.spherical_global_motion.motion_in_sphere import RotationGrating
from visuals.cmn_redesign import CMN3D20240606Vel140Scale7Long


def paramsTranslation(waveform, motion_type, motion_axis, ang_vel, ang_period):
    return {
        SphericalBlackWhiteGrating.waveform: waveform,
        SphericalBlackWhiteGrating.motion_type: motion_type,
        SphericalBlackWhiteGrating.motion_axis: motion_axis,
        SphericalBlackWhiteGrating.angular_velocity: ang_vel,
        SphericalBlackWhiteGrating.angular_period: ang_period
    }

def paramsChirp(base_lum, sine_amp, f_start, f_final, chirp_dur):
    return {
        LogChirp.baseline_lum: base_lum,
        LogChirp.sine_amp: sine_amp,
        LogChirp.starting_freq: f_start,
        LogChirp.final_freq: f_final,
        LogChirp.chirp_duration: chirp_dur
    }

def paramsRotation(elev, azim, ang_vel, ang_per, waveform):
    return {
        RotationGrating.elevation: elev,
        RotationGrating.azimuth: azim,
        RotationGrating.angular_velocity: ang_vel,
        RotationGrating.angular_period: ang_per,
        RotationGrating.waveform: waveform
    }

class RF_characterization(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        self.global_visual_props['azim_angle'] = 0.

        # set fixed parameters CMN
        cmn_duration = 30 * 60  # sec

        # set fixed parameters binary jitter
        jitter_duration = 30 * 60 # sec

        # set fixed parameters On/Off
        onoff_duration = 12 #s ec

        # set fixed parameters chirp
        chirp_base_lum = 0.5
        chirp_sine_amp = 0.5
        chirp_f_start = 10  # Hz
        chirp_f_final = 0.5  # Hz
        chirp_dur = 20  # sec
        chirp_pause_dur = 6  # sec

        # set fixed parameters rotation
        rot_waveform = 'rect'
        rot_ang_per = 45    # °/cyc --> based on peak response found by DC
        rot_ang_vel = 90   # °/s --> 2Hz temp frequency
        rot_moving_duration = 12 # sec
        rot_pause_duration = 6  # sec

        # set parameters translation
        trans_waveform = 'rectangular'
        trans_motion_type = 'translation'
        trans_ang_per = 45
        trans_ang_vel = 90
        trans_moving_duration = 12
        trans_pause_duration = 6

        # Variable conditions
        onoff_color = [0.5, 1, 0.5, -1]
        rot_conditions = [(90,0,1),(90,0,-1),(0,45,-1),(0,45,1),(0,-45,1),(0,-45,-1)]   #(elev, azim, direction): (90,0,+1/-1) = CW/CCW yaw, (0,45,-1/+1) = LARP+/-, (0,-45,+1/-1) = RALP+/-
        trans_conditions = [('vertical',1),('vertical',-1),('sideways',1),('sideways',-1),('forward',1),('forward',-1)]     # (motion axis, direction)

        # start protocol (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # 30 min CMN
        p = vxprotocol.Phase(cmn_duration)
        p.set_visual(CMN3D20240606Vel140Scale7Long, {CMN3D20240606Vel140Scale7Long.reset_time: 1})
        self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # 30 min binary jitter
        p = vxprotocol.Phase(jitter_duration)
        p.set_visual(BinaryBlackWhiteJitterNoise30deg)
        self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # On/Off
        repeats = 3
        for i in range(repeats):
            for color in onoff_color:
                p = vxprotocol.Phase(onoff_duration)
                p.set_visual(SphereUniformBackground,
                             {SphereUniformBackground.u_color: np.asarray([color, color, color])})
                self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # Chirp
        repeats = 3
        for i in range(repeats):
            # Pause Phase
            p = vxprotocol.Phase(chirp_pause_dur)
            p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
            self.add_phase(p)

            # Chirp Phase
            p = vxprotocol.Phase(chirp_dur)
            p.set_visual(LogChirp, paramsChirp(chirp_base_lum, chirp_sine_amp, chirp_f_start, chirp_f_final, chirp_dur))
            self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # Global Rotation
        repeats = 2
        for i in range(repeats):
            for elev, azim, direction in np.random.permutation(rot_conditions):
                #static phase
                p = vxprotocol.Phase(rot_pause_duration)
                p.set_visual(RotationGrating,paramsRotation(elev, azim, ang_vel=0, ang_per=rot_ang_per,waveform=rot_waveform))
                self.add_phase(p)

                #rotation phase
                p = vxprotocol.Phase(rot_moving_duration)
                p.set_visual(RotationGrating,paramsRotation(elev, azim, ang_vel=rot_ang_vel * direction, ang_per=rot_ang_per,waveform=rot_waveform))
                self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)

        # Global Translation
        repeats = 2
        for i in range(repeats):
            for motion_axis, direction in np.random.permutation(trans_conditions):
                # print(trans_ang_vel, direction, trans_ang_vel * int(direction))
                # static phase
                p = vxprotocol.Phase(trans_pause_duration)
                p.set_visual(SphericalBlackWhiteGrating,paramsTranslation(trans_waveform, trans_motion_type, motion_axis, ang_vel=0, ang_period=trans_ang_per))
                self.add_phase(p)

                # rotation phase
                p = vxprotocol.Phase(trans_moving_duration)
                p.set_visual(SphericalBlackWhiteGrating,paramsTranslation(trans_waveform, trans_motion_type, motion_axis, ang_vel= trans_ang_vel * int(direction), ang_period=trans_ang_per))
                self.add_phase(p)

        # Pause phase (10 sec uniform grey)
        p = vxprotocol.Phase(10)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: np.asarray([.5, .5, .5])})
        self.add_phase(p)




