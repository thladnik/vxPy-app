from typing import Type

import numpy as np

import vxpy.core.protocol as vxprotocol
from visuals.cmn_redesign import ContiguousMotionNoise3D, CMN3D20240410, CMN3D20240411
from visuals.spherical_global_motion import TranslationGrating, RotationGrating
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


class CMN3DBaseProtocol(vxprotocol.StaticProtocol):
    cmn_version: Type[ContiguousMotionNoise3D]

    def create(self):

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)

        # Grey
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        # CMN
        phase = vxprotocol.Phase(duration=20 * 60)
        phase.set_visual(self.cmn_version)
        self.add_phase(phase)

        # Grey
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(p)

        # Translation characterization
        for i in range(3):
            for azim in np.arange(-180, 180, 30):
                phase = vxprotocol.Phase(duration=6)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 0})
                self.add_phase(phase)

                phase = vxprotocol.Phase(duration=4)
                phase.set_visual(TranslationGrating,
                                 {TranslationGrating.azimuth: azim,
                                  TranslationGrating.elevation: 0.0,
                                  TranslationGrating.angular_period: 20,
                                  TranslationGrating.angular_velocity: 30})
                self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


from visuals.cmn_redesign import CMN3D20240411
class CMN3DProtocol20240411(CMN3DBaseProtocol):
    cmn_version = CMN3D20240411


from visuals.cmn_redesign import CMN3D20240606Vel140Scale7
class CMN3DProtocol20240607(CMN3DBaseProtocol):
    cmn_version = CMN3D20240606Vel140Scale7


class CMN3DRotAndTrans(vxprotocol.StaticProtocol):
    cmn_segment_duration = 5 * 60  # s
    cmn_intersegment_duration = 10  # s

    def create(self):


        for i in range(6):
            # Grey
            gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
            gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
            self.add_phase(gray_phase)

            # CMN
            phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
            phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: int(i == 0)})
            self.add_phase(phase)

            # Grey
            gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
            gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
            self.add_phase(gray_phase)

            if i < 3:
                # Translation
                for azim in np.arange(-180, 180, 30):
                    p = vxprotocol.Phase(duration=6)
                    p.set_visual(TranslationGrating,
                                      {TranslationGrating.azimuth: azim,
                                       TranslationGrating.elevation: 0.0,
                                       TranslationGrating.angular_period: 20,
                                       TranslationGrating.angular_velocity: 0})
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=4)
                    p.set_visual(TranslationGrating,
                                      {TranslationGrating.azimuth: azim,
                                       TranslationGrating.elevation: 0.0,
                                       TranslationGrating.angular_period: 20,
                                       TranslationGrating.angular_velocity: 30})
                    self.add_phase(p)
            else:
                for azim in np.linspace(0, 180, 5):
                    for elev in [-45, 0, 45, 90]:
                        p = vxprotocol.Phase(duration=6)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: 0,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)

                        p = vxprotocol.Phase(duration=4)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: 30,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)


        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


from visuals.cmn_redesign import CMN3D20240606Vel160Scale3LargePatches

class CMN3D20240705LargePatches(vxprotocol.StaticProtocol):
    cmn_segment_duration = 10 * 60  # s
    cmn_intersegment_duration = 15  # s

    def create(self):

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3D20240606Vel160Scale3LargePatches,
                         {CMN3D20240606Vel160Scale3LargePatches.reset_time: 1})
        self.add_phase(phase)

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        for i in range(2):

            # Translation
            for azim in np.arange(-180, 180, 30):
                p = vxprotocol.Phase(duration=6)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 0})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 30})
                self.add_phase(p)

            for azim in np.linspace(0, 180, 5):
                for elev in [-45, 0, 45, 90]:
                    for direction in [-1, 1]:
                        p = vxprotocol.Phase(duration=6)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: 0,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)

                        p = vxprotocol.Phase(duration=4)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: direction * 30,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3D20240606Vel160Scale3LargePatches,
                         {CMN3D20240606Vel160Scale3LargePatches.reset_time: 0})
        self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


from visuals.cmn_redesign import CMN3DOcto20240710

class CMN3DOctoProtocol20240710(vxprotocol.StaticProtocol):
    cmn_segment_duration = 10 * 60  # s
    cmn_intersegment_duration = 15  # s

    def create(self):

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3DOcto20240710,
                         {CMN3DOcto20240710.reset_time: 1})
        self.add_phase(phase)

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        for i in range(2):

            # Translation
            for azim in np.arange(-180, 180, 30):
                p = vxprotocol.Phase(duration=6)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 0})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 30})
                self.add_phase(p)

            for azim in np.linspace(0, 180, 5):
                for elev in [-45, 0, 45, 90]:
                    for direction in [-1, 1]:
                        p = vxprotocol.Phase(duration=6)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: 0,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)

                        p = vxprotocol.Phase(duration=4)
                        p.set_visual(RotationGrating,
                                     {RotationGrating.azimuth: azim,
                                      RotationGrating.elevation: elev,
                                      RotationGrating.angular_period: 20,
                                      RotationGrating.angular_velocity: direction * 30,
                                      RotationGrating.waveform: 'rect'})
                        self.add_phase(p)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3DOcto20240710,
                         {CMN3DOcto20240710.reset_time: 0})
        self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)


class CMN3DOctoProtocol20240710WithPause(vxprotocol.StaticProtocol):
    cmn_segment_duration = 10 * 60  # s
    cmn_intersegment_duration = 15  # s

    def create(self):

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3DOcto20240710,
                         {CMN3DOcto20240710.reset_time: 1})
        self.add_phase(phase)

        # Grey
        gray_phase = vxprotocol.Phase(self.cmn_intersegment_duration)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        for i in range(2):

            # Translation
            for azim in np.arange(-180, 180, 30):
                p = vxprotocol.Phase(duration=6)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 0})
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(TranslationGrating,
                                  {TranslationGrating.azimuth: azim,
                                   TranslationGrating.elevation: 0.0,
                                   TranslationGrating.angular_period: 20,
                                   TranslationGrating.angular_velocity: 30})
                self.add_phase(p)

        # Grey (long pause)
        gray_phase = vxprotocol.Phase(duration=10*60)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=self.cmn_segment_duration)
        phase.set_visual(CMN3DOcto20240710,
                         {CMN3DOcto20240710.reset_time: 0})
        self.add_phase(phase)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)

class CMN3DRotAndTrans20240718Final(vxprotocol.StaticProtocol):

    def create(self):
        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 60)
        phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: 1})
        self.add_phase(phase)

        # Black
        gray_phase = vxprotocol.Phase(duration=5 * 60)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(gray_phase)

        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        # CMN
        phase = vxprotocol.Phase(duration=10 * 60)
        phase.set_visual(CMN3D20240606Vel140Scale7, {CMN3D20240606Vel140Scale7.reset_time: 0})
        self.add_phase(phase)

        # Grey
        gray_phase = vxprotocol.Phase(duration=10)
        gray_phase.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.5, 0.5, 0.5]})
        self.add_phase(gray_phase)

        for i in range(2):

            # Translation
            for azim in np.arange(-180, 180, 45):
                for elev in [-90, -45, 0, 45, 90]:
                    p = vxprotocol.Phase(duration=4)
                    p.set_visual(TranslationGrating,
                                      {TranslationGrating.azimuth: azim,
                                       TranslationGrating.elevation: elev,
                                       TranslationGrating.angular_period: 20,
                                       TranslationGrating.angular_velocity: 0})
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=4)
                    p.set_visual(TranslationGrating,
                                      {TranslationGrating.azimuth: azim,
                                       TranslationGrating.elevation: elev,
                                       TranslationGrating.angular_period: 20,
                                       TranslationGrating.angular_velocity: 30})
                    self.add_phase(p)


        for i in range(2):
            for azim in np.arange(-180, 180, 45):
                for elev in [-90, -45, 0, 45, 90]:
                    p = vxprotocol.Phase(duration=4)
                    p.set_visual(RotationGrating,
                                 {RotationGrating.azimuth: azim,
                                  RotationGrating.elevation: elev,
                                  RotationGrating.angular_period: 20,
                                  RotationGrating.angular_velocity: 0,
                                  RotationGrating.waveform: 'rect'})
                    self.add_phase(p)

                    p = vxprotocol.Phase(duration=4)
                    p.set_visual(RotationGrating,
                                 {RotationGrating.azimuth: azim,
                                  RotationGrating.elevation: elev,
                                  RotationGrating.angular_period: 20,
                                  RotationGrating.angular_velocity: 30,
                                  RotationGrating.waveform: 'rect'})
                    self.add_phase(p)

        # Black
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground, {SphereUniformBackground.u_color: [0.0, 0.0, 0.0]})
        self.add_phase(p)

