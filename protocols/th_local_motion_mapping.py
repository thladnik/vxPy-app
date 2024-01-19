"""
vxpy_app ./protocols/spherical_gratings.py
Copyright (C) 2020 Tim Hladnik

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os

import numpy as np
import scipy.io

import vxpy.core.protocol as vxprotocol
from vxpy.utils.sphere import IcosahedronSphere
from vxpy.utils.geometry import cart2sph1

from visuals.local_translation_grating import LocalTranslationGrating_RoundArea
from vxpy.visuals.spherical_uniform_background import SphereUniformBackground


def create_params(period, velocity, center_az, center_el, field_diameter):
    return {
        LocalTranslationGrating_RoundArea.grating_angular_period: period,
        LocalTranslationGrating_RoundArea.grating_angular_velocity: velocity,
        LocalTranslationGrating_RoundArea.stimulus_patch_center_azimuth: center_az,
        LocalTranslationGrating_RoundArea.stimulus_patch_center_elevation: center_el,
        LocalTranslationGrating_RoundArea.stimulus_patch_diameter: field_diameter
    }


class ProtocolRE(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        selected = np.logical_and(az > 0.0, el <= 25)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            combos.append((15, *coords))
            combos.append((40, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolLE(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = 30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        selected = np.logical_and(az < 0.0, el <= 25)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            combos.append((15, *coords))
            combos.append((40, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolRE20221223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az > -22.5, el <= (45 - min(diameters) / 2))
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if coords[0] == 0.0 and coords[1] == 0.0:
                continue

            for dia in diameters:
                combos.append((dia, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolLE20221223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az < 22.5, el <= (45 - min(diameters) / 2))
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140Demo(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_and(az > -50, az < 90), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]
        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140RightEye20230216(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_or(az > -30, az < -(180 - 30)), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140LeftEye20230216(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [25]
        selected = np.logical_and(np.logical_or(az < 30, az > (180 - 30)), el < 45.0)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for a, e in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters:
                combos.append((dia, a, e, vsign * base_velocity))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40RightEye20230223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [45]
        selected_sparse = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40LeftEye20230223(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        self.combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                self.combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [45]
        selected_sparse = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                self.combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(self.combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

class RepulsiveSphere140and40LeftEyeLargePatches20230810(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        self.combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                self.combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [60]
        selected_sparse = np.logical_and(np.logical_or(az < 40, az > (180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                self.combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(self.combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40RightEyeLargePatches20230810(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_dense = [25]
        selected_dense = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_dense = az[selected_dense]
        selected_el_dense = el[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters_sparse = [60]
        selected_sparse = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        selected_az_sparse = az[selected_sparse]
        selected_el_sparse = el[selected_sparse]

        for a, e in zip(selected_az_sparse, selected_el_sparse):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_sparse:
                combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class RepulsiveSphere140and40RightEyeLargePatchesOrderedAzimuths20231124(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        base_velocity = 30
        repeat_num = 3

        # Create list for all combos
        combos = []

        # Densely sampled stimuli
        verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_140.mat')['ans'][0, 0]
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180


        diameters_dense = [25]
        selected_range = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        az_selected = az[selected_range]
        el_selected = el[selected_range]

        # Rectify
        az_sort = np.copy(az_selected)
        az_sort[-90 > az_sort] = 360 + az_sort[-90 > az_sort]

        # Sort by azimuth
        selected_dense = np.argsort(az_sort)

        selected_az_dense = az_selected[selected_dense]
        selected_el_dense = el_selected[selected_dense]

        for a, e in zip(selected_az_dense, selected_el_dense):
            # Skip 0/0, as that is a singularity
            if a == 0.0 and e == 0.0:
                continue

            # Set sign of velocity
            if a >= 0:
                vsign = -1
            else:
                vsign = 1

            for dia in diameters_dense:
                combos.append((dia, a, e, vsign * base_velocity))

        # Sparsely sampled stimuli
        # verts = scipy.io.loadmat('./protocols/th_local_motion_mapping_data/repulsive_sphere_40.mat')['ans'][0, 0]
        # az, el = cart2sph1(*verts.T)
        # az = az / np.pi * 180
        # el = el / np.pi * 180
        #
        # diameters_sparse = [60]
        # selected_sparse = np.logical_and(np.logical_or(az > -40, az < -(180 - 15)), el < 45.0)
        # selected_az_sparse = az[selected_sparse]
        # selected_el_sparse = el[selected_sparse]
        #
        # for a, e in zip(selected_az_sparse, selected_el_sparse):
        #     # Skip 0/0, as that is a singularity
        #     if a == 0.0 and e == 0.0:
        #         continue
        #
        #     # Set sign of velocity
        #     if a >= 0:
        #         vsign = -1
        #     else:
        #         vsign = 1
        #
        #     for dia in diameters_sparse:
        #         combos.append((dia, a, e, vsign * base_velocity))

        # Shuffle all
        # np.random.seed(1)
        # shuffled_combos = np.random.permutation(combos)

        # Create protocol phases
        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(repeat_num):
            for size, az, el, vel in combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, vel, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(15)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)


class ProtocolRE_DEMO(vxprotocol.StaticProtocol):

    def __init__(self, *args, **kwargs):
        vxprotocol.StaticProtocol.__init__(self, *args, **kwargs)

        period = 20
        velocity = -30

        sphere = IcosahedronSphere(2)
        verts = sphere.get_vertices()
        az, el = cart2sph1(*verts.T)
        az = az / np.pi * 180
        el = el / np.pi * 180

        diameters = [5, 35]
        selected = np.logical_and(az > -22.5, el <= (45 - min(diameters) / 2))
        selected = np.logical_and(az < 90.0, selected)
        selected_az = az[selected]
        selected_el = el[selected]

        combos = []
        for coords in zip(selected_az, selected_el):
            # Skip 0/0, as that is a singularity
            if coords[0] == 0.0 and coords[1] == 0.0:
                continue

            for dia in diameters:
                combos.append((dia, *coords))

        np.random.seed(1)
        shuffled_combos = np.random.permutation(combos)

        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        for i in range(2):
            for size, az, el in shuffled_combos:
                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, 0, az, el, size))
                self.add_phase(p)

                p = vxprotocol.Phase(duration=4)
                p.set_visual(LocalTranslationGrating_RoundArea,
                             create_params(period, velocity, az, el, size))
                self.add_phase(p)

        p = vxprotocol.Phase(5)
        p.set_visual(SphereUniformBackground)
        self.add_phase(p)

        self.combos = combos


if __name__ == '__main__':
    os.chdir('../')

    import matplotlib.pyplot as plt

    def cart2sph1(cx, cy, cz):
        cxy = cx + cy * 1.j
        azi = np.angle(cxy)
        elv = np.angle(np.abs(cxy) + cz * 1.j)
        return azi, elv


    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r


    def sph2cart(theta, phi, r):
        rcos_theta = r * np.cos(phi)
        x = rcos_theta * np.cos(theta)
        y = rcos_theta * np.sin(theta)
        z = r * np.sin(phi)
        return np.array([x, y, z])


    def calculate_circular_mask(stimulus_center_az, stimulus_center_el, stimulus_diameter):
        # Convert to radians
        stimulus_center_az *= np.pi / 180
        stimulus_center_el *= np.pi / 180
        stimulus_diameter *= np.pi / 180

        # New mask
        mask = np.zeros((xsteps, ysteps))

        # Calculate center position of stimulus patch
        pos = sph2cart(stimulus_center_az, stimulus_center_el, 1)

        # Calculate angular distance to of each mask coordinate to center
        angles = np.arccos(np.dot(stimulus_mask_cart_positions.reshape((-1, 3)), pos)).reshape((xsteps, ysteps))

        # Calculate solid angle (optional, for inverse normalization by stimulation area)
        # patch_solid_angle = 2 * np.pi * (1 - np.cos(stimulus_diameter / 2))  # sr
        # mask[angles < row.diameter / 2] = 1 / patch_solid_angle

        # Set mask to 1 based on angular distance to center and angular diameter
        mask[angles < stimulus_diameter / 2] = 1

        return mask


    # Set resolution for equirectangular map
    xres = 1  # degrees
    yres = 1  # degrees

    # Create coordinates for stimulus mask
    mask_azimuths = np.arange(-180, 180, xres) / 180 * np.pi
    xsteps = mask_azimuths.shape[0]
    mask_elevations = np.arange(-90, 90, yres) / 180 * np.pi
    ysteps = mask_elevations.shape[0]
    stimulus_mask_cart_positions = np.array([[sph2cart(a, e, 1) for e in mask_elevations] for a in mask_azimuths])
    mask_az_mesh, mask_el_mesh = np.meshgrid(mask_azimuths, mask_elevations)

    # Instantiate protocol
    protocol = RepulsiveSphere140and40LeftEyeLargePatches20230810()

    # Calculate stimulus masks
    stimulus_masks = np.array([calculate_circular_mask(a, e, d) for d, a, e, _ in protocol.combos])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(stimulus_masks.sum(axis=0).T,
              origin='lower', aspect=1,
              extent=[-180.0, 180.0, -90.0, 90.0],
              cmap='viridis')
    ax.set_aspect(1)
    plt.show()
