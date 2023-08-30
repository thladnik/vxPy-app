import numpy as np


class UnitSphere:
    azim_levels = 100
    elev_levels = 50
    az = np.linspace(0, 2 * np.pi, azim_levels, endpoint=True)
    el = np.linspace(-np.pi / 2, +np.pi / 2, elev_levels, endpoint=True)
    azims, elevs = np.meshgrid(az, el)

    @staticmethod
    def vertices():
        rcos_theta = np.cos(UnitSphere.elevs.flatten())
        x = rcos_theta * np.cos(UnitSphere.azims.flatten())
        y = rcos_theta * np.sin(UnitSphere.azims.flatten())
        z = np.sin(UnitSphere.elevs.flatten())

        return np.array([x, y, z], dtype=np.float32).T

    @staticmethod
    def indices():
        az_lvls = UnitSphere.azim_levels
        el_lvls = UnitSphere.elev_levels
        idcs = list()
        for i in np.arange(el_lvls-1):
            for j in np.arange(az_lvls-1):
                idcs.append([i * az_lvls + j, i * az_lvls + j + 1, (i + 1) * az_lvls + j + 1])
                idcs.append([i * az_lvls + j, (i + 1) * az_lvls + j, (i + 1) * az_lvls + j + 1])
        return np.ascontiguousarray(np.array(idcs).flatten(), dtype=np.uint32)


class Cube:
    vertices = np.array([[+1, -1, -1],
                         [-1, -1, -1],
                         [+1, -1, +1],
                         [-1, -1, +1],
                         [+1, +1, +1],
                         [-1, +1, +1],
                         [+1, +1, -1],
                         [-1, +1, -1]], dtype=np.float32)

    indices = np.array([0, 1, 2,
                        1, 2, 3,
                        2, 3, 4,
                        3, 4, 5,
                        4, 5, 6,
                        5, 6, 7,
                        6, 7, 0,
                        7, 0, 1,
                        1, 3, 5,
                        1, 5, 7,
                        0, 2, 4,
                        0, 4, 6], dtype=np.uint32)

    @staticmethod
    def vertices():
        return Cube.vertices

    @staticmethod
    def indices():
        return Cube.indices