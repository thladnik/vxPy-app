import vxpy.core.visual as vxvisual
from vispy import gloo
from vxpy.utils import sphere
import numpy as np


class CMN_foreground(vxvisual.SphericalVisual):
    def __init__(self, *args, **kwargs):
        vxvisual.SphericalVisual.__init__(self, *args, **kwargs)
        self.sphere = sphere.IcosahedronSphere

    def initialize(self, **params):
