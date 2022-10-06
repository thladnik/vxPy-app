"""
vxpy
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
from vispy import gloo
import numpy as np

from vxpy.core import visual
from vxpy.utils import plane


class KeepLast(visual.PlainVisual):

    def __init__(self, *args, **kwargs):
        visual.PlainVisual.__init__(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        pass

    def render(self, frame_time):
        pass


class ClearBlack(visual.PlainVisual):

    def __init__(self, *args, **kwargs):
        visual.PlainVisual.__init__(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        gloo.clear('black')

    def render(self, frame_time):
        pass


class ClearWhite(visual.PlainVisual):

    def __init__(self, *args, **kwargs):
        visual.PlainVisual.__init__(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        gloo.clear('white')

    def render(self, frame_time):
        pass


class Clear(visual.PlainVisual):

    class Saturation(visual.FloatParameter):
        def __init__(self):
            visual.FloatParameter.__init__(self, 'saturation', limits=(0., 1.), step_size=0.01)

            self.value_map = self._do_clear

        @staticmethod
        def _do_clear(data):
            gloo.clear((data,) * 3)

    saturation = Saturation()

    def __init__(self, *args, **kwargs):
        visual.PlainVisual.__init__(self, *args, **kwargs)

    def initialize(self, *args, **kwargs):
        pass

    def render(self, frame_time):
        pass

