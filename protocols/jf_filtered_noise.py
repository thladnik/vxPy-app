"""
vxpy_app ./protocols/planar_gratings.py
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
from vxpy.core.protocol import Phase, StaticProtocol
from vxpy.visuals import pause
from visuals.filtered_noise import FilteredNoise


class FilteredNoiseOverTime(StaticProtocol):
    def __init__(self, *args, **kwargs):
        StaticProtocol.__init__(self, *args, **kwargs)

        p = Phase(duration=300)
        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 300.0,
                      FilteredNoise.start_sigma: 5.0,
                      FilteredNoise.end_sigma: 10.0})
        self.add_phase(p)

        p = Phase(duration=300)
        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 180.0,
                      FilteredNoise.start_sigma: 10.0,
                      FilteredNoise.end_sigma: 10.0})
        self.add_phase(p)

        p = Phase(duration=300)
        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 300.0,
                      FilteredNoise.start_sigma: 10.0,
                      FilteredNoise.end_sigma: 15.0})
        self.add_phase(p)

        p = Phase(duration=300)
        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 180.0,
                      FilteredNoise.start_sigma: 15.0,
                      FilteredNoise.end_sigma: 15.0})
        self.add_phase(p)
        p = Phase(duration=300)

        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 300.0,
                      FilteredNoise.start_sigma: 15.0,
                      FilteredNoise.end_sigma: 20.0})
        self.add_phase(p)
        p = Phase(duration=300)

        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 180.0,
                      FilteredNoise.start_sigma: 20.0,
                      FilteredNoise.end_sigma: 20.0})
        self.add_phase(p)

        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 300.0,
                      FilteredNoise.start_sigma: 20.0,
                      FilteredNoise.end_sigma: 25.0})
        self.add_phase(p)

        p.set_visual(FilteredNoise,
                     {FilteredNoise.duration: 180.0,
                      FilteredNoise.start_sigma: 25.0,
                      FilteredNoise.end_sigma: 25.0})
        self.add_phase(p)