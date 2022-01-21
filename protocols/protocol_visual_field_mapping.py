

from vxpy.core.protocol import StaticPhasicProtocol

from visuals.sphere_visual_field_mapping import BinaryNoiseVisualFieldMapping
from visuals.sph_shader import Blank


class TestProtocol01(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for inv in [False, True]:
            for bias in [0.1, 0.2]:
                self.add_visual_phase(Blank, 15, {})
                self.add_visual_phase(BinaryNoiseVisualFieldMapping,
                                      300,
                                      {BinaryNoiseVisualFieldMapping.p_bias: bias,
                                BinaryNoiseVisualFieldMapping.p_interval: 1000,
                                BinaryNoiseVisualFieldMapping.p_inverted: inv})