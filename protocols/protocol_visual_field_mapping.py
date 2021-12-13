

from vxpy.core.protocol import StaticPhasicProtocol

from visuals.sphere_visual_field_mapping import BinaryNoiseVisualFieldMapping


class TestProtocol01(StaticPhasicProtocol):

    def __init__(self, *args, **kwargs):
        StaticPhasicProtocol.__init__(self, *args, **kwargs)

        for inv in [False, True]:
            for bias in [0.1, 0.2]:
                self.add_phase(BinaryNoiseVisualFieldMapping,
                               10,
                               {BinaryNoiseVisualFieldMapping.p_bias: bias,
                                BinaryNoiseVisualFieldMapping.p_interval: 500,
                                BinaryNoiseVisualFieldMapping.p_inverted: inv})