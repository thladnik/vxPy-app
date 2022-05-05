from vxpy.utils import widgets


class FreeswimZebrafishTracking(widgets.AddonCameraWidget):

    def structure(self):
        self.add_image('multiple_fish_vertical_swim_frame', 0)
