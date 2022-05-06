from typing import Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtGui, QtCore

from vxpy import config
import vxpy.core.attribute as vxattribute
import vxpy.core.gui as vxgui
import vxpy.core.routine as vxroutine
from vxpy.utils import widgets


# class FreeswimTrackerWidget(widgets.AddonCameraWidget):
#     display_name = 'FreeswimTracker'
#
#     def structure(self):
#         self.add_image('freeswim_tracked_zf_frame', 0)


class GraphicsWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, parent=parent, **kwargs)

        self._calibrate = False
        self._points = []
        self._attribute = vxattribute.get_attribute('freeswim_tracked_zf_frame')

        # Set context menu
        self.context_menu = QtWidgets.QMenu()

        # Set new line
        self.menu_new = QtGui.QAction('Set calibration rectangle')
        self.menu_new.triggered.connect(self.start_calibration)
        self.context_menu.addAction(self.menu_new)

        # Set up plot image item
        self.image_plot = self.addPlot(0, 0, 1, 10)
        self.image_item = pg.ImageItem()
        self.image_plot.hideAxis('left')
        self.image_plot.hideAxis('bottom')
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)
        self.image_plot.addItem(self.image_item)

        # Add calibration ROI
        self.rect_roi = pg.RectROI((800, 800), (1000, 1000), pen=pg.mkPen(color='blue', width=2))
        self.image_plot.vb.addItem(self.rect_roi)

        # Bind mouse click event
        self.image_plot.scene().sigMouseClicked.connect(self.mouse_clicked)
        # Bind context menu call function
        self.image_plot.vb.raiseContextMenu = self.raise_context_menu

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_image)
        self.timer.setInterval(50)
        self.timer.start()

    def raise_context_menu(self, ev):
        self.context_menu.popup(QtCore.QPoint(ev.screenPos().x(), ev.screenPos().y()))

    def start_calibration(self):
        self._calibrate = True
        self._points = []

    def mouse_clicked(self, ev):
        if not self._calibrate:
            return

        pos = self.image_plot.vb.mapSceneToView(ev.scenePos())

        # First click
        if len(self._points) == 0:
            self._points.append(pos)
            return

        self._points.append(pos)

        self._set_rect_roi()

        self._calibrate = False

    def _set_rect_roi(self):
        size = self._points[1]-self._points[0]
        self.rect_roi.setPos(self._points[0])
        self.rect_roi.setSize(size)

    def _update_image(self):
        if self._attribute is None:
            return

        # Read last frame
        idx, time, frame = self._attribute.read()

        if idx[0] is None:
            return

        # Set frame data on image plot
        self.image_item.setImage(frame[0])


class FreeswimTrackerWidget(vxgui.CameraAddonWidget):
    display_name = 'FreeswimTracker'

    def __init__(self, *args, **kwargs):
        vxgui.AddonWidget.__init__(self, *args, **kwargs)
        self.setLayout(QtWidgets.QHBoxLayout())

        self.image_widget = GraphicsWidget(self)
        self.image_widget.rect_roi.sigRegionChangeFinished.connect(self._update_roi_parameters)
        self.layout().addWidget(self.image_widget)

    def _update_roi_parameters(self):
        pos = self.image_widget.rect_roi.pos()
        self.call_routine(FreeswimTrackerRoutine.set_calibration_rect_pos, (pos.x(), pos.y()))
        size = self.image_widget.rect_roi.size()
        self.call_routine(FreeswimTrackerRoutine.set_calibration_rect_size, (size.x(), size.y()))


class FreeswimTrackerRoutine(vxroutine.CameraRoutine):

    rect_size = (60, 60)
    camera_device_id = 'multiple_fish_vertical_swim'

    def setup(self):

        # Get camera properties
        camera_config = config.CONF_CAMERA_DEVICES.get(self.camera_device_id)
        self.res_x = camera_config['width']
        self.res_y = camera_config['height']

        # Set processing parameters
        self.dimension_size = np.array([100., 80.])  # mm
        self.thresh_val = 25
        self.min_area = 10
        self.filter_size = (31, 31)
        self.calibration_rect_pos = np.array([0, 0])
        self.calibration_rect_size = np.array([self.res_x, self.res_y])
        self.mog = cv2.createBackgroundSubtractorMOG2(400, detectShadows=False)

        self.freeswim_tracked_zf_frame = vxattribute.ArrayAttribute('freeswim_tracked_zf_frame',
                                                                    (self.res_x, self.res_y, 3),
                                                                    vxattribute.ArrayType.uint8)
        self.particle_rois = vxattribute.ArrayAttribute('particle_rois',
                                                        (10, *self.rect_size),
                                                        dtype=vxattribute.ArrayType.uint8,
                                                        chunked=True)

        self.exposed.append(FreeswimTrackerRoutine.set_calibration_rect_pos)
        self.exposed.append(FreeswimTrackerRoutine.set_calibration_rect_size)
        self.exposed.append(FreeswimTrackerRoutine.set_x_dimension_size)
        self.exposed.append(FreeswimTrackerRoutine.set_y_dimension_size)

    def set_calibration_rect_pos(self, pos):
        self.calibration_rect_pos = np.array(pos)

    def set_calibration_rect_size(self, size):
        self.calibration_rect_size = np.array(size)

    def set_x_dimension_size(self, x_size):
        self.dimension_size[0] = x_size

    def set_y_dimension_size(self, y_size):
        self.dimension_size[1] = y_size

    def _apply_dimensions(self, point: np.ndarray) -> np.ndarray:
        p = point.copy()
        p = p - self.calibration_rect_pos
        p = p / self.calibration_rect_size
        p = p * self.dimension_size
        p[1] = self.dimension_size[1] - p[1]

        return p

    def main(self, **frames):
        frame = frames.get(self.camera_device_id)

        if frame is None:
            return

        if frame.ndim > 2:
            frame = frame[:,:,0].T
        frame = frame.T

        display_frame = np.repeat(frame[:,:,np.newaxis], 3, axis=-1)

        foreground_mask = self.mog.apply(frame)
        blurred = cv2.GaussianBlur(foreground_mask, self.filter_size, cv2.BORDER_DEFAULT)

        _, thresh = cv2.threshold(blurred, self.thresh_val, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        # print('---')
        # print('Pos', self.calibration_rect_pos)
        # print('Size', self.calibration_rect_size)
        if len(contours) > 0:
            for cnt in contours:

                M = cv2.moments(cnt)

                if cv2.contourArea(cnt) < self.min_area:
                    continue

                # Calculate centroid
                c = np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
                c_rev = np.array([c[1], c[0]])
                # print(c_rev, '->', self._apply_dimensions(c_rev))

                # Mark ROIs on display frame
                xdiff, ydiff = self.rect_size[0] // 2, self.rect_size[1] // 2
                cv2.rectangle(display_frame, [c[0] - xdiff, c[1] - ydiff], [c[0] + xdiff, c[1] + ydiff], (255, 0, 0), 2)
                # rect = frame[c[1]-ydiff:c[1]+ydiff, c[0]-xdiff:c[0]+xdiff]
                # if rect.shape[0] > 0:
                #     rectangles.append(rect)
        # if len(rectangles) > 0:
        #     cv2.imshow('Rectangles', cv2.hconcat(rectangles))
        # cv2.imshow('Frame', display_frame)

        self.freeswim_tracked_zf_frame.write(display_frame)
