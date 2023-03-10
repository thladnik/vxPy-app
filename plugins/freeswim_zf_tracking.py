"""
vxPy-app ./plugins/freeswim_zf_tracking.py
Copyright (C) 2022 Tim Hladnik

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
from typing import Tuple, List

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QLabel

from vxpy import config
import vxpy.core.attribute as vxattribute
import vxpy.core.devices.camera as vxcamera
import vxpy.core.ui as vxui
import vxpy.core.ipc as vxipc
import vxpy.core.routine as vxroutine
from vxpy.definitions import *
from vxpy.utils import widgets


class RoiView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, parent=parent, **kwargs)

        self.setFixedHeight(120)

        # Set up plot image item
        self.image_plot = self.addPlot(0, 0, 1, 10)
        self.image_item = pg.ImageItem()
        self.image_plot.hideAxis('left')
        self.image_plot.hideAxis('bottom')
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)
        self.image_plot.addItem(self.image_item)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_image)
        self.timer.setInterval(50)
        self.timer.start()

    def _update_image(self):
        # Read last frame
        idx, time, frame = vxattribute.read_attribute('freeswim_tracked_particle_rects')

        if idx[0] is None:
            return

        # Set frame data on image plot
        self.image_item.setImage(np.vstack(frame[0]))


class Roi(pg.RectROI):
    def __init__(self):
        self.default_pen = pg.mkPen(color='darkgreen', width=4)
        pg.RectROI.__init__(self, FreeswimTrackerRoutine.calibration_rect_pos,
                            [100, 100], sideScalers=True,
                            pen=self.default_pen, hoverPen=self.default_pen,
                            handlePen=self.default_pen, handleHoverPen=self.default_pen,
                            maxBounds=QtCore.QRectF(0, 0, 1924, 1080))

        self.active_calibration = False

    def set_calibration_mode(self, active):
        # self.translatable = False

        # self.active_calibration = active
        self.translatable = active
        self.resizable = active  # TODO: resizable seems to have no effect as of pyqtgraph 0.13.1


class FrameView(pg.GraphicsLayoutWidget):
    def __init__(self, parent, **kwargs):
        pg.GraphicsLayoutWidget.__init__(self, parent=parent, **kwargs)

        self._calibrate = False
        self._points = []
        self._attribute = None

        # Set up plot image item
        self.image_plot = self.addPlot(0, 0, 1, 10)
        self.image_item = pg.ImageItem()
        self.image_plot.hideAxis('left')
        self.image_plot.hideAxis('bottom')
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)
        self.image_plot.addItem(self.image_item)

        # Add calibration ROI
        self.rect_roi = Roi()
        self.image_plot.vb.addItem(self.rect_roi)

        # Create ROIs to mark tracked particles
        self._tracking_rois: List[pg.RectROI] = []
        self._tracking_texts: List[pg.TextItem] = []
        self.default_pen = pg.mkPen('red', width=5)
        for _ in range(FreeswimTrackerRoutine.max_particle_num):
            # Add ROI
            rect = pg.RectROI((50, 50), FreeswimTrackerRoutine.rect_size,
                              pen=self.default_pen, handlePen=pg.mkPen(None), hoverPen=self.default_pen)
            self._tracking_rois.append(rect)
            self.image_plot.vb.addItem(self._tracking_rois[-1])

            # Add text
            text = pg.TextItem('')
            self._tracking_texts.append(text)
            self.image_plot.vb.addItem(self._tracking_texts[-1])

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.setInterval(50)
        self.timer.start()

    def set_attribute(self, frame_name):
        self._attribute = vxattribute.get_attribute(frame_name)

    def _update(self):
        if self._attribute is None:
            return

        # Read last frame
        idx, time, frame = self._attribute.read()

        if idx[0] is None:
            return

        # Set frame data on image plot
        self.image_item.setImage(frame[0])

        # Fetch ROI positions
        _, _, pixel_positions = vxattribute.read_attribute('freeswim_tracked_particle_pixel_position')
        _, _, mapped_positions = vxattribute.read_attribute('freeswim_tracked_particle_mapped_position')
        pixel_positions = pixel_positions[0]
        mapped_positions = mapped_positions[0]

        # Update ROI positions
        rect_size = FreeswimTrackerRoutine.rect_size
        for i, (rect, text) in enumerate(zip(self._tracking_rois, self._tracking_texts)):

            # Update ROI and text
            pos = pixel_positions[i]
            if pos[0] < 0:
                rect.setPen(pg.mkPen(None))
                text.setText('')
                continue

            # Move pos to center of ROI
            pos = (pos[0] - rect_size[0] / 2, pos[1] - rect_size[1] / 2)

            # Update ROI
            rect.setPos(pos)
            rect.setPen(self.default_pen)

            # Update text
            mapped_pos = mapped_positions[i]
            text.setPos(QtCore.QPoint(pos[0], pos[1] + rect_size[0]))
            text.setText(f'{mapped_pos[0]:.1f} / {mapped_pos[1]:.1f}', 'red')


class FreeswimTrackerWidget(vxui.CameraAddonWidget):
    display_name = 'FreeswimTracker'

    def __init__(self, *args, **kwargs):
        vxui.CameraAddonWidget.__init__(self, *args, **kwargs)
        self.central_widget.setLayout(QtWidgets.QHBoxLayout())

        # Add plots
        self.plots = QtWidgets.QWidget(self)
        self.plots.setLayout(QtWidgets.QVBoxLayout())
        self.central_widget.layout().addWidget(self.plots)
        # Frame
        self.frame_view = FrameView(self)
        self.frame_view.rect_roi.sigRegionChangeFinished.connect(self._update_calibration_roi_parameters)
        self.plots.layout().addWidget(self.frame_view)

        # ROIs
        self.roi_view = RoiView(self)
        self.plots.layout().addWidget(self.roi_view)
        # Add paraemters

        # Add parameter console
        self.console = QtWidgets.QWidget(self)
        self.console.setLayout(QtWidgets.QVBoxLayout())
        self.console.setMaximumWidth(300)
        self.central_widget.layout().addWidget(self.console)

        # Uniform width for labels (make it pretty)
        uniform_width = widgets.UniformWidth()

        # MOG MODEL
        self.console.layout().addWidget(QLabel('<b>Background MOG model</b>'))
        # MOG history length
        self.mog_history = widgets.IntSliderWidget(self.console, label='MOG history',
                                                   default=FreeswimTrackerRoutine.mog_history_len,
                                                   limits=(1, 5000))
        self.mog_history.connect_callback(self.set_mog_history_len)
        uniform_width.add_widget(self.mog_history.label)
        self.console.layout().addWidget(self.mog_history)

        # Reset button
        self.reset_btn = QtWidgets.QPushButton('Reset')
        self.reset_btn.clicked.connect(self.reset_mog_model)
        self.reset_mog = widgets.ParameterWidget('Reset MOG model', self.reset_btn)
        uniform_width.add_widget(self.reset_mog.label)
        self.console.layout().addWidget(self.reset_mog)

        # FILTERING
        # Display choice
        self.display_choice_cb = widgets.ComboBox(self)
        self.display_choice_cb.connect_callback(self.frame_view.set_attribute)
        self.display_choice_cb.add_items(['freeswim_tracked_zf_frame',
                                          'freeswim_tracked_zf_filtered',
                                          'freeswim_tracked_zf_foreground',
                                          'freeswim_tracked_zf_binary'])
        self.display_choice = widgets.ParameterWidget('Display frame', self.display_choice_cb)
        uniform_width.add_widget(self.display_choice.label)
        self.console.layout().addWidget(self.display_choice)

        self.console.layout().addWidget(QLabel('<b>Filter parameters</b>'))
        # Threshold
        self.binary_threshold = widgets.IntSliderWidget(self.console, label='Threshold [au]',
                                                        default=FreeswimTrackerRoutine.binary_thresh_val,
                                                        limits=(1, 254))
        self.binary_threshold.connect_callback(self.set_binary_threshold)
        uniform_width.add_widget(self.binary_threshold.label)
        self.console.layout().addWidget(self.binary_threshold)

        # Filter size
        self.filter_size = widgets.IntSliderWidget(self.console, label='Filter size [au]',
                                                   default=FreeswimTrackerRoutine.filter_size,
                                                   limits=(1, 255))
        self.filter_size.connect_callback(self.set_filter_size)
        uniform_width.add_widget(self.filter_size.label)
        self.console.layout().addWidget(self.filter_size)

        # Min area
        self.min_area = widgets.IntSliderWidget(self.console, label='Min. area [au]',
                                                default=FreeswimTrackerRoutine.min_area,
                                                limits=(1, 255))
        self.min_area.connect_callback(self.set_min_area)
        uniform_width.add_widget(self.min_area.label)
        self.console.layout().addWidget(self.min_area)

        # SUBFRAME CALIBRATION
        self.console.layout().addWidget(QLabel('<b>Subframe calibration</b>'))
        # Calibration mode
        self.calibration_cb = widgets.ComboBox(self)
        self.calibration_cb.add_items(['Open', 'Locked'])
        self.calibration_cb.connect_callback(self.set_calibration_mode)
        self.calibration = widgets.ParameterWidget('Calibration', self.calibration_cb)
        uniform_width.add_widget(self.calibration.label)
        self.console.layout().addWidget(self.calibration)
        # X dim
        self.x_dimension_length = widgets.IntSliderWidget(self.console, label='X dimension [mm]',
                                                          default=FreeswimTrackerRoutine.dimension_size[0],
                                                          limits=(1, 1000))
        self.x_dimension_length.connect_callback(self.set_x_dimension_size)
        uniform_width.add_widget(self.x_dimension_length.label)
        self.console.layout().addWidget(self.x_dimension_length)
        # Y dim
        self.y_dimension_length = widgets.IntSliderWidget(self.console, label='Y dimension [mm]',
                                                          default=FreeswimTrackerRoutine.dimension_size[1],
                                                          limits=(1, 1000))
        self.y_dimension_length.connect_callback(self.set_y_dimension_size)
        uniform_width.add_widget(self.y_dimension_length.label)
        self.console.layout().addWidget(self.y_dimension_length)

        # Spacer
        spacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.console.layout().addItem(spacer)

    def set_calibration_mode(self, mode):
        mode_is_open = mode == 'Open'
        self.frame_view.rect_roi.set_calibration_mode(mode_is_open)
        self.x_dimension_length.setEnabled(mode_is_open)
        self.y_dimension_length.setEnabled(mode_is_open)

    def reset_mog_model(self):
        self.call_routine(FreeswimTrackerRoutine.reset_mog_model)

    def set_mog_history_len(self):
        self.call_routine(FreeswimTrackerRoutine.set_mog_history_len, self.mog_history.get_value())

    def set_filter_size(self):
        self.call_routine(FreeswimTrackerRoutine.set_filter_size, self.filter_size.get_value())

    def set_min_area(self):
        self.call_routine(FreeswimTrackerRoutine.set_min_area, self.min_area.get_value())

    def set_binary_threshold(self):
        self.call_routine(FreeswimTrackerRoutine.set_binary_threshold, self.binary_threshold.get_value())

    def set_x_dimension_size(self):
        self.call_routine(FreeswimTrackerRoutine.set_x_dimension_size, self.x_dimension_length.get_value())

    def set_y_dimension_size(self):
        self.call_routine(FreeswimTrackerRoutine.set_y_dimension_size, self.y_dimension_length.get_value())

    def _update_calibration_roi_parameters(self):
        pos = self.frame_view.rect_roi.pos()
        self.call_routine(FreeswimTrackerRoutine.set_calibration_rect_pos, (pos.x(), pos.y()))
        size = self.frame_view.rect_roi.size()
        self.call_routine(FreeswimTrackerRoutine.set_calibration_rect_size, (size.x(), size.y()))


class FreeswimTrackerRoutine(vxroutine.CameraRoutine):
    rect_size = (60, 60)
    camera_device_id = 'multiple_fish_vertical_swim'

    # Set processing parameters
    dimension_size = np.array([100., 80.])  # mm
    binary_thresh_val = 25
    min_area = 10
    filter_size = 31
    mog_history_len = 3000
    calibration_rect_pos = np.array([0, 0])
    calibration_rect_size = np.array([1, 1])
    max_particle_num = 10

    attr_name_frame = ''

    def __init__(self, *args, **kwargs):
        vxroutine.CameraRoutine.__init__(self, *args, **kwargs)

        # Create mixture of gaussian BG subtractor
        self._mog: cv2.MOG2BackgroundSubtractor = None

        self.reset_mog_model()

    @classmethod
    def require(cls):

        # Get camera properties
        camera = vxcamera.get_camera_by_id(cls.camera_device_id)

        width, height = camera.width, camera.height

        vxattribute.VideoStreamAttribute(camera.frame_rate, 'freeswim_tracked_zf_frame',
                                         (width, height),
                                         vxattribute.ArrayType.uint8)
        vxattribute.ArrayAttribute('freeswim_tracked_zf_foreground',
                                   (width, height),
                                   vxattribute.ArrayType.uint8)
        vxattribute.ArrayAttribute('freeswim_tracked_zf_filtered',
                                   (width, height),
                                   vxattribute.ArrayType.uint8)
        vxattribute.ArrayAttribute('freeswim_tracked_zf_binary',
                                   (width, height),
                                   vxattribute.ArrayType.uint8)
        vxattribute.ArrayAttribute('freeswim_tracked_particle_rects',
                                   (cls.max_particle_num, *cls.rect_size),
                                   dtype=vxattribute.ArrayType.uint8,
                                   chunked=True)

        vxattribute.ArrayAttribute('freeswim_tracked_particle_count_total',
                                   (1,),
                                   dtype=vxattribute.ArrayType.uint64)
        vxattribute.ArrayAttribute('freeswim_tracked_particle_count_filtered',
                                   (1,),
                                   dtype=vxattribute.ArrayType.uint64)
        vxattribute.ArrayAttribute('freeswim_tracked_particle_pixel_position',
                                   (cls.max_particle_num, 2,),
                                   dtype=vxattribute.ArrayType.float64)
        vxattribute.ArrayAttribute('freeswim_tracked_particle_mapped_position',
                                   (cls.max_particle_num, 2,),
                                   dtype=vxattribute.ArrayType.float64)

    def initialize(self):

        vxui.register_with_plotter('freeswim_tracked_particle_count_total', axis='particle_count')
        vxui.register_with_plotter('freeswim_tracked_particle_count_filtered', axis='particle_count')

        vxattribute.write_to_file(self, 'freeswim_tracked_zf_frame')
        vxattribute.write_to_file(self, 'freeswim_tracked_particle_rects')
        vxattribute.write_to_file(self, 'freeswim_tracked_particle_count_total')
        vxattribute.write_to_file(self, 'freeswim_tracked_particle_count_filtered')
        vxattribute.write_to_file(self, 'freeswim_tracked_particle_pixel_position')
        vxattribute.write_to_file(self, 'freeswim_tracked_particle_mapped_position')

    @vxroutine.CameraRoutine.callback
    def reset_mog_model(self):
        # self._mog = cv2.createBackgroundSubtractorMOG2(int(self.mog_history_len), detectShadows=False)
        self._mog = cv2.createBackgroundSubtractorKNN(int(self.mog_history_len), detectShadows=False)

    @vxroutine.CameraRoutine.callback
    def set_mog_history_len(self, value):
        self.mog_history_len = np.array(value)

    @vxroutine.CameraRoutine.callback
    def set_calibration_rect_pos(self, value):
        self.calibration_rect_pos = np.array(value)

    @vxroutine.CameraRoutine.callback
    def set_calibration_rect_size(self, value):
        self.calibration_rect_size = np.array(value)

    @vxroutine.CameraRoutine.callback
    def set_x_dimension_size(self, value):
        self.dimension_size[0] = value

    @vxroutine.CameraRoutine.callback
    def set_y_dimension_size(self, value):
        self.dimension_size[1] = value

    @vxroutine.CameraRoutine.callback
    def set_min_area(self, value):
        self.min_area = value

    @vxroutine.CameraRoutine.callback
    def set_binary_threshold(self, value):
        self.binary_thresh_val = value

    @vxroutine.CameraRoutine.callback
    def set_filter_size(self, value):
        self.filter_size = value // 2 * 2 + 1  # always make sure that filter size is odd integer

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
            frame = frame[:, :, 0].T
        frame = frame.T
        vxattribute.write_attribute('freeswim_tracked_zf_frame', frame)

        # Smooth frame
        filtered_frame = cv2.GaussianBlur(frame, (self.filter_size,) * 2, cv2.BORDER_DEFAULT)
        vxattribute.write_attribute('freeswim_tracked_zf_filtered', filtered_frame)

        # Calculate background distribution and foreground mask
        foreground_mask = self._mog.apply(filtered_frame)
        vxattribute.write_attribute('freeswim_tracked_zf_foreground', foreground_mask)

        # Apply threshold
        _, thresh_frame = cv2.threshold(foreground_mask, self.binary_thresh_val, 255, cv2.THRESH_BINARY)
        vxattribute.write_attribute('freeswim_tracked_zf_binary', thresh_frame)

        # Detect and filter contours
        contours, hierarchy = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        xdiff, ydiff = self.rect_size[0] // 2, self.rect_size[1] // 2
        particle_rectangles = []
        particle_areas = []
        particle_pixel_positions = []
        particle_mapped_positions = []

        # Write number of all detected particles to attribute
        vxattribute.write_attribute('freeswim_tracked_particle_count_total', len(contours))

        # Go through all contours now and filter them
        print('---')
        if len(contours) > 0:
            i = 0
            for cnt in contours:

                # Filter by particle area
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                # Calculate moments
                M = cv2.moments(cnt)

                # Calculate centroid
                centroid = np.array([int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
                c_rev = np.array([centroid[1], centroid[0]])
                mapped_position = self._apply_dimensions(c_rev)

                # Crop rectangular ROI
                rect = frame[centroid[1] - ydiff:centroid[1] + ydiff, centroid[0] - xdiff:centroid[0] + xdiff]

                # Skip if one dimension is empty
                # if np.any(rect.shape == 0):
                #     continue

                # Fix rects overlapping source image boundary
                if rect.shape != self.rect_size:
                    print(rect.sum())
                    right_rect = np.zeros(self.rect_size)
                    right_rect[:rect.shape[0], :rect.shape[1]] = rect[:,:]
                    rect = right_rect
                    print(rect.sum())

                particle_rectangles.append(rect)
                particle_areas.append((area, i))
                particle_pixel_positions.append(c_rev)
                particle_mapped_positions.append(mapped_position)
                i += 1

        vxattribute.write_attribute('freeswim_tracked_particle_count_filtered', len(particle_areas))

        # If no suitable particles were detected
        if len(particle_areas) == 0:
            return

        # Sort particles by size
        particle_areas = sorted(particle_areas)[::-1]
        if len(particle_areas) > 10:
            particle_areas = particle_areas[:10]

        # Write particle ROIs to attribute
        particle_rois_attr = vxattribute.get_attribute('freeswim_tracked_particle_rects')
        particle_mapped_position_attr = vxattribute.get_attribute('freeswim_tracked_particle_mapped_position')
        particle_pixel_position_attr = vxattribute.get_attribute('freeswim_tracked_particle_pixel_position')
        new_rects = np.zeros(particle_rois_attr.shape)
        new_mapped_positions = -np.ones(particle_mapped_position_attr.shape)
        new_pixel_positions = -np.ones(particle_mapped_position_attr.shape)
        for k, (_, i) in enumerate(particle_areas):
            new_rects[k] = particle_rectangles[i]
            new_mapped_positions[k] = particle_mapped_positions[i]
            new_pixel_positions[k] = particle_pixel_positions[i]

        # Write particle boxes
        particle_rois_attr.write(new_rects)

        # Write centroid positions
        particle_pixel_position_attr.write(new_pixel_positions)
        particle_mapped_position_attr.write(new_mapped_positions)
