from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .image_utils import numpy_to_qpixmap

CurvePoints = List[Tuple[float, float]]


class CurveEditor(QtWidgets.QWidget):
    """Lightweight curve editor widget for intensity mapping."""

    curveChanged = QtCore.pyqtSignal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(240, 180)
        self.setMouseTracking(True)
        self._points: CurvePoints = [(0.0, 0.0), (1.0, 1.0)]
        self._active_index: Optional[int] = None
        self._grid_pen = QtGui.QPen(QtGui.QColor("#3c3f46"))
        self._grid_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        self._curve_pen = QtGui.QPen(QtGui.QColor("#1a73e8"))
        self._curve_pen.setWidth(2)
        self._handle_brush = QtGui.QBrush(QtGui.QColor("#1a73e8"))
        self._handle_pen = QtGui.QPen(QtGui.QColor("#0f4aa3"))

    # Public API
    def set_points(self, points: CurvePoints) -> None:
        if not points:
            points = [(0.0, 0.0), (1.0, 1.0)]
        clamped = []
        for x, y in points:
            clamped.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))
        clamped.sort(key=lambda p: p[0])
        if len(clamped) < 2:
            value = clamped[0][1]
            clamped = [(0.0, value), (1.0, value)]
        if clamped[0][0] != 0.0:
            clamped[0] = (0.0, clamped[0][1])
        if clamped[-1][0] != 1.0:
            clamped[-1] = (1.0, clamped[-1][1])
        self._points = clamped
        self.update()

    def points(self) -> CurvePoints:
        return list(self._points)

    @staticmethod
    def build_lut(points: CurvePoints, resolution: int = 256) -> np.ndarray:
        if not points:
            points = [(0.0, 0.0), (1.0, 1.0)]
        pts = np.array(points, dtype=np.float32)
        xs = np.clip(pts[:, 0], 0.0, 1.0)
        ys = np.clip(pts[:, 1], 0.0, 1.0)
        interp_x = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
        interp_y = np.interp(interp_x, xs, ys)
        lut = np.clip(interp_y * 255.0, 0, 255).astype(np.uint8)
        return lut

    # Painting
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QtGui.QColor("#101218"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#242832")))
        painter.drawRect(rect)
        self._draw_grid(painter, rect)
        self._draw_curve(painter, rect)
        self._draw_handles(painter, rect)
        painter.end()

    def _draw_grid(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        painter.setPen(self._grid_pen)
        step_x = rect.width() / 4
        step_y = rect.height() / 4
        for i in range(1, 4):
            x = int(round(rect.left() + i * step_x))
            y = int(round(rect.top() + i * step_y))
            painter.drawLine(x, rect.top(), x, rect.bottom())
            painter.drawLine(rect.left(), y, rect.right(), y)

    def _draw_curve(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        if len(self._points) < 2:
            return
        painter.setPen(self._curve_pen)
        path = QtGui.QPainterPath()
        first = self._map_to_rect(self._points[0], rect)
        path.moveTo(first)
        for point in self._points[1:]:
            path.lineTo(self._map_to_rect(point, rect))
        painter.drawPath(path)

    def _draw_handles(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        painter.setPen(self._handle_pen)
        radius = 5
        for idx, point in enumerate(self._points):
            pos = self._map_to_rect(point, rect)
            if idx in (0, len(self._points) - 1):
                painter.setBrush(QtGui.QColor("#28313f"))
                painter.drawEllipse(QtCore.QPointF(pos.x(), pos.y()), radius + 2, radius + 2)
            painter.setBrush(self._handle_brush)
            painter.drawEllipse(QtCore.QPointF(pos.x(), pos.y()), radius, radius)

    # Interaction 
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            idx = self._find_point(event.position())
            if idx not in (0, len(self._points) - 1) and idx is not None:
                self._points.pop(idx)
                self.curveChanged.emit(self.points())
                self.update()
            return
        idx = self._find_point(event.position())
        if idx is None:
            self._add_point(event.position())
        else:
            self._active_index = idx

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._active_index is None:
            return
        self._move_point(event.position())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._active_index = None

    # Helpers 
    def _map_to_rect(self, point: Tuple[float, float], rect: QtCore.QRect) -> QtCore.QPointF:
        x = rect.left() + point[0] * rect.width()
        y = rect.bottom() - point[1] * rect.height()
        return QtCore.QPointF(x, y)

    def _from_rect(self, pos: QtCore.QPointF, rect: QtCore.QRect) -> Tuple[float, float]:
        x = (pos.x() - rect.left()) / rect.width()
        y = (rect.bottom() - pos.y()) / rect.height()
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def _find_point(self, pos: QtCore.QPointF) -> Optional[int]:
        rect = self.rect().adjusted(8, 8, -8, -8)
        radius = 10
        for idx, point in enumerate(self._points):
            mapped = self._map_to_rect(point, rect)
            dx = mapped.x() - pos.x()
            dy = mapped.y() - pos.y()
            if math.hypot(dx, dy) <= radius:
                return idx
        return None

    def _add_point(self, pos: QtCore.QPointF) -> None:
        if len(self._points) >= 8:
            return
        rect = self.rect().adjusted(8, 8, -8, -8)
        new_point = self._from_rect(pos, rect)
        self._points.append(new_point)
        self._points.sort(key=lambda p: p[0])
        self.curveChanged.emit(self.points())
        self.update()

    def _move_point(self, pos: QtCore.QPointF) -> None:
        rect = self.rect().adjusted(8, 8, -8, -8)
        new_point = self._from_rect(pos, rect)
        idx = self._active_index
        if idx is None:
            return
        if idx == 0:
            self._points[0] = (0.0, new_point[1])
        elif idx == len(self._points) - 1:
            self._points[-1] = (1.0, new_point[1])
        else:
            prev_x = self._points[idx - 1][0] + 0.01
            next_x = self._points[idx + 1][0] - 0.01
            clamped_x = float(np.clip(new_point[0], prev_x, next_x))
            self._points[idx] = (clamped_x, new_point[1])
        self.curveChanged.emit(self.points())
        self.update()


class IntensityPaletteDialog(QtWidgets.QDialog):
    """Dedicated popup for window/level, gamma, curve and colormap controls."""

    def __init__(self, main_window, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent or main_window)
        self.setWindowTitle("Intensity Palette")
        self.setModal(True)
        self.resize(760, 420)
        self.main_window = main_window
        self._preview_timer: Optional[QtCore.QTimer] = None
        self.result_state: Optional[dict] = None
        self._build_ui()
        self._apply_initial_state()
        self._update_preview()

    # UI construction 
    def _build_ui(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        controls = QtWidgets.QVBoxLayout()
        self.enable_checkbox = QtWidgets.QCheckBox("Enable palette adjustments")
        self.enable_checkbox.stateChanged.connect(lambda _: self._schedule_preview())
        controls.addWidget(self.enable_checkbox)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.center_slider, self.center_value = self._make_slider(0, 255, 128)
        self.center_slider.valueChanged.connect(lambda v: self._on_value_changed(self.center_value, v))
        form.addRow("Window center", self._wrap_slider(self.center_slider, self.center_value))

        self.width_slider, self.width_value = self._make_slider(1, 512, 256)
        self.width_slider.valueChanged.connect(lambda v: self._on_value_changed(self.width_value, v))
        form.addRow("Window width", self._wrap_slider(self.width_slider, self.width_value))

        self.gamma_slider, self.gamma_value = self._make_slider(10, 300, 100)
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        form.addRow("Gamma", self._wrap_slider(self.gamma_slider, self.gamma_value))

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Gray", "Hot", "Jet"])
        self.cmap_combo.currentTextChanged.connect(lambda _: self._schedule_preview())
        form.addRow("Colormap", self.cmap_combo)

        self.apply_checkbox = QtWidgets.QCheckBox("Apply transform to model input")
        controls.addLayout(form)
        controls.addWidget(self.apply_checkbox)

        curve_label = QtWidgets.QLabel("Curves")
        curve_label.setObjectName("caption")
        controls.addWidget(curve_label)
        self.curve_editor = CurveEditor()
        self.curve_editor.curveChanged.connect(lambda _: self._schedule_preview())
        controls.addWidget(self.curve_editor)

        curve_hint = QtWidgets.QLabel("Left-click to add/move points, right-click to delete.")
        curve_hint.setObjectName("hint")
        controls.addWidget(curve_hint)
        controls.addStretch()

        button_row = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self._reset_values)
        button_row.addWidget(self.reset_btn)
        self.update_preview_btn = QtWidgets.QPushButton("Update Preview")
        self.update_preview_btn.clicked.connect(self._update_preview)
        button_row.addWidget(self.update_preview_btn)
        controls.addLayout(button_row)

        layout.addLayout(controls, 1)

        preview_col = QtWidgets.QVBoxLayout()
        self.preview_label = QtWidgets.QLabel()
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #28313f; background: #0b0c10;")
        preview_col.addWidget(self.preview_label)
        self.preview_status = QtWidgets.QLabel("Enable palette to preview changes.")
        self.preview_status.setObjectName("hint")
        preview_col.addWidget(self.preview_status)
        preview_col.addStretch()

        dialog_buttons = QtWidgets.QDialogButtonBox()
        dialog_buttons.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        dialog_buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Apply")
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        preview_col.addWidget(dialog_buttons)

        layout.addLayout(preview_col, 1)

    # Helpers
    def _make_slider(self, minimum: int, maximum: int, value: int):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.valueChanged.connect(lambda _: self._schedule_preview())
        label = QtWidgets.QLabel(str(value))
        label.setFixedWidth(45)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return slider, label

    @staticmethod
    def _wrap_slider(slider: QtWidgets.QSlider, label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        return container

    def _on_value_changed(self, label: QtWidgets.QLabel, value: int) -> None:
        label.setText(str(int(value)))

    def _on_gamma_changed(self, value: int) -> None:
        gamma = float(value) / 100.0
        self.gamma_value.setText(f"{gamma:.2f}")
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        if self._preview_timer is None:
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(90)

    def _reset_values(self) -> None:
        self.center_slider.setValue(128)
        self.width_slider.setValue(256)
        self.gamma_slider.setValue(100)
        self.cmap_combo.setCurrentText("Gray")
        self.apply_checkbox.setChecked(False)
        self.enable_checkbox.setChecked(False)
        self.curve_editor.set_points([(0.0, 0.0), (1.0, 1.0)])
        self._update_preview()

    def _apply_initial_state(self) -> None:
        state = {
            "center": getattr(self.main_window, "wl_center", 128),
            "width": getattr(self.main_window, "wl_width", 256),
            "gamma": getattr(self.main_window, "gamma", 1.0),
            "colormap": getattr(self.main_window, "colormap", "Gray"),
            "apply_to_model": getattr(self.main_window, "apply_wl_to_model", False),
            "enabled": getattr(self.main_window, "intensity_enabled", False),
            "curve_points": getattr(self.main_window, "curve_points", [(0.0, 0.0), (1.0, 1.0)]),
        }
        self.center_slider.setValue(int(state["center"]))
        self.width_slider.setValue(int(state["width"]))
        self.gamma_slider.setValue(int(float(state["gamma"]) * 100))
        self.cmap_combo.setCurrentText(state["colormap"])
        self.apply_checkbox.setChecked(bool(state["apply_to_model"]))
        self.enable_checkbox.setChecked(bool(state["enabled"]))
        self.curve_editor.set_points(state["curve_points"])
        self._on_value_changed(self.center_value, self.center_slider.value())
        self._on_value_changed(self.width_value, self.width_slider.value())
        self._on_gamma_changed(self.gamma_slider.value())

    def _collect_state(self) -> dict:
        return {
            "center": int(self.center_slider.value()),
            "width": int(self.width_slider.value()),
            "gamma": float(self.gamma_slider.value()) / 100.0,
            "colormap": self.cmap_combo.currentText(),
            "apply_to_model": bool(self.apply_checkbox.isChecked()),
            "enabled": bool(self.enable_checkbox.isChecked()),
            "curve_points": self.curve_editor.points(),
        }

    def _update_preview(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()
        state = self._collect_state()
        preview_np = None
        if hasattr(self.main_window, "render_intensity_preview"):
            preview_np = self.main_window.render_intensity_preview(state)
        if preview_np is None:
            self.preview_label.clear()
            self.preview_label.setText("Open an image to preview intensity changes.")
            if state.get("enabled", False):
                self.preview_status.setText("Load an image to preview intensity changes.")
            else:
                self.preview_status.setText("Palette disabled — showing original image.")
            return
        pixmap = numpy_to_qpixmap(preview_np)
        if pixmap is None:
            self.preview_label.clear()
            self.preview_label.setText("Preview unavailable.")
            self.preview_status.setText("Preview unavailable.")
            return
        self.preview_label.clear()
        scaled = pixmap.scaled(self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                               QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        if state.get("enabled", False):
            self.preview_status.setText("Live preview updates as you tweak controls.")
        else:
            self.preview_status.setText("Palette disabled — showing original image.")

    def accept(self) -> None:
        self.result_state = self._collect_state()
        super().accept()