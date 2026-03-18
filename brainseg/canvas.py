import math
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
import uuid
from .image_utils import numpy_to_qpixmap


class _VertexHandle(QtWidgets.QGraphicsEllipseItem):
	def __init__(self, canvas: 'ImageCanvas', index: int, pos: QtCore.QPointF):
		super().__init__(-5, -5, 10, 10)
		self.canvas = canvas
		self.index = index
		self.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
		self.setPen(QtGui.QPen(QtGui.QColor("#1f2933"), 1))
		self.setFlags(
			QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable
			| QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
			| QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
		)
		self.setZValue(7)
		self.setPos(pos)

	def itemChange(self, change: 'QtWidgets.QGraphicsItem.GraphicsItemChange', value):
		if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionChange:
			self.canvas._on_handle_moved(self.index, value)
		return super().itemChange(change, value)


class ImageCanvas(QtWidgets.QGraphicsView):
	annotationCommitted = QtCore.pyqtSignal(str, str, list, bool)
	annotationEditCanceled = QtCore.pyqtSignal(str, str)

	def __init__(self, title: str, parent=None, overlay_title: bool = False):
		super().__init__(parent)
		self._scene = QtWidgets.QGraphicsScene(self)
		self.setScene(self._scene)
		self._pixitem = None
		self._title = title
		self._current_pixmap = QtGui.QPixmap()
		self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#2f2f2f")))
		self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
		self.setRenderHints(
			QtGui.QPainter.RenderHint.Antialiasing
			| QtGui.QPainter.RenderHint.SmoothPixmapTransform
		)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
		self.setMouseTracking(True)
		self._is_space_pressed = False
		self._zoom = 0

		self._title_overlay = overlay_title
		self.title_label = QtWidgets.QLabel(self._title)
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.title_label.setObjectName("viewTitle")
		self.title_label.setStyleSheet("font-weight: 600; font-size: 15px;")
		self.title_label.setWordWrap(True)
		self._scene_overlay_text = None

		self.wrapper = QtWidgets.QWidget()
		self._overlay_lines: list[str] = []
		if self._title_overlay:
			self.title_label.setVisible(False)
			self._build_overlay_wrapper()
			self._init_scene_overlay()
		else:
			self._build_standard_wrapper()
		self._update_title_label()

		self._annotation_active = False
		self._annotation_mode = None
		self._editing_shape_id = None
		self._annotation_label = None
		self._annotation_color = QtGui.QColor("#ff0000")
		self._annotation_points = []
		self._annotation_preview_pos = None
		self._annotation_temp_path = None
		self._annotation_handles = []
		self._annotation_items = {}
		self._draw_handles = []
		self._first_point_marker = None
		self._snap_threshold_px = 12.0

	def container(self) -> QtWidgets.QWidget:
		return self.wrapper

	def _update_title_label(self):
		if self._title_overlay:
			text = self._title
			self.title_label.setVisible(False)
		else:
			text = self._title
		self.title_label.setText(text)

	def set_overlay_lines(self, lines: list[str] | None):
		if not self._title_overlay:
			return
		if not lines:
			self._overlay_lines = []
		else:
			cleaned = []
			for entry in lines:
				if entry is None:
					continue
				text = str(entry).strip()
				if text:
					cleaned.append(text)
			self._overlay_lines = cleaned
		self._update_scene_overlay()

	def _init_scene_overlay(self):
		if not self._title_overlay or self._scene is None:
			return
		self._scene_overlay_text = QtWidgets.QGraphicsTextItem()
		self._scene_overlay_text.setZValue(10)
		self._scene_overlay_text.setVisible(False)
		self._scene_overlay_text.setFlag(
			QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
		)
		self._scene.addItem(self._scene_overlay_text)

	def _update_scene_overlay(self):
		if not self._title_overlay:
			return
		if self._scene_overlay_text is None:
			self._init_scene_overlay()
		if self._scene_overlay_text is None:
			return
		try:
			_ = self._scene_overlay_text.isVisible()
		except RuntimeError:
			self._init_scene_overlay()
			if self._scene_overlay_text is None:
				return
		if not self._overlay_lines:
			self._scene_overlay_text.setVisible(False)
			return
		body = self._title + "<br>" + "<br>".join(self._overlay_lines)
		html = (
			"<div style='"
			"background-color: rgba(255,255,255,225);"
			"color: #ff4d88;"
			"font-weight: 600;"
			"font-size: 13px;"
			"padding: 6px 8px;"
			"border-radius: 8px;"
			"'>"
			+ body
			+ "</div>"
		)
		self._scene_overlay_text.setHtml(html)
		self._scene_overlay_text.setVisible(True)
		self._reposition_scene_overlay()

	def _reposition_scene_overlay(self):
		if not self._scene_overlay_text:
			return
		try:
			visible = self._scene_overlay_text.isVisible()
		except RuntimeError:
			self._init_scene_overlay()
			return
		if not visible:
			return
		top_left = self.mapToScene(self.viewport().rect().topLeft())
		margin = 10
		self._scene_overlay_text.setPos(top_left.x() + margin, top_left.y() + margin)

	def resizeEvent(self, event: QtGui.QResizeEvent):
		super().resizeEvent(event)
		self._reposition_scene_overlay()

	def scrollContentsBy(self, dx: int, dy: int):
		super().scrollContentsBy(dx, dy)
		self._reposition_scene_overlay()

	def _build_standard_wrapper(self):
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		layout = QtWidgets.QVBoxLayout(self.wrapper)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(8)
		layout.addWidget(self.title_label)
		layout.addWidget(self)

	def _build_overlay_wrapper(self):
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
		self.title_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		overlay_style = self.title_label.styleSheet() + " padding: 4px 10px; background-color: rgba(255, 255, 255, 0.85); color: #1f2933; border-radius: 8px;"
		self.title_label.setStyleSheet(overlay_style)
		layout = QtWidgets.QVBoxLayout(self.wrapper)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)
		overlay_container = QtWidgets.QWidget()
		stack = QtWidgets.QStackedLayout(overlay_container)
		stack.setStackingMode(QtWidgets.QStackedLayout.StackingMode.StackAll)
		stack.setContentsMargins(0, 0, 0, 0)
		stack.setSpacing(0)
		stack.addWidget(self)
		title_holder = QtWidgets.QWidget()
		title_holder.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
		title_layout = QtWidgets.QVBoxLayout(title_holder)
		title_layout.setContentsMargins(12, 12, 12, 12)
		title_layout.addWidget(
			self.title_label,
			0,
			QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
		)
		stack.addWidget(title_holder)
		layout.addWidget(overlay_container)

	def has_image(self) -> bool:
		return self._pixitem is not None

	def set_image_np(self, img: np.ndarray):
		if img is None:
			self.clear_image()
			return
		self.set_pixmap(numpy_to_qpixmap(img))

	def update_image_np(self, img: np.ndarray):
		if img is None:
			self.clear_image()
			return
		self.update_pixmap(numpy_to_qpixmap(img))

	def set_pixmap(self, pixmap: QtGui.QPixmap):
		self._current_pixmap = pixmap
		self._scene.clear()
		self._scene_overlay_text = None
		self._annotation_temp_path = None
		self._clear_handles()
		self._annotation_items = {}
		self._annotation_preview_pos = None
		self._annotation_active = False
		self._annotation_mode = None
		self._editing_shape_id = None
		self._pixitem = self._scene.addPixmap(pixmap)
		self._scene.setSceneRect(self._pixitem.boundingRect())
		self._zoom = 0
		if self._title_overlay:
			self._init_scene_overlay()
			self._update_scene_overlay()
		self.fit_to_window()
		self._reposition_scene_overlay()

	def update_pixmap(self, pixmap: QtGui.QPixmap):
		if self._pixitem is None:
			self.set_pixmap(pixmap)
			return
		current_transform = QtGui.QTransform(self.transform())
		view_center = self.mapToScene(self.viewport().rect().center())
		self._current_pixmap = pixmap
		self._pixitem.setPixmap(pixmap)
		self._scene.setSceneRect(self._pixitem.boundingRect())
		self.setTransform(current_transform, False)
		self.centerOn(view_center)
		self._reposition_scene_overlay()

	def clear_image(self):
		self._finish_annotation_state()
		self._scene.clear()
		self._pixitem = None
		self._current_pixmap = QtGui.QPixmap()
		self._zoom = 0
		self._annotation_temp_path = None
		self._annotation_items = {}
		self._scene_overlay_text = None

	def fit_to_window(self):
		if self._pixitem is None:
			return
		self.fitInView(self._pixitem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
		self._zoom = 0
		self._reposition_scene_overlay()

	def zoom_1x(self):
		if self._pixitem is None:
			return
		self.resetTransform()
		self.centerOn(self._pixitem)
		self._zoom = 0
		self._reposition_scene_overlay()

	def wheelEvent(self, event: QtGui.QWheelEvent):
		if self._pixitem is None:
			return
		delta = event.angleDelta().y()
		factor = 1.25 if delta > 0 else 0.8
		self._zoom += 1 if delta > 0 else -1
		if self._zoom < -10:
			self._zoom = -10
			return
		self.scale(factor, factor)
		self._reposition_scene_overlay()

	def keyPressEvent(self, event: QtGui.QKeyEvent):
		key = event.key()
		if key == QtCore.Qt.Key.Key_Space and not self._is_space_pressed:
			self._is_space_pressed = True
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
			return
		if key == QtCore.Qt.Key.Key_Escape and self._annotation_active:
			self._cancel_annotation()
			return
		if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and self._annotation_active and self._annotation_mode == "edit":
			self._finalize_annotation()
			return
		if self._annotation_active and self._annotation_mode == "draw":
			if key == QtCore.Qt.Key.Key_Backspace:
				self._remove_last_annotation_point()
				return
			if key == QtCore.Qt.Key.Key_Z and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
				self._remove_last_annotation_point()
				return
		super().keyPressEvent(event)

	def keyReleaseEvent(self, event: QtGui.QKeyEvent):
		if event.key() == QtCore.Qt.Key.Key_Space:
			self._is_space_pressed = False
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
			self._set_default_cursor()
			return
		super().keyReleaseEvent(event)

	def begin_annotation(self, label_name: str, color_hex: str):
		if self._pixitem is None:
			return False
		self._annotation_active = True
		self._annotation_mode = "draw"
		self._editing_shape_id = None
		self._annotation_label = label_name
		self._annotation_color = QtGui.QColor(color_hex)
		self._annotation_points = []
		self._annotation_preview_pos = None
		self._clear_handles()
		self._clear_draw_handles()
		self._remove_first_point_marker()
		self._ensure_temp_path(color_hex)
		self._update_temp_path()
		self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
		self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
		self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
		return True

	def edit_annotation(self, label_name: str, color_hex: str, points: list[tuple[float, float]], shape_id: str):
		if self._pixitem is None or len(points) < 3:
			return False
		self.remove_annotation_item(shape_id)
		self._annotation_active = True
		self._annotation_mode = "edit"
		self._editing_shape_id = shape_id
		self._annotation_label = label_name
		self._annotation_color = QtGui.QColor(color_hex)
		self._annotation_points = [QtCore.QPointF(float(x), float(y)) for x, y in points]
		self._annotation_preview_pos = None
		self._clear_draw_handles()
		self._remove_first_point_marker()
		self._create_handles()
		self._ensure_temp_path(color_hex)
		self._update_temp_path()
		self.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
		self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
		self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
		return True

	def add_annotation_polygon(self, label_name: str, shape_id: str, points: list[tuple[float, float]], color_hex: str):
		if self._pixitem is None or len(points) < 3:
			return
		qpoints = [QtCore.QPointF(float(x), float(y)) for x, y in points]
		item = self._create_polygon_item(qpoints, color_hex)
		self._annotation_items[shape_id] = {"label": label_name, "item": item}

	def remove_annotation_item(self, shape_id: str):
		entry = self._annotation_items.pop(shape_id, None)
		if entry:
			self._scene.removeItem(entry["item"])

	def clear_annotations(self, label_name: str | None = None):
		if label_name is None:
			ids = list(self._annotation_items.keys())
		else:
			ids = [sid for sid, entry in self._annotation_items.items() if entry["label"] == label_name]
		for sid in ids:
			self.remove_annotation_item(sid)
		if label_name is None or self._annotation_label == label_name:
			self._finish_annotation_state()

	def mousePressEvent(self, event: QtGui.QMouseEvent):
		if self._annotation_active and self._annotation_mode == "draw":
			scene_pos = self.mapToScene(event.position().toPoint())
			if event.button() == QtCore.Qt.MouseButton.LeftButton:
				if len(self._annotation_points) >= 3 and self._is_near_first_point(scene_pos):
					self._finalize_annotation()
					return
				self._append_annotation_point(scene_pos)
				return
			if event.button() == QtCore.Qt.MouseButton.RightButton and self._annotation_points:
				self._remove_last_annotation_point()
				return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event: QtGui.QMouseEvent):
		if self._annotation_active and self._annotation_mode == "draw":
			scene_pos = self.mapToScene(event.position().toPoint())
			if self._annotation_points:
				self._annotation_preview_pos = scene_pos
				self._update_temp_path()
			self._update_first_point_marker(scene_pos)
		super().mouseMoveEvent(event)

	def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
		if not self._annotation_active:
			super().mouseDoubleClickEvent(event)
			return
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			if self._annotation_mode == "draw" and len(self._annotation_points) >= 3:
				self._finalize_annotation()
				return
			if self._annotation_mode == "edit":
				self._finalize_annotation()
				return
		super().mouseDoubleClickEvent(event)

	def _finalize_annotation(self):
		if len(self._annotation_points) < 3 or not self._annotation_label:
			self._cancel_annotation()
			return
		is_edit = self._annotation_mode == "edit" and self._editing_shape_id is not None
		shape_id = self._editing_shape_id if is_edit else str(uuid.uuid4())
		color_hex = self._annotation_color.name()
		points_payload = [(pt.x(), pt.y()) for pt in self._annotation_points]
		self.add_annotation_polygon(self._annotation_label, shape_id, points_payload, color_hex)
		self.annotationCommitted.emit(self._annotation_label, shape_id, points_payload, is_edit)
		self._finish_annotation_state()

	def _ensure_temp_path(self, color_hex: str):
		if self._annotation_temp_path is None:
			self._annotation_temp_path = QtWidgets.QGraphicsPathItem()
			self._annotation_temp_path.setZValue(5)
			self._scene.addItem(self._annotation_temp_path)
		pen = QtGui.QPen(QtGui.QColor(color_hex))
		pen.setStyle(QtCore.Qt.PenStyle.DashLine)
		pen.setWidth(2)
		self._annotation_temp_path.setPen(pen)
		self._annotation_temp_path.setVisible(True)

	def _create_polygon_item(self, qpoints: list[QtCore.QPointF], color_hex: str) -> QtWidgets.QGraphicsPolygonItem:
		polygon = QtGui.QPolygonF(qpoints)
		item = QtWidgets.QGraphicsPolygonItem(polygon)
		fill_color = QtGui.QColor(color_hex)
		fill_color.setAlpha(80)
		item.setBrush(QtGui.QBrush(fill_color))
		item.setPen(QtGui.QPen(QtGui.QColor(color_hex), 2))
		item.setZValue(5)
		self._scene.addItem(item)
		return item

	def _create_handles(self):
		self._clear_handles()
		for idx, point in enumerate(self._annotation_points):
			handle = _VertexHandle(self, idx, point)
			self._scene.addItem(handle)
			self._annotation_handles.append(handle)

	def _on_handle_moved(self, index: int, proposed_pos: QtCore.QPointF) -> QtCore.QPointF:
		rect = self._scene.sceneRect()
		clamped = QtCore.QPointF(
			min(max(rect.left(), proposed_pos.x()), rect.right()),
			min(max(rect.top(), proposed_pos.y()), rect.bottom()),
		)
		if 0 <= index < len(self._annotation_points):
			self._annotation_points[index] = clamped
			self._update_temp_path()
		return clamped

	def _update_temp_path(self):
		if not self._annotation_temp_path:
			return
		if not self._annotation_points:
			self._annotation_temp_path.setPath(QtGui.QPainterPath())
			return
		path = QtGui.QPainterPath(self._annotation_points[0])
		for pt in self._annotation_points[1:]:
			path.lineTo(pt)
		if self._annotation_mode == "draw" and self._annotation_preview_pos is not None:
			path.lineTo(self._annotation_preview_pos)
		elif self._annotation_mode == "edit" and len(self._annotation_points) >= 3:
			path.closeSubpath()
		self._annotation_temp_path.setPath(path)

	def _finish_annotation_state(self):
		self._annotation_active = False
		self._annotation_mode = None
		self._editing_shape_id = None
		self._annotation_label = None
		self._annotation_points = []
		self._annotation_preview_pos = None
		self._clear_handles()
		self._clear_draw_handles()
		self._remove_first_point_marker()
		self._cancel_annotation_path()
		self._set_default_cursor()

	def _cancel_annotation(self):
		was_edit = self._annotation_mode == "edit"
		label = self._annotation_label
		shape_id = self._editing_shape_id
		self._finish_annotation_state()
		if was_edit and label and shape_id:
			self.annotationEditCanceled.emit(label, shape_id)

	def _cancel_annotation_path(self):
		if self._annotation_temp_path:
			self._annotation_temp_path.setPath(QtGui.QPainterPath())
			self._annotation_temp_path.setVisible(False)

	def _clear_handles(self):
		for handle in self._annotation_handles:
			self._scene.removeItem(handle)
		self._annotation_handles = []

	def _append_annotation_point(self, scene_pos: QtCore.QPointF):
		self._annotation_points.append(scene_pos)
		self._annotation_preview_pos = None
		self._update_temp_path()
		self._sync_draw_handles()
		self._update_first_point_marker(scene_pos)

	def _remove_last_annotation_point(self):
		if not self._annotation_points:
			return
		self._annotation_points.pop()
		self._annotation_preview_pos = None
		self._update_temp_path()
		self._sync_draw_handles()
		self._update_first_point_marker()

	def _sync_draw_handles(self):
		if self._annotation_mode != "draw":
			self._clear_draw_handles()
			return
		self._clear_draw_handles()
		for point in self._annotation_points:
			marker = QtWidgets.QGraphicsEllipseItem(-3, -3, 6, 6)
			marker.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
			marker.setPen(QtGui.QPen(QtGui.QColor("#111827"), 1))
			marker.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
			marker.setZValue(6)
			marker.setPos(point)
			self._scene.addItem(marker)
			self._draw_handles.append(marker)

	def _clear_draw_handles(self):
		for marker in self._draw_handles:
			self._scene.removeItem(marker)
		self._draw_handles = []

	def _update_first_point_marker(self, hover_pos: QtCore.QPointF | None = None):
		if self._annotation_mode != "draw" or not self._annotation_points:
			self._remove_first_point_marker()
			return
		first_point = self._annotation_points[0]
		if self._first_point_marker is None:
			marker = QtWidgets.QGraphicsEllipseItem(-5, -5, 10, 10)
			marker.setZValue(6)
			marker.setBrush(QtGui.QBrush(QtGui.QColor("#fffbeb")))
			marker.setPen(QtGui.QPen(QtGui.QColor("#f97316"), 2))
			marker.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
			self._scene.addItem(marker)
			self._first_point_marker = marker
		self._first_point_marker.setPos(first_point)
		is_hot = hover_pos is not None and self._is_near_first_point(hover_pos)
		color = QtGui.QColor("#fb923c" if is_hot else "#fffbeb")
		self._first_point_marker.setBrush(QtGui.QBrush(color))

	def _remove_first_point_marker(self):
		if self._first_point_marker is not None:
			self._scene.removeItem(self._first_point_marker)
			self._first_point_marker = None

	def _is_near_first_point(self, scene_pos: QtCore.QPointF) -> bool:
		if not self._annotation_points:
			return False
		first = self._annotation_points[0]
		dx = scene_pos.x() - first.x()
		dy = scene_pos.y() - first.y()
		dist = math.hypot(dx, dy)
		scale_x = abs(self.transform().m11()) or 1.0
		scale_y = abs(self.transform().m22()) or 1.0
		max_scale = max(scale_x, scale_y, 1e-3)
		threshold = self._snap_threshold_px / max_scale
		return dist <= threshold
	def _set_default_cursor(self):
		if self._is_space_pressed:
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
		else:
			self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
