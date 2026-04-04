from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
from .image_utils import numpy_to_qpixmap


class ImageCanvas(QtWidgets.QGraphicsView):
	def __init__(self, title: str, parent=None, overlay_title: bool = False):
		super().__init__(parent)
		self._scene = QtWidgets.QGraphicsScene(self)
		self.setScene(self._scene)
		self._pixitem = None
		self._title = title
		self._current_pixmap = QtGui.QPixmap()
		self._title_overlay = overlay_title
		self._overlay_lines: list[str] = []
		self._scene_overlay_text = None
		self._zoom = 0
		self._is_space_pressed = False

		self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#2f2f2f")))
		self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
		self.setRenderHints(
			QtGui.QPainter.RenderHint.Antialiasing
			| QtGui.QPainter.RenderHint.SmoothPixmapTransform
		)
		self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
		self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
		self.setMouseTracking(True)

		self.title_label = QtWidgets.QLabel(self._title)
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.title_label.setObjectName("viewTitle")
		self.title_label.setStyleSheet("font-weight: 600; font-size: 15px;")
		self.title_label.setWordWrap(True)

		self.wrapper = QtWidgets.QWidget()
		if self._title_overlay:
			self.title_label.setVisible(False)
			self._build_overlay_wrapper()
			self._init_scene_overlay()
		else:
			self._build_standard_wrapper()

	def container(self) -> QtWidgets.QWidget:
		return self.wrapper

	def _build_standard_wrapper(self):
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		layout = QtWidgets.QVBoxLayout(self.wrapper)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(8)
		layout.addWidget(self.title_label)
		layout.addWidget(self)

	def _build_overlay_wrapper(self):
		layout = QtWidgets.QVBoxLayout(self.wrapper)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setSpacing(0)
		layout.addWidget(self)

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

	def set_overlay_lines(self, lines: list[str] | None):
		if not self._title_overlay:
			return
		if not lines:
			self._overlay_lines = []
		else:
			self._overlay_lines = [str(x).strip() for x in lines if x is not None and str(x).strip()]
		self._update_scene_overlay()

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
		self._scene_overlay_text.setPos(top_left.x() + 10, top_left.y() + 10)

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
		self._scene.clear()
		self._pixitem = None
		self._current_pixmap = QtGui.QPixmap()
		self._zoom = 0
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
		if event.key() == QtCore.Qt.Key.Key_Space and not self._is_space_pressed:
			self._is_space_pressed = True
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
			return
		super().keyPressEvent(event)

	def keyReleaseEvent(self, event: QtGui.QKeyEvent):
		if event.key() == QtCore.Qt.Key.Key_Space:
			self._is_space_pressed = False
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
			self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
			return
		super().keyReleaseEvent(event)

	def resizeEvent(self, event: QtGui.QResizeEvent):
		super().resizeEvent(event)
		self._reposition_scene_overlay()

	def scrollContentsBy(self, dx: int, dy: int):
		super().scrollContentsBy(dx, dy)
		self._reposition_scene_overlay()
