
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np
from .image_utils import numpy_to_qpixmap

class ImageCanvas(QtWidgets.QGraphicsView):
	def __init__(self, title: str, parent=None):
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
		self._is_space_pressed = False
		self._zoom = 0
		self.title_label = QtWidgets.QLabel(self._title)
		self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.title_label.setStyleSheet("font-weight: 600; font-size: 15px; color: #1a1a1a;")
		self.wrapper = QtWidgets.QWidget()
		v = QtWidgets.QVBoxLayout(self.wrapper)
		v.setContentsMargins(0, 0, 0, 0)
		v.setSpacing(8)
		v.addWidget(self.title_label)
		v.addWidget(self)
	def container(self) -> QtWidgets.QWidget:
		return self.wrapper
	def set_image_np(self, img: np.ndarray):
		if img is None:
			self.clear_image()
			return
		pm = numpy_to_qpixmap(img)
		self.set_pixmap(pm)
	def set_pixmap(self, pixmap: QtGui.QPixmap):
		self._current_pixmap = pixmap
		self._scene.clear()
		self._pixitem = self._scene.addPixmap(pixmap)
		self._scene.setSceneRect(self._pixitem.boundingRect())
		self._zoom = 0
		self.fit_to_window()
	def clear_image(self):
		self._scene.clear()
		self._pixitem = None
		self._current_pixmap = QtGui.QPixmap()
		self._zoom = 0
	def fit_to_window(self):
		if self._pixitem is None:
			return
		self.fitInView(self._pixitem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
		self._zoom = 0
	def zoom_1x(self):
		if self._pixitem is None:
			return
		self.resetTransform()
		rect = self._pixitem.boundingRect()
		self.centerOn(self._pixitem)
		self._zoom = 0
	def wheelEvent(self, event: QtGui.QWheelEvent):
		if self._pixitem is None:
			return
		delta = event.angleDelta().y()
		zoom_in_factor = 1.25
		zoom_out_factor = 1 / zoom_in_factor
		if delta > 0:
			factor = zoom_in_factor
			self._zoom += 1
		else:
			factor = zoom_out_factor
			self._zoom -= 1
		if self._zoom < -10:
			self._zoom = -10
			return
		self.scale(factor, factor)
	def keyPressEvent(self, event: QtGui.QKeyEvent):
		if event.key() == QtCore.Qt.Key.Key_Space and not self._is_space_pressed:
			self._is_space_pressed = True
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
			self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
		super().keyPressEvent(event)
	def keyReleaseEvent(self, event: QtGui.QKeyEvent):
		if event.key() == QtCore.Qt.Key.Key_Space:
			self._is_space_pressed = False
			self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
			self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
		super().keyReleaseEvent(event)
