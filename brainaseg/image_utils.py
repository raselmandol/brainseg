
import numpy as np
from PyQt6 import QtGui
import cv2

def numpy_to_qpixmap(img_np: np.ndarray) -> QtGui.QPixmap:
	if img_np is None:
		return QtGui.QPixmap()
	h, w = img_np.shape[:2]
	if img_np.dtype != np.uint8:
		img_np = np.clip(img_np, 0, 255).astype(np.uint8)
	if img_np.ndim == 2:
		qimg = QtGui.QImage(img_np.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
	else:
		qimg = QtGui.QImage(img_np.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
	return QtGui.QPixmap.fromImage(qimg)

def pil_or_cv_to_rgb_np(path_or_array):
	from PIL import Image
	if isinstance(path_or_array, str):
		img = cv2.imread(path_or_array, cv2.IMREAD_UNCHANGED)
		if img is None:
			raise ValueError(f"Unable to read image: {path_or_array}")
		if img.ndim == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img
	elif isinstance(path_or_array, np.ndarray):
		img = path_or_array
		if img.ndim == 2:
			img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		else:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img
	elif isinstance(path_or_array, Image.Image):
		return np.array(path_or_array.convert("RGB"))
	else:
		raise ValueError("Unsupported image type")
