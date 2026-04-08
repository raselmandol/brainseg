
import numpy as np
from PyQt6 import QtGui
import cv2


def _is_nifti_path(path: str) -> bool:
	path_lower = path.lower()
	return path_lower.endswith(".nii") or path_lower.endswith(".nii.gz")


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
	arr = np.asarray(img, dtype=np.float32)
	if arr.size == 0:
		return np.zeros((1, 1), dtype=np.uint8)
	arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
	min_val = float(arr.min())
	max_val = float(arr.max())
	if max_val <= min_val:
		return np.zeros(arr.shape, dtype=np.uint8)
	arr = (arr - min_val) / (max_val - min_val)
	arr = np.clip(arr * 255.0, 0, 255)
	return arr.astype(np.uint8)


def load_nifti_slice(path: str, preferred_slice: int | None = None) -> tuple[np.ndarray, int, int]:
	try:
		import nibabel as nib
	except Exception as exc:
		raise RuntimeError(
			"NIfTI support requires nibabel. Install it with: pip install nibabel"
		) from exc

	data = np.asarray(nib.load(path).get_fdata())
	data = np.squeeze(data)

	if data.ndim == 2:
		return data, 0, 1

	if data.ndim >= 4:
		data = np.asarray(data[..., 0])
		data = np.squeeze(data)

	if data.ndim != 3:
		raise ValueError(f"Unsupported NIfTI shape: {data.shape}")

	total_slices = int(data.shape[2])
	if total_slices <= 0:
		raise ValueError("NIfTI volume has no slices")

	if preferred_slice is None:
		slice_idx = total_slices // 2
	else:
		slice_idx = int(max(0, min(total_slices - 1, preferred_slice)))

	slice_2d = np.asarray(data[:, :, slice_idx])
	return slice_2d, slice_idx, total_slices


def load_nifti_rgb_np(path: str, preferred_slice: int | None = None) -> tuple[np.ndarray, int, int]:
	slice_2d, slice_idx, total_slices = load_nifti_slice(path, preferred_slice=preferred_slice)
	slice_u8 = _normalize_to_uint8(slice_2d)
	rgb = cv2.cvtColor(slice_u8, cv2.COLOR_GRAY2RGB)
	return rgb, slice_idx, total_slices


def get_nifti_spacing(path: str) -> tuple[float, float, float | None] | None:
	if not _is_nifti_path(path):
		return None
	try:
		import nibabel as nib
	except Exception:
		return None
	try:
		header = nib.load(path).header
		zooms = header.get_zooms()
		sx = float(zooms[0]) if len(zooms) > 0 else 0.0
		sy = float(zooms[1]) if len(zooms) > 1 else 0.0
		sz = float(zooms[2]) if len(zooms) > 2 else None
		if sx <= 0 or sy <= 0:
			return None
		return sx, sy, sz
	except Exception:
		return None


def load_mask_np(path: str, preferred_slice: int | None = None) -> tuple[np.ndarray, int | None, int | None]:
	if _is_nifti_path(path):
		slice_2d, slice_idx, total_slices = load_nifti_slice(path, preferred_slice=preferred_slice)
		return slice_2d, slice_idx, total_slices

	if path.lower().endswith(".npy"):
		mask = np.load(path)
		if mask.ndim >= 3:
			mask = np.asarray(mask[..., 0])
		return mask, None, None

	mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if mask is None:
		raise ValueError(f"Unable to read mask file: {path}")
	if mask.ndim == 3:
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	return mask, None, None

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

def pil_or_cv_to_rgb_np(path_or_array, preferred_slice: int | None = None):
	from PIL import Image
	if isinstance(path_or_array, str):
		if _is_nifti_path(path_or_array):
			rgb, _, _ = load_nifti_rgb_np(path_or_array, preferred_slice=preferred_slice)
			return rgb
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
