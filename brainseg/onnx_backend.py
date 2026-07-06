import os
import traceback

import cv2
import numpy as np

from .model import (
	_resize_to_multiple_of_32,
	postprocess_mask,
	compute_highlight,
)


def _normalize_01(img_rgb: np.ndarray) -> np.ndarray:
	"""Scale an uint8 RGB image to float32 in [0, 1]."""
	return img_rgb.astype(np.float32) / 255.0


class ONNXSegmenter:
	"""Run brain abnormality segmentation using ONNX Runtime.

	The ONNX model is expected to accept a [1, 3, H, W] float32 input in the
	[0, 1] range (matching the PyTorch preprocessing pipeline) and return a
	[1, 1, H, W] probability map (sigmoid output).
	"""

	def __init__(self, path: str, providers=None):
		if not os.path.exists(path):
			raise FileNotFoundError(f"ONNX model file not found: {path}")
		try:
			import onnxruntime as ort
		except Exception as exc:
			raise RuntimeError(
				"ONNX Runtime is not installed. Install it with: "
				"pip install onnxruntime (CPU) or onnxruntime-gpu (CUDA)."
			) from exc

		if providers is None:
			available = ort.get_available_providers()
			providers = []
			if "CUDAExecutionProvider" in available:
				providers.append("CUDAExecutionProvider")
			if "DmlExecutionProvider" in available:
				providers.append("DmlExecutionProvider")
			providers.append("CPUExecutionProvider")

		self.path = path
		self.session = ort.InferenceSession(path, providers=providers)
		self.input_name = self.session.get_inputs()[0].name
		self.input_meta = self.session.get_inputs()[0]
		self.output_names = [o.name for o in self.session.get_outputs()]
		self.providers = self.session.get_providers()

	def _prepare_input(self, img_rgb: np.ndarray):
		resized, orig_shape = _resize_to_multiple_of_32(img_rgb)
		normalized = _normalize_01(resized)
		# HWC -> CHW -> NCHW
		chw = np.transpose(normalized, (2, 0, 1))
		nchw = np.ascontiguousarray(chw[np.newaxis, ...], dtype=np.float32)
		return nchw, orig_shape

	def run_inference(self, img_rgb: np.ndarray):
		"""Return (mask_up, highlighted, confidence_summary) like the PyTorch path."""
		nchw, orig_shape = self._prepare_input(img_rgb)
		outputs = self.session.run(self.output_names, {self.input_name: nchw})
		pred = outputs[0]
		# Collapse to 2D probability map
		if pred.ndim == 4:
			pred = pred[0, 0]
		elif pred.ndim == 3:
			pred = pred[0]
		pred = np.squeeze(pred)
		pred_prob = pred.astype(np.float32)

		# If the model emits logits (range outside [0, 1]), apply sigmoid.
		if pred_prob.min() < -0.01 or pred_prob.max() > 1.01:
			pred_prob = 1.0 / (1.0 + np.exp(-pred_prob))
		pred_prob = np.clip(pred_prob, 0.0, 1.0)

		pred_bin = (pred_prob > 0.5).astype(np.uint8)
		mask_up = postprocess_mask(pred_bin, orig_shape)
		pred_prob_up = cv2.resize(
			pred_prob, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR
		)
		mask_bool = mask_up > 0
		if np.any(mask_bool):
			lesion_mean = float(np.mean(pred_prob_up[mask_bool]))
		else:
			lesion_mean = 0.0
		confidence_summary = {
			"lesion_mean": lesion_mean,
			"global_mean": float(np.mean(pred_prob_up)),
			"global_max": float(np.max(pred_prob_up)),
		}
		highlighted = compute_highlight(img_rgb, mask_up)
		return mask_up, highlighted, confidence_summary


def load_onnx_segmenter(path: str) -> ONNXSegmenter:
	"""Create an ONNXSegmenter, surfacing clear errors on failure."""
	try:
		return ONNXSegmenter(path)
	except Exception as exc:
		tb = traceback.format_exc()
		raise RuntimeError(
			f"Failed to load ONNX model '{path}':\n{exc}\n\nTraceback:\n{tb}"
		) from exc
