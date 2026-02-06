import os
import platform
import segmentation_models_pytorch as smp
import traceback
from scipy.ndimage import binary_dilation
import numpy as np
import cv2
from torchvision import transforms as T
# .dll loading issue on Windows when using PyTorch - see
# https://github.com/pytorch/pytorch/issues/166628#issuecomment-3479375122

if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass
# Strict note: The above code is necessary to ensure that the PyTorch library loads correctly on Windows, especially when using CUDA. It attempts to load the "c10.dll" library, which is a core component of PyTorch, before any other part of the library (torch) is imported. This can help prevent issues related to missing DLLs that can arise when PyTorch is installed in certain ways on Windows.
# Strict note: import torch before PyQt6
import torch

def set_model_path(path: str):
	global MODEL_PATH
	MODEL_PATH = path
	_model_singleton["model"] = None

MODEL_PATH = "brain_segmentation_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path: str):
	model = smp.Unet(
		encoder_name="efficientnet-b7",
		encoder_weights=None,
		in_channels=3,
		classes=1,
		activation="sigmoid",
	)
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model file not found: {path}")
	try:
		state = torch.load(path, map_location=DEVICE)
	except Exception as exc:
		raise RuntimeError(f"Failed to load model file '{path}': {exc}") from exc
	try:
		model.load_state_dict(state)
	except Exception as exc:
		# Provide helpful details about why the state dict failed to load.
		tb = traceback.format_exc()
		raise RuntimeError(
			f"State dictionary incompatible with expected model architecture.\n" 
			f"File: {path}\nError: {exc}\n\nTraceback:\n{tb}"
		) from exc
	model.to(DEVICE)
	model.eval()
	return model

_model_singleton = {"model": None}
def get_model():
	if _model_singleton["model"] is None:
		_model_singleton["model"] = load_model(MODEL_PATH)
	return _model_singleton["model"]

def _resize_to_multiple_of_32(img: np.ndarray) -> (np.ndarray, tuple[int, int]):
	h, w = img.shape[:2]
	new_h = max(32, int(np.ceil(h / 32) * 32))
	new_w = max(32, int(np.ceil(w / 32) * 32))
	scale = min(new_h / h, new_w / w)
	nh = int(np.round(h * scale))
	nw = int(np.round(w * scale))
	nh = int(np.ceil(nh / 32) * 32)
	nw = int(np.ceil(nw / 32) * 32)
	resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
	return resized, (h, w)

def preprocess_for_model(img_rgb: np.ndarray):
	tensor = T.functional.to_tensor(img_rgb).unsqueeze(0)
	return tensor.float()

def postprocess_mask(mask_tensor: np.ndarray, orig_shape):
	mask_uint8 = (mask_tensor.astype(np.uint8) * 255)
	mask_up = cv2.resize(mask_uint8, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
	return mask_up

def compute_highlight(original_rgb: np.ndarray, mask_uint8: np.ndarray):
	mask_bool = (mask_uint8 > 0).astype(np.uint8)
	edges = np.clip(mask_bool - binary_dilation(mask_bool, structure=np.ones((3, 3))).astype(np.uint8), 0, 1)
	highlighted = original_rgb.copy()
	highlighted[edges.astype(bool)] = [255, 0, 0]
	return highlighted

def run_inference_on_image(img_rgb: np.ndarray):
	model = get_model()
	resized, orig_shape = _resize_to_multiple_of_32(img_rgb)
	inp = preprocess_for_model(resized).to(DEVICE)
	with torch.no_grad():
		pred = model(inp)
	pred_np = pred.squeeze().cpu().numpy()
	if pred_np.ndim == 3:
		pred_np = pred_np[0]
	pred_bin = (pred_np > 0.5).astype(np.uint8)
	mask_up = postprocess_mask(pred_bin, orig_shape)
	highlighted = compute_highlight(img_rgb, mask_up)
	return mask_up, highlighted
