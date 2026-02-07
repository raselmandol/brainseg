import cv2
import os
import sys
import traceback
import psutil, time
import numpy as np
import json
import uuid
from typing import Optional
from .model import get_model, run_inference_on_image, MODEL_PATH # torch needed to be installed before importing pyqt6
from .canvas import ImageCanvas
from .image_utils import pil_or_cv_to_rgb_np, numpy_to_qpixmap
from PyQt6 import QtCore, QtGui, QtWidgets
from .worker import InferenceWorker
from .help_window import HelpWindow
from .settings_window import SettingsWindow
from .intensity_palette import IntensityPaletteDialog, CurveEditor
from .theme import LIGHT_THEME, DARK_THEME, get_icon_path
from .annotation_window import AnnotationWindow
from .statistics_tracker import statistics_tracker
from .statistics_window import StatisticsWindow
# .dll loading issue on Windows when using PyTorch - see
# https://github.com/pytorch/pytorch/issues/166628#issuecomment-3479375122

ACCENT_COLORS = {
	"Azure": "#1a73e8",
	"Emerald": "#2ecc71",
	"Amber": "#f4b400",
	"Rose": "#ff4d6d",
}


class SegmentationApp(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Brain Abnormality Segmentation")
		self.resize(1100, 550)
		self.theme = "light"
		self.theme_style = 1  # 1 or 2, for icon style
		self.statistics_window = None
		self.settings_window = None
		self.annotation_window = None
		self.accent_name = "Azure"
		self.accent_color = ACCENT_COLORS[self.accent_name]
		self.theme_color = None
		self._base_theme_stylesheet = LIGHT_THEME
		# PyInstaller missing icon issue
		# In PyInstaller, data files are unpacked to sys._MEIPASS
		self.assets_dir = None
		if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
			candidates = [
				os.path.join(sys._MEIPASS, 'brainseg', 'assets'),  # (Use) when bundled with dest 'brainseg/assets'
				os.path.join(sys._MEIPASS, 'assets'),              # (Use) when bundled with dest 'assets'
			]
			for p in candidates:
				if os.path.exists(p):
					self.assets_dir = p
					break
		if not self.assets_dir:
			try:
				exe_dir = os.path.dirname(sys.executable)
				candidates = [
					os.path.join(exe_dir, 'brainseg', 'assets'),
					os.path.join(exe_dir, 'assets'),
				]
				for p in candidates:
					if os.path.exists(p):
						self.assets_dir = p
						break
			except Exception:
				pass
		if not self.assets_dir:
			module_dir = os.path.dirname(os.path.abspath(__file__))
			self.assets_dir = os.path.join(module_dir, 'assets')
		self.setStyleSheet(LIGHT_THEME)
		self.current_image = None
		self.base_image = None  # store original loaded image (next-->temp)
		self.current_mask = None
		self.current_highlight = None
		self.current_path = None
		self.ground_truth_mask = None
		self.ground_truth_path = None
		self.annotation_labels = {}
		self.annotation_shapes = {}
		self.annotation_list = None
		self.btn_add_label = None
		self.btn_remove_label = None
		self.btn_start_polygon = None
		self.btn_export_labels = None
		self.annotation_shape_table = None
		self.btn_rename_label = None
		self.btn_edit_shape = None
		self.btn_delete_shape = None
		self._pending_edit_backup = None
		# Brightness adjustment state (-100..100)
		self.brightness_value = 0
		self.contrast_value = 0
		# Intensity / Window-Level parameters (for medical imaging)
		self.wl_center = 128
		self.wl_width = 256
		self.gamma = 1.0
		self.colormap = "Gray"
		self.apply_wl_to_model = False
		self.curve_points = [(0.0, 0.0), (1.0, 1.0)]
		self.intensity_enabled = False
		self.canvas_orig = ImageCanvas("Original")
		self.canvas_mask = ImageCanvas("Segmented Mask", overlay_title=True)
		self.canvas_high = ImageCanvas("Highlighted Region")
		self.canvas_orig.annotationCommitted.connect(self._on_annotation_committed)
		self.canvas_orig.annotationEditCanceled.connect(self._on_annotation_edit_canceled)
		# View background colors (None means default automatic)
		self.default_view_bg_color = "#2f2f2f"
		self.view_bg_original = self.default_view_bg_color
		self.view_bg_mask = self.default_view_bg_color
		self.view_bg_highlight = self.default_view_bg_color
		self._settings = QtCore.QSettings("BrainSeg", "BrainSeg")
		self._load_user_preferences()
		left_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
		left_splitter.addWidget(self.canvas_orig.container())
		left_splitter.addWidget(self.canvas_mask.container())
		left_splitter.setHandleWidth(1)
		left_splitter.setStyleSheet("QSplitter::handle { background-color: transparent; border: none; margin: 0px; }")
		left_splitter.setSizes([1, 1])
		views_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		views_splitter.addWidget(left_splitter)
		views_splitter.addWidget(self.canvas_high.container())
		views_splitter.setHandleWidth(1)
		views_splitter.setStyleSheet("QSplitter::handle { background-color: transparent; border: none; margin: 0px; }")
		views_splitter.setSizes([2, 1])
		center = QtWidgets.QWidget()
		center_layout = QtWidgets.QVBoxLayout(center)
		center_layout.setContentsMargins(0, 0, 0, 0)
		center_layout.setSpacing(0)
		center_layout.addWidget(views_splitter)
		self.setCentralWidget(center)
		self._build_left_dock()
		self._build_menubar()
		self._build_toolbar()
		# self._build_footer()
		self.set_theme(self.theme, persist=False)
		self.set_accent(self.accent_name, persist=False)

	def show_statistics_window(self):
		if self.statistics_window is None or not self.statistics_window.isVisible():
			self.statistics_window = StatisticsWindow(self)
			self.statistics_window.show()
		else:
			self.statistics_window.raise_()
			self.statistics_window.activateWindow()

	def _get_theme_icon(self):
		# Always use night-mode1.png for light, white-mode1.png for dark
		if self.theme == "light":
			icon_file = "night_mode1.png"
		else:
			icon_file = "white_mode1.png"
		icon_path = os.path.join(self.assets_dir, icon_file)
		if not os.path.exists(icon_path):
			print(f"Warning: Icon not found at {icon_path}")
		return QtGui.QIcon(icon_path)
	def _build_left_dock(self):
		dock = QtWidgets.QDockWidget("Controls", self)
		dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
		panel = QtWidgets.QWidget()
		v = QtWidgets.QVBoxLayout(panel)
		v.setContentsMargins(10, 10, 10, 10)
		v.setSpacing(10)
		btn_open = QtWidgets.QPushButton("Open Image")
		btn_open.clicked.connect(self.action_open_image)
		btn_ground_truth = QtWidgets.QPushButton("Load Ground Truth")
		btn_ground_truth.clicked.connect(self.action_load_ground_truth)
		btn_select_model = QtWidgets.QPushButton("Select Model File")
		btn_select_model.clicked.connect(self.action_select_model)
		btn_run = QtWidgets.QPushButton("Run Segmentation")
		btn_run.clicked.connect(self.action_run_segmentation)
		btn_save_mask = QtWidgets.QPushButton("Save Mask")
		btn_save_mask.clicked.connect(self.action_save_mask)
		btn_save_high = QtWidgets.QPushButton("Save Highlight")
		btn_save_high.clicked.connect(self.action_save_highlight)
		v.addWidget(btn_open)
		v.addWidget(btn_ground_truth)
		v.addWidget(btn_select_model)
		v.addWidget(btn_run)
		v.addWidget(btn_save_mask)
		v.addWidget(btn_save_high)
		v.addSpacing(8)
		# Brightness slider
		brightness_row = QtWidgets.QHBoxLayout()
		lbl_b = QtWidgets.QLabel("Brightness:")
		lbl_b.setObjectName("hint")
		self.brightness_label = QtWidgets.QLabel("0")
		self.brightness_label.setFixedWidth(28)
		self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.brightness_slider.setRange(-100, 100)
		self.brightness_slider.setSingleStep(1)
		self.brightness_slider.setPageStep(10)
		self.brightness_slider.setValue(0)
		self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
		brightness_row.addWidget(lbl_b)
		brightness_row.addWidget(self.brightness_slider)
		brightness_row.addWidget(self.brightness_label)
		v.addLayout(brightness_row)
		# Contrast slider
		contrast_row = QtWidgets.QHBoxLayout()
		lbl_c = QtWidgets.QLabel("Contrast:   ")
		lbl_c.setObjectName("hint")
		self.contrast_label = QtWidgets.QLabel("0")
		self.contrast_label.setFixedWidth(28)
		self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
		self.contrast_slider.setRange(-100, 100)
		self.contrast_slider.setSingleStep(1)
		self.contrast_slider.setPageStep(10)
		self.contrast_slider.setValue(0)
		self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
		contrast_row.addWidget(lbl_c)
		contrast_row.addWidget(self.contrast_slider)
		contrast_row.addWidget(self.contrast_label)
		v.addLayout(contrast_row)
		v.addSpacing(8)
		controls_frame = QtWidgets.QFrame()
		controls_layout = QtWidgets.QHBoxLayout(controls_frame)
		controls_layout.setContentsMargins(0, 0, 0, 0)
		btn_fit = QtWidgets.QPushButton("Fit")
		btn_fit.setToolTip("Fit all views")
		btn_fit.clicked.connect(self.fit_all)
		btn_1x = QtWidgets.QPushButton("1:1")
		btn_1x.setToolTip("Reset zoom to 1:1")
		btn_1x.clicked.connect(self.one_to_one_all)
		controls_layout.addWidget(btn_fit)
		controls_layout.addWidget(btn_1x)
		v.addWidget(controls_frame)
		v.addSpacing(8)
		self.label_filename = QtWidgets.QLabel("No image loaded")
		self.label_filename.setWordWrap(True)
		v.addWidget(self.label_filename)

		# Ground-truth row: thumbnail + status label
		gt_row = QtWidgets.QWidget()
		gt_layout = QtWidgets.QHBoxLayout(gt_row)
		gt_layout.setContentsMargins(0, 0, 0, 0)
		gt_layout.setSpacing(8)
		self.ground_truth_thumb = QtWidgets.QLabel()
		self.ground_truth_thumb.setFixedSize(88, 64)
		self.ground_truth_thumb.setScaledContents(False)
		self.ground_truth_thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		self.ground_truth_thumb.setStyleSheet("border: 1px solid #283f45; background: transparent;")
		self.label_ground_truth = QtWidgets.QLabel("Ground truth: not loaded")
		self.label_ground_truth.setObjectName("hint")
		self.label_ground_truth.setWordWrap(True)
		gt_layout.addWidget(self.ground_truth_thumb)
		gt_layout.addWidget(self.label_ground_truth, 1)
		v.addWidget(gt_row)
		self.status_label = QtWidgets.QLabel("Ready")
		self.status_label.setObjectName("hint")
		v.addWidget(self.status_label)

		# Segmentation progress bar
		self.segmentation_progress = QtWidgets.QProgressBar()
		self.segmentation_progress.setRange(0, 100)
		self.segmentation_progress.setValue(0)
		self.segmentation_progress.setTextVisible(False)
		self.segmentation_progress.setFixedHeight(12)
		self.segmentation_progress.setToolTip("Segmentation progress")
		v.addWidget(self.segmentation_progress)
		self._apply_accent_palette()
		v.addStretch()
		panel.setMinimumWidth(240)
		dock.setWidget(panel)
		self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

	def _refresh_annotation_list(self, select_name: Optional[str] = None):
		if self.annotation_list is None:
			return
		self.annotation_list.blockSignals(True)
		self.annotation_list.clear()
		for label_name, color_hex in self.annotation_labels.items():
			item = QtWidgets.QListWidgetItem(label_name)
			item.setData(QtCore.Qt.ItemDataRole.UserRole, label_name)
			color = QtGui.QColor(color_hex)
			pix = QtGui.QPixmap(14, 14)
			pix.fill(color)
			item.setIcon(QtGui.QIcon(pix))
			item.setToolTip(color_hex)
			brush = QtGui.QBrush(color.lighter(160))
			item.setBackground(brush)
			self.annotation_list.addItem(item)
			if select_name and label_name == select_name:
				self.annotation_list.setCurrentItem(item)
		self.annotation_list.blockSignals(False)
		if select_name:
			for row in range(self.annotation_list.count()):
				item = self.annotation_list.item(row)
				if item.data(QtCore.Qt.ItemDataRole.UserRole) == select_name:
					self.annotation_list.setCurrentItem(item)
					break
		self._refresh_shape_table(select_name)
		self._update_annotation_buttons()

	def _current_label_selection(self) -> Optional[str]:
		if self.annotation_list is None:
			return None
		item = self.annotation_list.currentItem()
		if not item:
			return None
		return item.data(QtCore.Qt.ItemDataRole.UserRole)

	def _current_shape_selection(self) -> Optional[tuple[str, str]]:
		if self.annotation_shape_table is None:
			return None
		row = self.annotation_shape_table.currentRow()
		if row < 0:
			return None
		item = self.annotation_shape_table.item(row, 0)
		if item is None:
			return None
		shape_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
		label = self._current_label_selection()
		if not label or not shape_id:
			return None
		return (label, shape_id)

	def _refresh_shape_table(self, label_name: Optional[str] = None, select_shape_id: Optional[str] = None):
		if self.annotation_shape_table is None:
			return
		self.annotation_shape_table.blockSignals(True)
		if label_name is None:
			label_name = self._current_label_selection()
		shapes = self.annotation_shapes.get(label_name, []) if label_name else []
		self.annotation_shape_table.setRowCount(len(shapes))
		for row, shape in enumerate(shapes):
			shape_id = shape.get("id")
			if not shape_id:
				shape_id = f"{label_name or 'shape'}_{row+1}"
				shape["id"] = shape_id
			points = shape.get("points", [])
			id_item = QtWidgets.QTableWidgetItem(shape_id or "")
			id_item.setData(QtCore.Qt.ItemDataRole.UserRole, shape_id)
			id_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
			verts_item = QtWidgets.QTableWidgetItem(str(len(points)))
			verts_item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
			self.annotation_shape_table.setItem(row, 0, id_item)
			self.annotation_shape_table.setItem(row, 1, verts_item)
			if select_shape_id and shape_id == select_shape_id:
				self.annotation_shape_table.selectRow(row)
		self.annotation_shape_table.blockSignals(False)
		self._update_annotation_buttons()

	def _handle_label_selection_change(self):
		current = self._current_label_selection()
		self._refresh_shape_table(current)
		self._update_annotation_buttons()

	def _on_shape_selection_change(self):
		self._update_annotation_buttons()

	def _add_annotation_label(self):
		name, ok = QtWidgets.QInputDialog.getText(self, "Add Label", "Label name:")
		if not ok:
			return
		name = name.strip()
		if not name:
			return
		if name in self.annotation_labels:
			QtWidgets.QMessageBox.warning(self, "Annotations", f"Label '{name}' already exists.")
			return
		color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.accent_color), self, "Choose label color")
		if not color.isValid():
			return
		self.annotation_labels[name] = color.name()
		self.annotation_shapes[name] = []
		self._refresh_annotation_list(select_name=name)
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Label '{name}' added. Select it and click Start Polygon to annotate.")

	def _remove_annotation_label(self):
		if self.annotation_list is None:
			return
		item = self.annotation_list.currentItem()
		if not item:
			return
		label_name = item.data(QtCore.Qt.ItemDataRole.UserRole)
		if not label_name:
			return
		self.annotation_labels.pop(label_name, None)
		self.annotation_shapes.pop(label_name, None)
		self.canvas_orig.clear_annotations(label_name)
		self._refresh_annotation_list()
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Label '{label_name}' removed.")

	def _start_polygon_annotation(self):
		if self.base_image is None:
			QtWidgets.QMessageBox.information(self, "Annotations", "Load an image before annotating.")
			return
		if self.annotation_list is None:
			return
		item = self.annotation_list.currentItem()
		if not item:
			QtWidgets.QMessageBox.information(self, "Annotations", "Select a label to annotate.")
			return
		label_name = item.data(QtCore.Qt.ItemDataRole.UserRole)
		color_hex = self.annotation_labels.get(label_name)
		if not color_hex:
			QtWidgets.QMessageBox.warning(self, "Annotations", "Selected label is not available.")
			return
		self.canvas_orig.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
		started = self.canvas_orig.begin_annotation(label_name, color_hex)
		if not started:
			QtWidgets.QMessageBox.warning(self, "Annotations", "Unable to start annotation on the current view.")
			return
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Annotating '{label_name}'. Double-click to finish, Esc to cancel.")

	def _rename_annotation_label(self):
		label_name = self._current_label_selection()
		if not label_name:
			return
		new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Label", "New label name:", text=label_name)
		if not ok:
			return
		new_name = new_name.strip()
		if not new_name or new_name == label_name:
			return
		if new_name in self.annotation_labels:
			QtWidgets.QMessageBox.warning(self, "Annotations", f"Label '{new_name}' already exists.")
			return
		color_hex = self.annotation_labels.pop(label_name)
		self.annotation_labels[new_name] = color_hex
		self.annotation_shapes[new_name] = self.annotation_shapes.pop(label_name, [])
		self._rebuild_canvas_annotations()
		self._refresh_annotation_list(select_name=new_name)
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Label renamed to '{new_name}'.")

	def _edit_annotation_shape(self):
		selection = self._current_shape_selection()
		if not selection:
			return
		label_name, shape_id = selection
		shapes = self.annotation_shapes.get(label_name, [])
		shape = next((s for s in shapes if s.get("id") == shape_id), None)
		if not shape:
			return
		color_hex = self.annotation_labels.get(label_name, self.accent_color)
		points = shape.get("points", [])
		self._pending_edit_backup = {
			"label": label_name,
			"shape_id": shape_id,
			"points": [tuple(p) for p in points],
		}
		shapes.remove(shape)
		started = self.canvas_orig.edit_annotation(label_name, color_hex, points, shape_id)
		if not started:
			shapes.append(shape)
			self._pending_edit_backup = None
			QtWidgets.QMessageBox.warning(self, "Annotations", "Unable to start edit mode for this shape.")
			return
		self._refresh_shape_table(label_name)
		self._update_annotation_buttons()
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Editing shape '{shape_id}' — drag handles, double-click or press Enter to save.")

	def _delete_annotation_shape(self):
		selection = self._current_shape_selection()
		if not selection:
			return
		label_name, shape_id = selection
		shapes = self.annotation_shapes.get(label_name, [])
		idx = next((i for i, s in enumerate(shapes) if s.get("id") == shape_id), None)
		if idx is None:
			return
		shapes.pop(idx)
		self.canvas_orig.remove_annotation_item(shape_id)
		if self._pending_edit_backup and self._pending_edit_backup.get("shape_id") == shape_id:
			self._pending_edit_backup = None
		self._refresh_shape_table(label_name)
		self._update_annotation_buttons()
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Deleted shape '{shape_id}'.")

	def _on_annotation_committed(self, label_name: str, shape_id: str, points: list[tuple[float, float]], is_edit: bool):
		if not label_name or not shape_id or not points:
			return
		shapes = self.annotation_shapes.setdefault(label_name, [])
		if is_edit:
			updated = False
			for shape in shapes:
				if shape.get("id") == shape_id:
					shape["points"] = points
					updated = True
					break
			if not updated:
				shapes.append({"id": shape_id, "points": points})
		else:
			shapes.append({"id": shape_id, "points": points})
		if is_edit and self._pending_edit_backup and self._pending_edit_backup.get("shape_id") == shape_id:
			self._pending_edit_backup = None
		self._refresh_shape_table(label_name, select_shape_id=shape_id)
		self._update_annotation_buttons()
		if hasattr(self, 'status_label') and self.status_label:
			action = "updated" if is_edit else "saved"
			self.status_label.setText(f"Annotation {action} for '{label_name}'.")

	def _on_annotation_edit_canceled(self, label_name: str, shape_id: str):
		if not self._pending_edit_backup:
			return
		backup = self._pending_edit_backup
		if backup.get("label") != label_name or backup.get("shape_id") != shape_id:
			return
		points = backup.get("points", [])
		self.annotation_shapes.setdefault(label_name, []).append({"id": shape_id, "points": points})
		color_hex = self.annotation_labels.get(label_name, self.accent_color)
		self.canvas_orig.add_annotation_polygon(label_name, shape_id, points, color_hex)
		self._pending_edit_backup = None
		self._refresh_shape_table(label_name, select_shape_id=shape_id)
		self._update_annotation_buttons()
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText("Edit cancelled; original shape restored.")

	def _export_annotations(self):
		if self.base_image is None:
			QtWidgets.QMessageBox.information(self, "Annotations", "Load an image before exporting.")
			return
		if not self._has_any_annotations():
			QtWidgets.QMessageBox.information(self, "Annotations", "Create at least one polygon before exporting.")
			return
		default_dir = os.path.dirname(self.current_path) if self.current_path else os.getcwd()
		base = os.path.splitext(os.path.basename(self.current_path or "annotations"))[0]
		default_name = os.path.join(default_dir, f"{base}_labels.json")
		fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Annotations", default_name, "JSON Files (*.json)")
		if not fname:
			return
		if not fname.lower().endswith(".json"):
			fname += ".json"
		payload = self._build_labelme_payload()
		try:
			with open(fname, "w", encoding="utf-8") as handle:
				json.dump(payload, handle, indent=2)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Export Failed", f"Failed to write annotation file:\n{exc}")
			return
		attachments = []
		warnings = []
		mask = self._render_annotation_mask()
		if mask is not None:
			mask_path = os.path.splitext(fname)[0] + "_mask.png"
			try:
				cv2.imwrite(mask_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
			except Exception as exc:
				warnings.append(f"mask export failed: {exc}")
			else:
				attachments.append(os.path.basename(mask_path))
		overlay = self._render_annotation_overlay()
		if overlay is not None:
			overlay_path = os.path.splitext(fname)[0] + "_overlay.png"
			try:
				cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
			except Exception as exc:
				warnings.append(f"overlay export failed: {exc}")
			else:
				attachments.append(os.path.basename(overlay_path))
		status_suffix = ""
		if attachments:
			status_suffix += f" (+ {', '.join(attachments)})"
		if warnings:
			status_suffix += f" (warnings: {'; '.join(warnings)})"
		if hasattr(self, 'status_label') and self.status_label:
			self.status_label.setText(f"Annotations exported: {os.path.basename(fname)}{status_suffix}")

	def _build_labelme_payload(self) -> dict:
		if self.base_image is None:
			return {}
		height, width = self.base_image.shape[:2]
		shapes = []
		for label_name, polygons in self.annotation_shapes.items():
			color_hex = self.annotation_labels.get(label_name, "#ff0000")
			for shape in polygons:
				pts = shape.get("points", [])
				if len(pts) < 3:
					continue
				shape_entry = {
					"label": label_name,
					"points": [[float(x), float(y)] for (x, y) in pts],
					"shape_type": "polygon",
					"fill_color": color_hex,
					"line_color": color_hex,
				}
				shapes.append(shape_entry)
		payload = {
			"version": "brainseg-annotations-1.0",
			"flags": {},
			"imagePath": os.path.basename(self.current_path) if self.current_path else None,
			"imageHeight": int(height),
			"imageWidth": int(width),
			"shapes": shapes,
		}
		return payload

	def _render_annotation_mask(self) -> Optional[np.ndarray]:
		if self.base_image is None:
			return None
		height, width = self.base_image.shape[:2]
		mask_img = np.zeros((height, width, 3), dtype=np.uint8)
		any_poly = False
		for label_name, polygons in self.annotation_shapes.items():
			color_hex = self.annotation_labels.get(label_name)
			if not color_hex:
				continue
			r, g, b = self._hex_to_rgb(color_hex)
			for shape in polygons:
				pts = shape.get("points", [])
				if len(pts) < 3:
					continue
				poly = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
				if poly.size == 0:
					continue
				cv2.fillPoly(mask_img, [poly], (r, g, b))
				any_poly = True
		return mask_img if any_poly else None

	def _render_annotation_overlay(self) -> Optional[np.ndarray]:
		if self.base_image is None:
			return None
		base_rgb = self.base_image
		if base_rgb.ndim == 2:
			base_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_GRAY2RGB)
		base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
		overlay = base_bgr.copy()
		any_poly = False
		height, width = base_rgb.shape[:2]
		for label_name, polygons in self.annotation_shapes.items():
			color_hex = self.annotation_labels.get(label_name)
			if not color_hex:
				continue
			r, g, b = self._hex_to_rgb(color_hex)
			bgr_color = (b, g, r)
			for shape in polygons:
				pts = shape.get("points", [])
				if len(pts) < 3:
					continue
				poly = np.array([(int(round(x)), int(round(y))) for x, y in pts], dtype=np.int32)
				if poly.size == 0:
					continue
				cv2.fillPoly(overlay, [poly], bgr_color)
				cv2.polylines(overlay, [poly], True, bgr_color, 2, cv2.LINE_AA)
				any_poly = True
		if not any_poly:
			return None
		blended = cv2.addWeighted(overlay, 0.35, base_bgr, 0.65, 0)
		return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

	def _reset_annotations(self, clear_labels: bool = False):
		self.canvas_orig.clear_annotations()
		if clear_labels:
			self.annotation_labels.clear()
		self.annotation_shapes = {label: [] for label in self.annotation_labels}
		self._pending_edit_backup = None
		self._refresh_annotation_list()

	def _rebuild_canvas_annotations(self):
		self.canvas_orig.clear_annotations()
		for label_name, shapes in self.annotation_shapes.items():
			color_hex = self.annotation_labels.get(label_name, self.accent_color)
			for shape in shapes:
				points = shape.get("points", [])
				if len(points) < 3:
					continue
				shape_id = shape.get("id") or str(uuid.uuid4())
				shape["id"] = shape_id
				self.canvas_orig.add_annotation_polygon(label_name, shape_id, points, color_hex)

	def _update_annotation_buttons(self):
		label_name = self._current_label_selection()
		has_label = label_name is not None
		has_image = self.base_image is not None
		shape_selection = self._current_shape_selection()
		if self.btn_remove_label is not None:
			self.btn_remove_label.setEnabled(has_label)
		if self.btn_rename_label is not None:
			self.btn_rename_label.setEnabled(has_label)
		if self.btn_start_polygon is not None:
			self.btn_start_polygon.setEnabled(has_label and has_image)
		if self.btn_edit_shape is not None:
			self.btn_edit_shape.setEnabled(shape_selection is not None and has_image)
		if self.btn_delete_shape is not None:
			self.btn_delete_shape.setEnabled(shape_selection is not None)
		if self.btn_export_labels is not None:
			self.btn_export_labels.setEnabled(self._has_any_annotations() and has_image)

	def _has_any_annotations(self) -> bool:
		return any(len(shapes) > 0 for shapes in self.annotation_shapes.values()) if self.annotation_shapes else False

	def _apply_image_adjustments(self, img_rgb: np.ndarray) -> np.ndarray:
		"""Apply brightness and contrast adjustments to an RGB uint8 image."""
		if img_rgb is None:
			return None
		if self.brightness_value == 0 and self.contrast_value == 0:
			return img_rgb
		img = img_rgb.astype(np.float32)
		if self.contrast_value != 0:
			factor = 1.0 + (self.contrast_value / 100.0)
			img = (img - 127.5) * factor + 127.5
		if self.brightness_value != 0:
			img += float(self.brightness_value)
		img = np.clip(img, 0, 255)
		return img.astype(np.uint8)

	def _get_adjusted_image(self) -> np.ndarray:
		"""Return image from base_image with current display adjustments."""
		if self.base_image is None:
			return None
		return self._apply_image_adjustments(self.base_image)

	def _apply_intensity_transform(self, img: Optional[np.ndarray], params: Optional[dict] = None) -> Optional[np.ndarray]:
		"""Apply window/level, gamma, curve, and colormap for display."""
		if img is None:
			return None
		img_u8 = img.astype(np.uint8)
		try:
			gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
		except Exception:
			gray = img_u8[..., 0]
		state = {
			"center": self.wl_center,
			"width": self.wl_width,
			"gamma": self.gamma,
			"colormap": self.colormap,
			"curve_points": self.curve_points,
			"enabled": self.intensity_enabled,
		}
		if params:
			state.update({k: v for k, v in params.items() if v is not None})
		if not state.get("enabled", False):
			return img
		c = float(state.get("center", 128))
		w = float(max(1.0, state.get("width", 256)))
		low = c - (w / 2.0)
		high = c + (w / 2.0)
		norm = (gray.astype(np.float32) - low) / (high - low)
		norm = np.clip(norm, 0.0, 1.0)
		mapped = (norm * 255.0).astype(np.float32)
		gamma_value = state.get("gamma", 1.0)
		if gamma_value is None or gamma_value <= 0:
			gamma_value = 1.0
		mapped = 255.0 * np.power(mapped / 255.0, 1.0 / float(gamma_value))
		mapped_u8 = np.clip(mapped, 0, 255).astype(np.uint8)
		curve_points = state.get("curve_points")
		if curve_points:
			try:
				lut = CurveEditor.build_lut(curve_points)
				mapped_u8 = lut[mapped_u8]
			except Exception:
				pass
		colormap = state.get("colormap", "Gray") or "Gray"
		if colormap == 'Gray':
			rgb = np.stack([mapped_u8] * 3, axis=-1)
		elif colormap == 'Jet':
			rgb = cv2.applyColorMap(mapped_u8, cv2.COLORMAP_JET)
			rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		elif colormap == 'Hot':
			rgb = cv2.applyColorMap(mapped_u8, cv2.COLORMAP_HOT)
			rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		else:
			rgb = np.stack([mapped_u8] * 3, axis=-1)
		return rgb

	def _refresh_original_display(self):
		"""Refresh the Original view according to current brightness."""
		if self.base_image is None:
			self.canvas_orig.clear_image()
			return
		adjusted = self._get_adjusted_image()
		# apply intensity transform for display (window/level/gamma/colormap)
		display_img = self._apply_intensity_transform(adjusted)
		if self.canvas_orig.has_image():
			self.canvas_orig.update_image_np(display_img)
		else:
			self.canvas_orig.set_image_np(display_img)

	def render_intensity_preview(self, params: Optional[dict] = None) -> Optional[np.ndarray]:
		adjusted = self._get_adjusted_image()
		if adjusted is None:
			return None
		return self._apply_intensity_transform(adjusted, params=params)

	def _on_brightness_changed(self, value: int):
		self.brightness_value = int(value)
		if hasattr(self, 'brightness_label'):
			self.brightness_label.setText(str(self.brightness_value))
		adjusted = self._get_adjusted_image()
		if self.apply_wl_to_model and self.intensity_enabled:
			self.current_image = self._apply_intensity_transform(adjusted)
		else:
			self.current_image = adjusted
		self._refresh_original_display()
		if self.settings_window is not None:
			self.settings_window.update_brightness_display(self.brightness_value)

	def _on_contrast_changed(self, value: int):
		self.contrast_value = int(value)
		if hasattr(self, 'contrast_label'):
			self.contrast_label.setText(str(self.contrast_value))
		adjusted = self._get_adjusted_image()
		if self.apply_wl_to_model and self.intensity_enabled:
			self.current_image = self._apply_intensity_transform(adjusted)
		else:
			self.current_image = adjusted
		self._refresh_original_display()
		if self.settings_window is not None:
			self.settings_window.update_contrast_display(self.contrast_value)
	def action_select_model(self):
		from .model import load_model, set_model_path, _model_singleton
		fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File", os.getcwd(), "Model Files (*.pth)")
		if not fname:
			self.status_label.setText("Model selection cancelled.")
			return
		# Try to validate the selected .pth immediately and show clear errors instead of crashing. (-->error handling)
		try:
			self.status_label.setText("Validating model file...")
			QtWidgets.QApplication.processEvents()
			model = load_model(fname)
			# Store into singleton so subsequent calls reuse the loaded model
			_model_singleton["model"] = model
			set_model_path(fname)
			self.status_label.setText(f"Model selected: {os.path.basename(fname)}")
		except Exception as exc:
			# Showing a dialog with the detailed error message
			tb = traceback.format_exc()
			QtWidgets.QMessageBox.critical(self, "Model Load Error",
				f"Failed to load the selected model file:\n\n{exc}\n\nTraceback:\n{tb}")
			self.status_label.setText("Model load failed. See dialog for details.")


	def set_intensity_params(self, center: int, width: int, gamma: float, colormap: str, apply_to_model: bool, curve_points=None, enabled=True):
		self.wl_center = int(center)
		self.wl_width = int(width)
		self.gamma = float(gamma)
		self.colormap = colormap
		self.apply_wl_to_model = bool(apply_to_model)
		self.intensity_enabled = bool(enabled)
		if curve_points is not None:
			self.curve_points = self._sanitize_curve_points(curve_points)
		adjusted = self._get_adjusted_image()
		if self.apply_wl_to_model and self.intensity_enabled:
			self.current_image = self._apply_intensity_transform(adjusted)
		else:
			self.current_image = adjusted
		self._refresh_original_display()
		self._sync_intensity_summary()

	def _sanitize_curve_points(self, points):
		if not points:
			return [(0.0, 0.0), (1.0, 1.0)]
		sanitized = []
		for pair in points:
			try:
				x_val, y_val = pair
			except (TypeError, ValueError):
				continue
			try:
				x_clamped = max(0.0, min(1.0, float(x_val)))
				y_clamped = max(0.0, min(1.0, float(y_val)))
			except (TypeError, ValueError):
				continue
			sanitized.append((x_clamped, y_clamped))
		if len(sanitized) < 2:
			return [(0.0, 0.0), (1.0, 1.0)]
		sanitized.sort(key=lambda p: p[0])
		sanitized[0] = (0.0, sanitized[0][1])
		sanitized[-1] = (1.0, sanitized[-1][1])
		return sanitized

	def set_view_background(self, view_key: str, color_hex: str | None, persist: bool = True):
		"""Set background color for one of the three views. Pass None to reset to default."""
		base_color = color_hex or self.default_view_bg_color
		qcolor = QtGui.QColor(base_color)
		brush = QtGui.QBrush(qcolor)
		if view_key == 'original':
			self.view_bg_original = color_hex or None
			self.canvas_orig.setBackgroundBrush(brush)
		elif view_key == 'mask':
			self.view_bg_mask = color_hex or None
			self.canvas_mask.setBackgroundBrush(brush)
		elif view_key == 'highlight':
			self.view_bg_highlight = color_hex or None
			self.canvas_high.setBackgroundBrush(brush)
		# Keep SettingsWindow display in sync
		if self.settings_window is not None and hasattr(self.settings_window, 'update_view_bg_display'):
			self.settings_window.update_view_bg_display(view_key, color_hex)
		if persist:
			self._save_user_preferences()

	def _sync_intensity_summary(self):
		if self.settings_window is None:
			return
		self.settings_window.update_intensity_summary(
			self.wl_center,
			self.wl_width,
			self.gamma,
			self.colormap,
			self.apply_wl_to_model,
			self.intensity_enabled,
		)

	def open_intensity_palette(self):
		dialog = IntensityPaletteDialog(self)
		if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dialog.result_state:
			return
		state = dialog.result_state
		self.set_intensity_params(
			state.get("center", self.wl_center),
			state.get("width", self.wl_width),
			state.get("gamma", self.gamma),
			state.get("colormap", self.colormap),
			state.get("apply_to_model", self.apply_wl_to_model),
			curve_points=state.get("curve_points", self.curve_points),
			enabled=state.get("enabled", self.intensity_enabled),
		)

	def _download_medsam_models(self):
		"""Download Medical SAM ONNX encoder+decoder from Hugging Face and load them."""
		try:
			from huggingface_hub import hf_hub_download
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Dependency Missing", f"huggingface_hub is required to download models: {exc}")
			return
		cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'brainseg', 'models')
		os.makedirs(cache_dir, exist_ok=True)
		try:
			enc_path = hf_hub_download(MEDSAM_REPO_ID, MEDSAM_ENCODER_FILENAME, cache_dir=cache_dir)
			dec_path = hf_hub_download(MEDSAM_REPO_ID, MEDSAM_DECODER_FILENAME, cache_dir=cache_dir)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Download Failed", f"Failed to download Medical SAM models: {exc}")
			return
		# Try to create sessions and register them
		try:
			enc_sess = self._create_onnx_session(enc_path)
			dec_sess = self._create_onnx_session(dec_path)
		except Exception as exc:
			QtWidgets.QMessageBox.critical(self, "Load Failed", f"Downloaded models could not be loaded: {exc}")
			return
		self.onnx_encoder_session = enc_sess
		self.onnx_encoder_input = enc_sess.get_inputs()[0].name if enc_sess.get_inputs() else None
		self.onnx_encoder_outputs = [o.name for o in enc_sess.get_outputs()]
		self.onnx_encoder_path = enc_path
		self.onnx_decoder_session = dec_sess
		self.onnx_decoder_input_names = [i.name for i in dec_sess.get_inputs()]
		self.onnx_decoder_output_names = [o.name for o in dec_sess.get_outputs()]
		self.onnx_decoder_path = dec_path
		self.chk_onnx.setChecked(True)
		self._update_onnx_status_labels()
		QtWidgets.QMessageBox.information(self, "Medical SAM", f"Medical SAM encoder+decoder downloaded and loaded.")
	def action_load_ground_truth(self):
		file_filter = "Masks (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.npy)"
		start_dir = os.path.dirname(self.ground_truth_path) if self.ground_truth_path else (
			os.path.dirname(self.current_path) if self.current_path else os.getcwd()
		)
		fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Ground Truth Mask", start_dir, file_filter)
		if not fname:
			return
		try:
			if fname.lower().endswith(".npy"):
				mask = np.load(fname)
			else:
				mask = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
				if mask is None:
					raise ValueError("Unable to read mask file")
				if mask.ndim == 3:
					mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		except Exception as exc:
			self.status_label.setText(f"Failed to load ground truth: {exc}")
			return
		if self.base_image is not None and mask.shape[:2] != self.base_image.shape[:2]:
			h, w = self.base_image.shape[:2]
			mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
		mask_bin = (mask > 0).astype(np.uint8)
		self.ground_truth_mask = mask_bin
		self.ground_truth_path = fname
		self.label_ground_truth.setText(f"Ground truth: {os.path.basename(fname)}")
		# A small thumbnail for the dock
		try:
			thumb_rgb = None
			if mask_bin.ndim == 2:
				thumb_rgb = np.stack([mask_bin * 255] * 3, axis=-1).astype(np.uint8)
			else:
				# Converting single-channel to 3-channel
				thumb_rgb = mask_bin.copy()
			pm = numpy_to_qpixmap(thumb_rgb)
			pm = pm.scaled(self.ground_truth_thumb.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
			self.ground_truth_thumb.setPixmap(pm)
			self.ground_truth_thumb.setStyleSheet("border: 1px solid #000000; background: transparent;")
		except Exception:
			# fallback:---> clear thumb
			self.ground_truth_thumb.clear()
			self.ground_truth_thumb.setStyleSheet("border: 1px solid #000000; background: transparent;")
		self.status_label.setText("Ground truth mask loaded.")
	def _build_menubar(self):
		menubar = self.menuBar()
		file_menu = menubar.addMenu("&File")
		act_open = QtGui.QAction("Open Image...", self)
		act_open.setShortcut("Ctrl+Alt+O")
		act_open.triggered.connect(self.action_open_image)
		act_run = QtGui.QAction("Run Segmentation", self)
		act_run.setShortcut("Ctrl+Alt+R")
		act_run.triggered.connect(self.action_run_segmentation)
		act_save_mask = QtGui.QAction("Save Mask...", self)
		act_save_mask.setShortcut("Ctrl+Alt+S")
		act_save_mask.triggered.connect(self.action_save_mask)
		act_save_high = QtGui.QAction("Save Highlight...", self)
		act_save_high.triggered.connect(self.action_save_highlight)
		act_exit = QtGui.QAction("Exit", self)
		act_exit.setShortcut("Ctrl+Alt+Q")
		act_exit.triggered.connect(self.close)
		file_menu.addAction(act_open)
		file_menu.addAction(act_run)
		file_menu.addSeparator()
		file_menu.addAction(act_save_mask)
		file_menu.addAction(act_save_high)
		file_menu.addSeparator()
		file_menu.addAction(act_exit)
		view_menu = menubar.addMenu("&View")
		act_fit = QtGui.QAction("Fit to Window", self)
		act_fit.setShortcut("Ctrl+Alt+F")
		act_fit.triggered.connect(self.fit_all)
		act_1x = QtGui.QAction("Zoom 1:1", self)
		act_1x.setShortcut("Ctrl+Alt+1")
		act_1x.triggered.connect(self.one_to_one_all)
		view_menu.addAction(act_fit)
		view_menu.addAction(act_1x)

		stats_menu = menubar.addMenu("&Statistics")
		act_stats = QtGui.QAction("Show Statistics", self)
		act_stats.triggered.connect(self.show_statistics_window)
		stats_menu.addAction(act_stats)
		# Another Theme switch action directly on the menubar
		# self.menu_theme_action = QtGui.QAction(self._get_theme_icon(), "Switch Theme", self)
		# self.menu_theme_action.setToolTip("Switch between day/night mode")
		# self.menu_theme_action.triggered.connect(self._toggle_theme)
		# Help menu and then insert Theme
		help_menu = menubar.addMenu("&Help")
		# menubar.insertAction(help_menu.menuAction(), self.menu_theme_action)
		act_help = QtGui.QAction("Shortcuts", self)
		act_help.triggered.connect(self._show_shortcuts)
		help_menu.addAction(act_help)

		settings_menu = menubar.addMenu("&Settings")
		act_settings = QtGui.QAction("Open Settings", self)
		act_settings.setShortcut("Ctrl+Alt+C")
		act_settings.triggered.connect(self._show_settings)
		settings_menu.addAction(act_settings)
	def _build_toolbar(self):
		tb = QtWidgets.QToolBar("Main")
		tb.setIconSize(QtCore.QSize(24, 24))
		# Icon beside text for toolbar buttons
		tb.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
		self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
		def action(text, slot, shortcut=None, tip=None, icon=None):
			a = QtGui.QAction(icon or QtGui.QIcon(), text, self)
			if shortcut:
				a.setShortcut(shortcut)
			if tip:
				a.setToolTip(tip)
				a.setStatusTip(tip)
			a.triggered.connect(slot)
			tb.addAction(a)
			return a
		action("Open", self.action_open_image, "Ctrl+Alt+O", "Open image")
		action("Run", self.action_run_segmentation, "Ctrl+Alt+R", "Run segmentation")
		tb.addSeparator()
		action("Fit", self.fit_all, "Ctrl+Alt+F", "Fit all views")
		action("1:1", self.one_to_one_all, "Ctrl+Alt+1", "Reset zoom to 1:1")
		tb.addSeparator()
		action("Save Mask", self.action_save_mask, "Ctrl+Alt+S", "Save mask image")
		action("Save Highlight", self.action_save_highlight, None, "Save highlighted image")

		# Theme switcher button to the toolbar (with icon and text --> you can change icon (source))
		theme_icon = self._get_theme_icon()
		self.theme_action = QtGui.QAction(theme_icon, "Switch Theme", self)
		self.theme_action.setToolTip("Switch between day/night mode")
		self.theme_action.triggered.connect(self._toggle_theme)
		tb.addSeparator()
		tb.addAction(self.theme_action)
		self.annotation_panel_action = QtGui.QAction("Annotations", self)
		self.annotation_panel_action.setToolTip("Open annotation manager")
		self.annotation_panel_action.triggered.connect(self._show_annotation_window)
		tb.addAction(self.annotation_panel_action)
	def _toggle_theme(self):
		new_theme = "dark" if self.theme == "light" else "light"
		self.set_theme(new_theme)

	def _load_user_preferences(self):
		settings = getattr(self, '_settings', None)
		if settings is None:
			return
		theme = settings.value("appearance/theme", self.theme)
		if theme in ("light", "dark"):
			self.theme = theme
		accent = settings.value("appearance/accent", self.accent_name)
		if accent in ACCENT_COLORS:
			self.accent_name = accent
			self.accent_color = ACCENT_COLORS[accent]
		theme_color = settings.value("appearance/theme_color", "")
		self.theme_color = theme_color if theme_color else None
		for view_key in ("original", "mask", "highlight"):
			stored = settings.value(f"appearance/view_bg_{view_key}", "")
			if stored:
				self.set_view_background(view_key, stored, persist=False)

	def _save_user_preferences(self):
		settings = getattr(self, '_settings', None)
		if settings is None:
			return
		settings.setValue("appearance/theme", self.theme)
		settings.setValue("appearance/accent", self.accent_name)
		settings.setValue("appearance/theme_color", self.theme_color or "")
		settings.setValue("appearance/view_bg_original", self.view_bg_original or "")
		settings.setValue("appearance/view_bg_mask", self.view_bg_mask or "")
		settings.setValue("appearance/view_bg_highlight", self.view_bg_highlight or "")
		settings.sync()

	def set_theme(self, theme_name: str, persist: bool = True):
		if theme_name not in ("light", "dark"):
			return
		self.theme = theme_name
		self._base_theme_stylesheet = LIGHT_THEME if theme_name == "light" else DARK_THEME
		self._apply_theme_stylesheet()
		if hasattr(self, 'theme_action'):
			self.theme_action.setIcon(self._get_theme_icon())
		# self._update_footer_label_style()
		self._apply_accent_palette()
		if self.annotation_window is not None:
			self.annotation_window.apply_theme(theme_name)
		if self.settings_window is not None:
			self.settings_window.update_theme_display(theme_name)
			self.settings_window.apply_accent(self.accent_color, self.theme)
			self.settings_window.update_theme_color_display(self.theme_color)
		if persist:
			self._save_user_preferences()

	def set_accent(self, accent_name: str, persist: bool = True):
		if accent_name not in ACCENT_COLORS:
			accent_name = "Azure"
		self.accent_name = accent_name
		self.accent_color = ACCENT_COLORS[accent_name]
		self._apply_accent_palette()
		if self.settings_window is not None:
			self.settings_window.update_accent_display(accent_name)
		if persist:
			self._save_user_preferences()

	def set_theme_color(self, color_hex: Optional[str], persist: bool = True):
		if color_hex:
			if not isinstance(color_hex, str) or not color_hex.startswith("#") or len(color_hex) != 7:
				return
			self.theme_color = color_hex.lower()
		else:
			self.theme_color = None
		self._apply_theme_stylesheet()
		self._apply_accent_palette()
		if self.settings_window is not None:
			self.settings_window.update_theme_color_display(self.theme_color)
		if persist:
			self._save_user_preferences()

	def _apply_theme_stylesheet(self):
		stylesheet = getattr(self, '_base_theme_stylesheet', LIGHT_THEME)
		if self.theme_color:
			text_color = self._ideal_text_color(self.theme_color)
			panel = self.theme_color
			dock = self._shade_color(self.theme_color, 0.94)
			title_bg = self._shade_color(self.theme_color, 0.9)
			button_bg = self._shade_color(self.theme_color, 0.92)
			button_hover = self._shade_color(self.theme_color, 1.05)
			button_border = self._shade_color(self.theme_color, 0.78)
			hint_color = self._shade_color(text_color, 1.35 if text_color == "#111111" else 0.7)
			stylesheet += f"""
QMainWindow, QWidget {{
	background-color: {panel};
	color: {text_color};
}}
QDockWidget {{
	background-color: {dock};
}}
QDockWidget::title {{
	background-color: {title_bg};
	color: {text_color};
}}
QPushButton {{
	background-color: {button_bg};
	color: {text_color};
	border: 1px solid {button_border};
}}
QPushButton:hover {{
	background-color: {button_hover};
	border-color: {button_border};
}}
QLabel#hint {{
	color: {hint_color};
}}
QToolBar {{
	background-color: {panel};
	border-bottom: 1px solid {button_border};
}}
QStatusBar {{
	background-color: {panel};
	border-top: 1px solid {button_border};
}}
"""
		self.setStyleSheet(stylesheet)

	def _apply_accent_palette(self):
		app = QtWidgets.QApplication.instance()
		if app:
			palette = app.palette()
			palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(self.accent_color))
			palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(self.accent_color))
			palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ffffff"))
			app.setPalette(palette)
		bar = getattr(self, 'segmentation_progress', None)
		if bar is not None:
			track = "rgba(255, 255, 255, 0.12)" if self.theme == "dark" else "#f3f4f6"
			border = "#444a57" if self.theme == "dark" else "#d0d0d0"
			bar.setStyleSheet(
				f"QProgressBar {{ border: 1px solid {border}; border-radius: 4px; background: {track}; padding: 0 2px; }}"
				f"QProgressBar::chunk {{ background-color: {self.accent_color}; border-radius: 4px; }}"
			)
		self._style_slider(self.brightness_slider)
		self._style_slider(self.contrast_slider)
		if self.settings_window is not None:
			self.settings_window.apply_accent(self.accent_color, self.theme)

	def _style_slider(self, slider):
		if slider is None:
			return
		groove = "#3f454f" if self.theme == "dark" else "#d6dbe3"
		qcolor = QtGui.QColor(self.accent_color)
		darker = qcolor.darker(120).name()
		slider.setStyleSheet(
			f"QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {groove}; }}"
			f"QSlider::handle:horizontal {{ background: {self.accent_color}; border: 1px solid {darker}; width: 14px; margin: -4px 0; border-radius: 7px; }}"
		)

	def _ideal_text_color(self, hex_color: str) -> str:
		r, g, b = self._hex_to_rgb(hex_color)
		luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
		return "#111111" if luminance > 186 else "#f5f5f5"

	def _shade_color(self, hex_color: str, factor: float) -> str:
		r, g, b = self._hex_to_rgb(hex_color)
		r = self._clamp(int(r * factor))
		g = self._clamp(int(g * factor))
		b = self._clamp(int(b * factor))
		return f"#{r:02x}{g:02x}{b:02x}"

	@staticmethod
	def _hex_to_rgb(hex_color: str):
		hex_color = hex_color.lstrip('#')
		return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

	@staticmethod
	def _clamp(value: int, minimum: int = 0, maximum: int = 255) -> int:
		return max(minimum, min(maximum, value))

	# def _update_footer_label_style(self):
	# 	if not hasattr(self, 'footer_label') or self.footer_label is None:
	# 		return
	# 	if self.theme == "dark":
	# 		self.footer_label.setStyleSheet("color: #ffffff; font-size: 12px;")
	# 	else:
	# 		self.footer_label.setStyleSheet("color: #111111; font-size: 12px;")
	# def _build_footer(self):
	# 	self.statusBar().setSizeGripEnabled(False)
	# 	container = QtWidgets.QWidget()
	# 	h = QtWidgets.QHBoxLayout(container)
	# 	h.setContentsMargins(0, 0, 0, 0)
	# 	h.addStretch()
		# self.footer_label = QtWidgets.QLabel("Developed by Md. Rasel Mandol — Smart Systems & Connectivity Lab, NIT Meghalaya")
		# self.footer_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		# h.addWidget(self.footer_label)
		# h.addStretch()
		# self.statusBar().addPermanentWidget(container, 1)
		# self._update_footer_label_style()
	def action_open_image(self):
		file_filter = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
		start_dir = os.path.dirname(self.current_path) if self.current_path else os.getcwd()
		fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", start_dir, file_filter)
		if not fname:
			return
		try:
			self.status_label.setText("Loading image...")
			QtWidgets.QApplication.processEvents()
			img_rgb = pil_or_cv_to_rgb_np(fname)
			self.base_image = img_rgb
			adjusted = self._get_adjusted_image()
			if self.apply_wl_to_model and self.intensity_enabled:
				self.current_image = self._apply_intensity_transform(adjusted)
			else:
				self.current_image = adjusted
			self.current_mask = None
			self.current_highlight = None
			self.current_path = fname
			self.ground_truth_mask = None
			self.ground_truth_path = None
			self._reset_annotations(clear_labels=False)
			self.label_filename.setText(os.path.basename(fname))
			self.label_ground_truth.setText("Ground truth: not loaded")
			# Clearing thumbnail when loading a new image
			self.ground_truth_thumb.clear()
			self.ground_truth_thumb.setStyleSheet("border: 1px solid #000000; background: transparent;")
			# Showing original with current brightness adjustment
			self.canvas_orig.clear_image()
			self._refresh_original_display()
			self.canvas_mask.clear_image()
			self.canvas_high.clear_image()
			self._update_annotation_buttons()
			self.status_label.setText("Image loaded. Press Run or Ctrl+R.")
		except Exception as e:
			self.status_label.setText(f"Error: {e}")
	def action_run_segmentation(self):
		if self.current_image is None:
			self.status_label.setText("No image loaded.")
			return
		if not self._ensure_model_ready():
			return

		process = psutil.Process()
		# indeterminate progress while running segmentation
		if hasattr(self, 'segmentation_progress'):
			self.segmentation_progress.setRange(0, 0)  # indeterminate
			QtWidgets.QApplication.processEvents()
		mem_before = process.memory_info().rss / (1024 * 1024)
		t0 = time.perf_counter()
		try:
			mask_up, highlighted = run_inference_on_image(self.current_image)
			t1 = time.perf_counter()
		except Exception as exc:
			# Ensure progress bar returns to normal and display the error details.
			t1 = time.perf_counter()
			if hasattr(self, 'segmentation_progress'):
				self.segmentation_progress.setRange(0, 100)
				self.segmentation_progress.setValue(0)
			tb = traceback.format_exc()
			QtWidgets.QMessageBox.critical(self, "Segmentation Error",
				f"An error occurred while running segmentation:\n\n{exc}\n\nTraceback:\n{tb}")
			self.status_label.setText("Segmentation failed. See dialog for details.")
			return
		mem_after = process.memory_info().rss / (1024 * 1024)
		latency = t1 - t0
		memory_peak = max(mem_before, mem_after)
		quality_record = None
		if self.ground_truth_mask is not None:
			quality_record = statistics_tracker.record_quality(
				mask_up,
				self.ground_truth_mask,
				os.path.basename(self.current_path) if self.current_path else None,
			)
		dice_accuracy = quality_record.aggregate["dice"] if quality_record else None
		if dice_accuracy is not None and np.isnan(dice_accuracy):
			dice_accuracy = None
		statistics_tracker.record_segmentation(latency, memory_peak, accuracy=dice_accuracy)
		self.current_mask = mask_up
		self.current_highlight = highlighted
		if self.current_mask.ndim == 2:
			mask_rgb = np.stack([self.current_mask]*3, axis=-1)
		else:
			mask_rgb = self.current_mask
		self.canvas_mask.set_image_np(mask_rgb)
		self.canvas_high.set_image_np(self.current_highlight)
		# marking progress complete
		if hasattr(self, 'segmentation_progress'):
			self.segmentation_progress.setRange(0, 100)
			self.segmentation_progress.setValue(100)
			QtCore.QTimer.singleShot(800, lambda: self.segmentation_progress.setValue(0))

		if quality_record and quality_record.aggregate["dice"] is not None:
			self.status_label.setText(
				f"Segmentation done in {latency:.3f}s · Dice {quality_record.aggregate['dice']:.3f}"
			)
		else:
			self.status_label.setText(f"Segmentation done in {latency:.3f}s")
	def _on_progress(self, text: str):
		self.status_label.setText(text)
		QtWidgets.QApplication.processEvents()
	def _on_inference_finished(self, result):
		mask_up, highlighted = result
		if mask_up is None or highlighted is None:
			return
		self.current_mask = mask_up
		self.current_highlight = highlighted
		if self.current_mask.ndim == 2:
			mask_rgb = np.stack([self.current_mask]*3, axis=-1)
		else:
			mask_rgb = self.current_mask
		self.canvas_mask.set_image_np(mask_rgb)
		self.canvas_high.set_image_np(self.current_highlight)
		# marking progress complete for worker-based inference
		if hasattr(self, 'segmentation_progress'):
			self.segmentation_progress.setRange(0, 100)
			self.segmentation_progress.setValue(100)
			QtCore.QTimer.singleShot(800, lambda: self.segmentation_progress.setValue(0))
		self.status_label.setText("Done.")

	def _ensure_model_ready(self) -> bool:
		try:
			model = get_model()
			return model is not None
		except FileNotFoundError:
			self.status_label.setText("Load a model first.")
		except Exception as exc:
			self.status_label.setText(f"Model load failed: {exc}")
		return False
	def action_save_mask(self):
		if self.current_mask is None:
			self.status_label.setText("No mask to save.")
			return
		default_dir = os.path.dirname(self.current_path) if self.current_path else os.getcwd()
		if self.current_path:
			base = os.path.splitext(os.path.basename(self.current_path))[0]
			default_name = f"{base}_mask.png"
		else:
			default_name = "mask.png"
		fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save mask", os.path.join(default_dir, default_name),
														 "PNG Files (*.png);;TIFF Files (*.tif)")
		if fname:
			cv2.imwrite(fname, self.current_mask)
			self.status_label.setText(f"Mask saved: {fname}")
	def action_save_highlight(self):
		if self.current_highlight is None:
			self.status_label.setText("No highlight to save.")
			return
		default_dir = os.path.dirname(self.current_path) if self.current_path else os.getcwd()
		if self.current_path:
			base = os.path.splitext(os.path.basename(self.current_path))[0]
			default_name = f"{base}_highlight.png"
		else:
			default_name = "highlight.png"
		fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save highlight", os.path.join(default_dir, default_name),
														 "PNG Files (*.png);;TIFF Files (*.tif)")
		if fname:
			cv2.imwrite(fname, cv2.cvtColor(self.current_highlight, cv2.COLOR_RGB2BGR))
			self.status_label.setText(f"Highlight saved: {fname}")
	def fit_all(self):
		self.canvas_orig.fit_to_window()
		self.canvas_mask.fit_to_window()
		self.canvas_high.fit_to_window()
	def one_to_one_all(self):
		self.canvas_orig.zoom_1x()
		self.canvas_mask.zoom_1x()
		self.canvas_high.zoom_1x()
	def _show_shortcuts(self):
		if not hasattr(self, '_help_window'):
			self._help_window = HelpWindow(self)
		self._help_window.show()
		self._help_window.raise_()
		self._help_window.activateWindow()
	def _show_settings(self):
		if self.settings_window is None:
			self.settings_window = SettingsWindow(self)
		else:
			self.settings_window.sync_from_main()
		self.settings_window.show()
		self.settings_window.raise_()
		self.settings_window.activateWindow()

	def _show_annotation_window(self):
		if self.annotation_window is None:
			self.annotation_window = AnnotationWindow(self)
		else:
			self.annotation_window.sync_from_main()
		self.annotation_window.show()
		self.annotation_window.raise_()
		self.annotation_window.activateWindow()

def main():
	app = QtWidgets.QApplication(sys.argv)
	font = app.font()
	font.setPointSize(10)
	app.setFont(font)
	window = SegmentationApp()
	window.show()
	def preload():
		try:
			get_model()
		except Exception:
			pass
	QtCore.QTimer.singleShot(10, preload)
	sys.exit(app.exec())
