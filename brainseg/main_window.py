import cv2
from PyQt6 import QtCore, QtGui, QtWidgets
import os
import sys
import numpy as np
from .canvas import ImageCanvas
from .image_utils import pil_or_cv_to_rgb_np
from .model import get_model, run_inference_on_image, MODEL_PATH

from .worker import InferenceWorker
from .help_window import HelpWindow
from .theme import LIGHT_THEME, DARK_THEME, get_icon_path

class SegmentationApp(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Brain Abnormality Segmentation")
		self.resize(1100, 550)
		self.theme = "light"
		self.theme_style = 1  # 1 or 2, for icon style
		# Get the directory where this file is located (filename_1.png, _2.png/reverse mode (day/night) icons are stored there)
		module_dir = os.path.dirname(os.path.abspath(__file__))
		self.assets_dir = os.path.join(module_dir, "assets")
		self.setStyleSheet(LIGHT_THEME)
		self.current_image = None
		self.current_mask = None
		self.current_highlight = None
		self.current_path = None
		self.canvas_orig = ImageCanvas("Original")
		self.canvas_mask = ImageCanvas("Segmented Mask")
		self.canvas_high = ImageCanvas("Highlighted Tumor")
		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		splitter.addWidget(self.canvas_orig.container())
		splitter.addWidget(self.canvas_mask.container())
		splitter.addWidget(self.canvas_high.container())
		splitter.setHandleWidth(8)
		splitter.setSizes([1, 1, 1])
		center = QtWidgets.QWidget()
		center_layout = QtWidgets.QVBoxLayout(center)
		center_layout.setContentsMargins(12, 8, 12, 0)
		center_layout.addWidget(splitter)
		self.setCentralWidget(center)
		self._build_left_dock()
		self._build_menubar()
		self._build_toolbar()
		self._build_footer()

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
		btn_select_model = QtWidgets.QPushButton("Select Model File")
		btn_select_model.clicked.connect(self.action_select_model)
		btn_run = QtWidgets.QPushButton("Run Segmentation")
		btn_run.clicked.connect(self.action_run_segmentation)
		btn_save_mask = QtWidgets.QPushButton("Save Mask")
		btn_save_mask.clicked.connect(self.action_save_mask)
		btn_save_high = QtWidgets.QPushButton("Save Highlight")
		btn_save_high.clicked.connect(self.action_save_highlight)
		v.addWidget(btn_open)
		v.addWidget(btn_select_model)
		v.addWidget(btn_run)
		v.addWidget(btn_save_mask)
		v.addWidget(btn_save_high)
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
		self.status_label = QtWidgets.QLabel("Ready")
		self.status_label.setObjectName("hint")
		v.addWidget(self.status_label)
		v.addStretch()
		panel.setMinimumWidth(240)
		dock.setWidget(panel)
		self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
	def action_select_model(self):
		from .model import set_model_path, MODEL_PATH
		fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File", os.getcwd(), "Model Files (*.pth)")
		if fname:
			set_model_path(fname)
			self.status_label.setText(f"Model selected: {os.path.basename(fname)}")
		else:
			self.status_label.setText("Model selection cancelled.")
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
	def _toggle_theme(self):
		if self.theme == "light":
			self.theme = "dark"
			self.setStyleSheet(DARK_THEME)
		else:
			self.theme = "light"
			self.setStyleSheet(LIGHT_THEME)
		# Always updating the icon to match the theme (reverse --> white/night)
		if hasattr(self, 'theme_action'):
			self.theme_action.setIcon(self._get_theme_icon())
		# Update menubar theme action icon as well
		# if hasattr(self, 'menu_theme_action'):
		# 	self.menu_theme_action.setIcon(self._get_theme_icon())
	def _build_footer(self):
		self.statusBar().setSizeGripEnabled(False)
		container = QtWidgets.QWidget()
		h = QtWidgets.QHBoxLayout(container)
		h.setContentsMargins(0, 0, 0, 0)
		h.addStretch()
		label = QtWidgets.QLabel("Made by Md. Rasel Mandol — Smart Systems & Connectivity Lab, NIT Meghalaya")
		label.setStyleSheet("color: #322c94; font-size: 12px;")
		label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
		h.addWidget(label)
		h.addStretch()
		self.statusBar().addPermanentWidget(container, 1)
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
			self.current_image = img_rgb
			self.current_mask = None
			self.current_highlight = None
			self.current_path = fname
			self.label_filename.setText(os.path.basename(fname))
			self.canvas_orig.set_image_np(img_rgb)
			self.canvas_mask.clear_image()
			self.canvas_high.clear_image()
			self.status_label.setText("Image loaded. Press Run or Ctrl+R.")
		except Exception as e:
			self.status_label.setText(f"Error: {e}")
	def action_run_segmentation(self):
		if self.current_image is None:
			self.status_label.setText("No image loaded.")
			return
		worker = InferenceWorker(self.current_image)
		worker.signals.progress.connect(self._on_progress)
		worker.signals.finished.connect(self._on_inference_finished)
		QtCore.QThreadPool.globalInstance().start(worker)
		self.status_label.setText("Queued segmentation...")
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
		self.status_label.setText("Done.")
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
