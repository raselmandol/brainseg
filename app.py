import sys
import os

import cv2
import numpy as np
from PIL import Image

from PyQt6 import QtCore, QtGui, QtWidgets

import torch
from torchvision import transforms as T
from scipy.ndimage import binary_dilation
import segmentation_models_pytorch as smp

# =============================
# Config
# =============================
MODEL_PATH = "brain_segmentation_model.pth"  # <-- update if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================
# Model loading
# =============================
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
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# Lazy load (so UI opens even if model path wrong; errors reported when used)
_model_singleton = {"model": None}

def get_model():
    if _model_singleton["model"] is None:
        _model_singleton["model"] = load_model(MODEL_PATH)
    return _model_singleton["model"]


# =============================
# Image utilities
# =============================
def numpy_to_qpixmap(img_np: np.ndarray) -> QtGui.QPixmap:
    """Convert HWC uint8 numpy array to QPixmap."""
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
    """Return RGB uint8 numpy from file path or OpenCV array or PIL Image."""
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
            # assume BGR as commonly returned by cv2; convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif isinstance(path_or_array, Image.Image):
        return np.array(path_or_array.convert("RGB"))
    else:
        raise ValueError("Unsupported image type")


# =============================
# Inference pipeline
# =============================
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
    # edge as mask - dilated(mask)
    edges = np.clip(mask_bool - binary_dilation(mask_bool, structure=np.ones((3, 3))).astype(np.uint8), 0, 1)
    highlighted = original_rgb.copy()
    # red edge
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


# =============================
# Graphics Canvas
# =============================
class ImageCanvas(QtWidgets.QGraphicsView):
    """Zoom/Pan canvas."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixitem = None
        self._title = title
        self._current_pixmap = QtGui.QPixmap()

        # Canvas styling: dark neutral background, no frame
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#2f2f2f")))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # Interaction
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing
            | QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._is_space_pressed = False
        self._zoom = 0

        # Title label above the canvas
        self.title_label = QtWidgets.QLabel(self._title)
        self.title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 15px; color: #1a1a1a;")

        # Wrap in a vertical layout container
        self.wrapper = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(self.wrapper)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        v.addWidget(self.title_label)
        v.addWidget(self)

    def container(self) -> QtWidgets.QWidget:
        """Return the widget to place in layouts (title + canvas)."""
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
        # Center the image
        rect = self._pixitem.boundingRect()
        self.centerOn(self._pixitem)
        self._zoom = 0

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self._pixitem is None:
            return
        # Zoom factor per step
        delta = event.angleDelta().y()
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if delta > 0:
            factor = zoom_in_factor
            self._zoom += 1
        else:
            factor = zoom_out_factor
            self._zoom -= 1

        # Prevent huge negative zooms (collapse)
        if self._zoom < -10:
            self._zoom = -10
            return

        self.scale(factor, factor)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Space to pan (hand tool)
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


# =============================
# Worker thread
# =============================
class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(str)


class InferenceWorker(QtCore.QRunnable):
    def __init__(self, img_rgb):
        super().__init__()
        self.img_rgb = img_rgb
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.signals.progress.emit("Running segmentation...")
            mask_up, highlighted = run_inference_on_image(self.img_rgb)
            self.signals.finished.emit((mask_up, highlighted))
        except Exception as e:
            self.signals.progress.emit(f"Error: {e}")
            self.signals.finished.emit((None, None))


# =============================
# Main Window
# =============================
class SegmentationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Abnormality Segmentation")
        self.resize(1100, 550)

        # --- Global style (white chrome, clean)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #ffffff;
                color: #2c3e50;
                font-family: 'Segoe UI', Helvetica, Arial;
                font-size: 14px;
            }
            QDockWidget {
                background: #f7f7f7;
                titlebar-close-icon: url(none);
                titlebar-normal-icon: url(none);
            }
            QDockWidget::title {
                background: #f0f0f0;
                padding: 6px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #fafafa;
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QPushButton:hover { background-color: #e8f0fe; border-color: #b9d0ff; }
            QLabel#hint { color: #666; font-size: 12px; }
            QToolBar {
                background: #ffffff;
                border-bottom: 1px solid #e6e6e6;
                spacing: 6px;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #e6e6e6;
            }
        """)

        # Data
        self.current_image = None    # RGB np
        self.current_mask = None     # uint8 0/255
        self.current_highlight = None  # RGB np
        self.current_path = None

        # --- Central area: three canvases inside a splitter (resizable)
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

        # --- Left dock (controls)
        self._build_left_dock()

        # --- Menu & Toolbar
        self._build_menubar()
        self._build_toolbar()

        # --- Footer (centered)
        self._build_footer()

    # ---------- UI Building ----------

    def _build_left_dock(self):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(10)

        # Buttons
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
        # Fit / Zoom controls
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
        # File info / status
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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File", os.getcwd(), "Model Files (*.pth)")
        if fname:
            global MODEL_PATH
            MODEL_PATH = fname
            _model_singleton["model"] = None
            self.status_label.setText(f"Model selected: {os.path.basename(fname)}")
        else:
            self.status_label.setText("Model selection cancelled.")

    def _build_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        act_open = QtGui.QAction("Open Image...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.action_open_image)

        act_run = QtGui.QAction("Run Segmentation", self)
        act_run.setShortcut("Ctrl+R")
        act_run.triggered.connect(self.action_run_segmentation)

        act_save_mask = QtGui.QAction("Save Mask...", self)
        act_save_mask.setShortcut("Ctrl+S")
        act_save_mask.triggered.connect(self.action_save_mask)

        act_save_high = QtGui.QAction("Save Highlight...", self)
        act_save_high.triggered.connect(self.action_save_highlight)

        act_exit = QtGui.QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
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
        act_fit.setShortcut("F")
        act_fit.triggered.connect(self.fit_all)
        act_1x = QtGui.QAction("Zoom 1:1", self)
        act_1x.setShortcut("1")
        act_1x.triggered.connect(self.one_to_one_all)
        view_menu.addAction(act_fit)
        view_menu.addAction(act_1x)

        help_menu = menubar.addMenu("&Help")
        act_help = QtGui.QAction("Shortcuts", self)
        act_help.triggered.connect(self._show_shortcuts)
        help_menu.addAction(act_help)

    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("Main")
        tb.setIconSize(QtCore.QSize(18, 18))
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

        action("Open", self.action_open_image, "Ctrl+O", "Open image")
        action("Run", self.action_run_segmentation, "Ctrl+R", "Run segmentation")
        tb.addSeparator()
        action("Fit", self.fit_all, "F", "Fit all views")
        action("1:1", self.one_to_one_all, "1", "Reset zoom to 1:1")
        tb.addSeparator()
        action("Save Mask", self.action_save_mask, "Ctrl+S", "Save mask image")
        action("Save Highlight", self.action_save_highlight, None, "Save highlighted image")

    def _build_footer(self):
        self.statusBar().setSizeGripEnabled(False)
        # Centered label inside status bar
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addStretch()
        label = QtWidgets.QLabel("Made by Md. Rasel Mandol — Smart Systems & Connectivity Lab, NIT Meghalaya")
        label.setStyleSheet("color: #666; font-size: 12px;")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        h.addWidget(label)
        h.addStretch()
        self.statusBar().addPermanentWidget(container, 1)

    # ---------- Actions ----------
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
            # error reported via progress already
            return
        self.current_mask = mask_up
        self.current_highlight = highlighted

        # For mask view, convert 0/255 to RGB (white mask on black)
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

    # View helpers
    def fit_all(self):
        self.canvas_orig.fit_to_window()
        self.canvas_mask.fit_to_window()
        self.canvas_high.fit_to_window()

    def one_to_one_all(self):
        self.canvas_orig.zoom_1x()
        self.canvas_mask.zoom_1x()
        self.canvas_high.zoom_1x()

    def _show_shortcuts(self):
        msg = (
            "<b>Shortcuts</b><br>"
            "Ctrl+O — Open Image<br>"
            "Ctrl+R — Run Segmentation<br>"
            "Ctrl+S — Save Mask<br>"
            "F — Fit to Window (all)<br>"
            "1 — Zoom 1:1 (all)<br>"
            "Mouse Wheel — Zoom<br>"
            "Space (hold) — Pan"
        )
        QtWidgets.QMessageBox.information(self, "Shortcuts", msg)


# =============================
# Entry
# =============================
def main():
    app = QtWidgets.QApplication(sys.argv)
    # Default font
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)

    window = SegmentationApp()
    window.show()

    # Warm-load the model in the background so first run is snappy
    def preload():
        try:
            get_model()
        except Exception:
            pass
    QtCore.QTimer.singleShot(10, preload)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

import gradio as gr

def segment_image(image, model_file):
    if not model_file:
        raise ValueError("No model file selected.")
    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    original_image_np, predicted_mask = predict_segmentation(image, model, device)
    highlighted_image, segmented_mask = plot_segmentation_results(
        original_image_np, predicted_mask
    )
    return original_image_np, segmented_mask, highlighted_image

iface = gr.Interface(
    fn=segment_image,
    inputs=[
        gr.Image(
            type="pil",
            label="Upload Brain MRI",
            format="png"
        ),
        gr.File(label="Select Model File (.pth)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Segmented Mask"),
        gr.Image(type="numpy", label="Abnormality Highlighted"),
    ],
    title="Brain Abnormality Segmentation",
    description="Upload a brain MRI image and select a model file (.pth) to get the abnormality segmentation."
)
