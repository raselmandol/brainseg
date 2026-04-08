
from PyQt6 import QtCore
from .model import run_inference_on_image

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
			mask_up, highlighted, confidence_summary = run_inference_on_image(self.img_rgb)
			self.signals.finished.emit((mask_up, highlighted, confidence_summary))
		except Exception as e:
			self.signals.progress.emit(f"Error: {e}")
			self.signals.finished.emit((None, None, None))
