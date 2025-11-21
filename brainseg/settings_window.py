from PyQt6 import QtWidgets, QtCore


class SettingsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.setMinimumSize(420, 320)
        self.setObjectName("SettingsWindow")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        adjustments_box = QtWidgets.QGroupBox("Image Adjustments")
        form = QtWidgets.QFormLayout(adjustments_box)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.brightness_slider, self.brightness_value_label = self._create_slider()
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        form.addRow("Brightness", self._wrap_slider(self.brightness_slider, self.brightness_value_label))

        self.contrast_slider, self.contrast_value_label = self._create_slider()
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        form.addRow("Contrast", self._wrap_slider(self.contrast_slider, self.contrast_value_label))

        layout.addWidget(adjustments_box)
        layout.addStretch(1)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(96)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.sync_from_main()

    def _create_slider(self):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        value_label = QtWidgets.QLabel("0")
        value_label.setFixedWidth(32)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        return slider, value_label

    @staticmethod
    def _wrap_slider(slider, value_label):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 1)
        row.addSpacing(8)
        row.addWidget(value_label)
        container = QtWidgets.QWidget()
        container.setLayout(row)
        return container

    def sync_from_main(self):
        mw = self.main_window
        if mw is None:
            return
        if hasattr(mw, 'brightness_value'):
            self._update_slider(self.brightness_slider, self.brightness_value_label, mw.brightness_value)
        if hasattr(mw, 'contrast_value'):
            self._update_slider(self.contrast_slider, self.contrast_value_label, mw.contrast_value)

    def _update_slider(self, slider, label, value):
        block = slider.blockSignals(True)
        slider.setValue(int(value))
        slider.blockSignals(block)
        label.setText(str(int(value)))

    def _on_brightness_changed(self, value):
        self.brightness_value_label.setText(str(int(value)))
        mw = self.main_window
        if mw is None:
            return
        if hasattr(mw, 'brightness_slider'):
            block = mw.brightness_slider.blockSignals(True)
            mw.brightness_slider.setValue(int(value))
            mw.brightness_slider.blockSignals(block)
        if hasattr(mw, '_on_brightness_changed'):
            mw._on_brightness_changed(int(value))

    def _on_contrast_changed(self, value):
        self.contrast_value_label.setText(str(int(value)))
        mw = self.main_window
        if mw is None:
            return
        if hasattr(mw, 'contrast_slider'):
            block = mw.contrast_slider.blockSignals(True)
            mw.contrast_slider.setValue(int(value))
            mw.contrast_slider.blockSignals(block)
        if hasattr(mw, '_on_contrast_changed'):
            mw._on_contrast_changed(int(value))

    def update_brightness_display(self, value):
        self._update_slider(self.brightness_slider, self.brightness_value_label, value)

    def update_contrast_display(self, value):
        self._update_slider(self.contrast_slider, self.contrast_value_label, value)
