from PyQt6 import QtWidgets, QtCore, QtGui


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

        theme_box = QtWidgets.QGroupBox("Theme & Appearance")
        theme_layout = QtWidgets.QVBoxLayout(theme_box)
        theme_layout.setSpacing(10)

        self.theme_light_radio = QtWidgets.QRadioButton("Light theme")
        self.theme_dark_radio = QtWidgets.QRadioButton("Dark theme")
        self.theme_button_group = QtWidgets.QButtonGroup(self)
        self.theme_button_group.addButton(self.theme_light_radio)
        self.theme_button_group.addButton(self.theme_dark_radio)
        self.theme_light_radio.toggled.connect(lambda checked: checked and self._on_theme_selected("light"))
        self.theme_dark_radio.toggled.connect(lambda checked: checked and self._on_theme_selected("dark"))
        radio_row = QtWidgets.QHBoxLayout()
        radio_row.addWidget(self.theme_light_radio)
        radio_row.addWidget(self.theme_dark_radio)
        radio_row.addStretch()
        theme_layout.addLayout(radio_row)

        accent_row = QtWidgets.QHBoxLayout()
        accent_label = QtWidgets.QLabel("Accent color:")
        self.accent_combo = QtWidgets.QComboBox()
        self.accent_combo.addItems(["Azure", "Emerald", "Amber", "Rose"])
        self.accent_combo.currentTextChanged.connect(self._on_accent_changed)
        accent_row.addWidget(accent_label)
        accent_row.addWidget(self.accent_combo, 1)
        theme_layout.addLayout(accent_row)

        color_row = QtWidgets.QHBoxLayout()
        color_row.setSpacing(10)
        color_row.addWidget(QtWidgets.QLabel("Custom color:"))
        self.theme_color_preview = QtWidgets.QLabel()
        self.theme_color_preview.setFixedSize(28, 28)
        self.theme_color_preview.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.theme_color_preview.setStyleSheet("background: transparent; border: 1px dashed #888888;")
        color_row.addWidget(self.theme_color_preview)
        self.theme_color_value = QtWidgets.QLabel("Automatic")
        self.theme_color_value.setObjectName("hint")
        color_row.addWidget(self.theme_color_value)
        color_row.addStretch()
        self.color_pick_btn = QtWidgets.QPushButton("Choose…")
        self.color_pick_btn.clicked.connect(self._choose_theme_color)
        color_row.addWidget(self.color_pick_btn)
        self.color_reset_btn = QtWidgets.QPushButton("Reset")
        self.color_reset_btn.clicked.connect(self._reset_theme_color)
        color_row.addWidget(self.color_reset_btn)
        theme_layout.addLayout(color_row)

        layout.addWidget(theme_box)

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
        if hasattr(mw, 'theme'):
            self.update_theme_display(mw.theme)
        if hasattr(mw, 'accent_name'):
            self.update_accent_display(mw.accent_name)
        if hasattr(mw, 'accent_color'):
            self.apply_accent(mw.accent_color, mw.theme if hasattr(mw, 'theme') else 'light')
        if hasattr(mw, 'theme_color'):
            self.update_theme_color_display(mw.theme_color)

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

    def update_theme_display(self, theme_name):
        block_light = self.theme_light_radio.blockSignals(True)
        block_dark = self.theme_dark_radio.blockSignals(True)
        if theme_name == "dark":
            self.theme_dark_radio.setChecked(True)
        else:
            self.theme_light_radio.setChecked(True)
        self.theme_light_radio.blockSignals(block_light)
        self.theme_dark_radio.blockSignals(block_dark)

    def update_accent_display(self, accent_name):
        block = self.accent_combo.blockSignals(True)
        index = self.accent_combo.findText(accent_name)
        if index >= 0:
            self.accent_combo.setCurrentIndex(index)
        self.accent_combo.blockSignals(block)

    def apply_accent(self, accent_hex, theme):
        groove = "#3f454f" if theme == "dark" else "#d6dbe3"
        qcolor = QtGui.QColor(accent_hex)
        darker = qcolor.darker(120).name()
        style = (
            f"QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {groove}; }}"
            f"QSlider::handle:horizontal {{ background: {accent_hex}; border: 1px solid {darker}; width: 14px; margin: -4px 0; border-radius: 7px; }}"
        )
        self.brightness_slider.setStyleSheet(style)
        self.contrast_slider.setStyleSheet(style)

    def update_theme_color_display(self, color_hex):
        if color_hex:
            self.theme_color_preview.setStyleSheet(f"background: {color_hex}; border: 1px solid #444444;")
            self.theme_color_value.setText(color_hex.upper())
        else:
            self.theme_color_preview.setStyleSheet("background: transparent; border: 1px dashed #888888;")
            self.theme_color_value.setText("Automatic")

    def _on_theme_selected(self, theme):
        mw = self.main_window
        if mw is None:
            return
        mw.set_theme(theme)

    def _on_accent_changed(self, accent_name):
        mw = self.main_window
        if mw is None:
            return
        mw.set_accent(accent_name)

    def _choose_theme_color(self):
        mw = self.main_window
        if mw is None:
            return
        initial = QtGui.QColor(mw.theme_color if getattr(mw, 'theme_color', None) else ("#1a73e8" if mw.theme == "light" else "#1b1b1b"))
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Application Color")
        if color.isValid():
            hex_color = color.name()
            mw.set_theme_color(hex_color)
            self.update_theme_color_display(hex_color)

    def _reset_theme_color(self):
        mw = self.main_window
        if mw is None:
            return
        mw.set_theme_color(None)
        self.update_theme_color_display(None)
