from PyQt6 import QtWidgets, QtCore


class SettingsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(False)
        self.setMinimumSize(400, 320)
        self.setObjectName("SettingsWindow")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        placeholder = QtWidgets.QFrame()
        placeholder.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        placeholder.setMinimumHeight(120)
        layout.addWidget(placeholder)
        layout.addStretch()

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedWidth(96)
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)
