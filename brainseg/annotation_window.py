from PyQt6 import QtCore, QtGui, QtWidgets


class AnnotationWindow(QtWidgets.QWidget):
	"""Floating window for managing labels and annotation shapes."""

	def __init__(self, main_window: 'SegmentationApp'):
		super().__init__(parent=main_window, flags=QtCore.Qt.WindowType.Window)
		self.setWindowTitle("Annotation Manager")
		self.setMinimumSize(520, 420)
		self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
		self.main_window = main_window
		self._build_ui()
		self._register_controls()
		self.sync_from_main()

	def _build_ui(self):
		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(16, 16, 16, 16)
		layout.setSpacing(16)

		title = QtWidgets.QLabel("Annotation Manager")
		title.setStyleSheet("font-weight: 600; font-size: 17px;")
		layout.addWidget(title)

		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		splitter.setChildrenCollapsible(False)

		self.labels_group = QtWidgets.QGroupBox("Labels")
		labels_layout = QtWidgets.QVBoxLayout(self.labels_group)
		labels_layout.setSpacing(8)
		self.annotation_list = QtWidgets.QListWidget()
		self.annotation_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
		self.annotation_list.setAlternatingRowColors(True)
		labels_layout.addWidget(self.annotation_list, 1)
		label_btns = QtWidgets.QHBoxLayout()
		self.btn_add_label = QtWidgets.QPushButton("Add")
		self.btn_rename_label = QtWidgets.QPushButton("Rename")
		self.btn_remove_label = QtWidgets.QPushButton("Remove")
		label_btns.addWidget(self.btn_add_label)
		label_btns.addWidget(self.btn_rename_label)
		label_btns.addWidget(self.btn_remove_label)
		labels_layout.addLayout(label_btns)
		splitter.addWidget(self.labels_group)

		self.shapes_group = QtWidgets.QGroupBox("Annotation Shapes")
		shapes_layout = QtWidgets.QVBoxLayout(self.shapes_group)
		shapes_layout.setSpacing(8)
		self.annotation_shape_table = QtWidgets.QTableWidget(0, 2)
		self.annotation_shape_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
		self.annotation_shape_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
		self.annotation_shape_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
		self.annotation_shape_table.setHorizontalHeaderLabels(["Shape", "Vertices"])
		self.annotation_shape_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
		self.annotation_shape_table.verticalHeader().setVisible(False)
		shapes_layout.addWidget(self.annotation_shape_table, 1)
		shape_btns = QtWidgets.QHBoxLayout()
		self.btn_start_polygon = QtWidgets.QPushButton("Start")
		self.btn_edit_shape = QtWidgets.QPushButton("Edit")
		self.btn_delete_shape = QtWidgets.QPushButton("Delete")
		self.btn_export_labels = QtWidgets.QPushButton("Export")
		shape_btns.addWidget(self.btn_start_polygon)
		shape_btns.addWidget(self.btn_edit_shape)
		shape_btns.addWidget(self.btn_delete_shape)
		shape_btns.addWidget(self.btn_export_labels)
		shapes_layout.addLayout(shape_btns)
		splitter.addWidget(self.shapes_group)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 2)
		layout.addWidget(splitter, 1)

		self.annotation_hint = QtWidgets.QLabel(
			"Select a label, then draw new polygons or pick an existing shape to edit. Drag handles to adjust vertices, double-click to commit, or press Esc to cancel."
		)
		self.annotation_hint.setWordWrap(True)
		self.annotation_hint.setObjectName("hint")
		layout.addWidget(self.annotation_hint)
		self.apply_theme(self.main_window.theme)

	def _register_controls(self):
		mw = self.main_window
		mw.annotation_list = self.annotation_list
		mw.btn_add_label = self.btn_add_label
		mw.btn_remove_label = self.btn_remove_label
		mw.btn_rename_label = self.btn_rename_label
		mw.btn_start_polygon = self.btn_start_polygon
		mw.btn_export_labels = self.btn_export_labels
		mw.btn_edit_shape = self.btn_edit_shape
		mw.btn_delete_shape = self.btn_delete_shape
		mw.annotation_shape_table = self.annotation_shape_table

		self.annotation_list.itemSelectionChanged.connect(mw._handle_label_selection_change)
		self.annotation_list.itemDoubleClicked.connect(self._handle_list_double_click)
		self.annotation_shape_table.itemSelectionChanged.connect(mw._on_shape_selection_change)
		self.btn_add_label.clicked.connect(mw._add_annotation_label)
		self.btn_rename_label.clicked.connect(mw._rename_annotation_label)
		self.btn_remove_label.clicked.connect(mw._remove_annotation_label)
		self.btn_start_polygon.clicked.connect(mw._start_polygon_annotation)
		self.btn_edit_shape.clicked.connect(mw._edit_annotation_shape)
		self.btn_delete_shape.clicked.connect(mw._delete_annotation_shape)
		self.btn_export_labels.clicked.connect(mw._export_annotations)

	def _handle_list_double_click(self, _item: QtWidgets.QListWidgetItem):
		self.main_window._start_polygon_annotation()

	def sync_from_main(self):
		self.main_window._refresh_annotation_list()
		self.main_window._update_annotation_buttons()
		self.apply_theme(self.main_window.theme)

	def apply_theme(self, theme: str):
		border_color = "#111111" if theme == "light" else "#444a57"
		group_style = (
			f"QGroupBox {{ border: 1px solid {border_color}; border-radius: 10px; margin-top: 12px; }}"
			"QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; font-weight: 600; }"
		)
		for group in (getattr(self, "labels_group", None), getattr(self, "shapes_group", None)):
			if group is not None:
				group.setStyleSheet(group_style)

	def closeEvent(self, event: QtGui.QCloseEvent):
		if self.main_window.isVisible():
			self.hide()
			event.ignore()
			return
		super().closeEvent(event)
