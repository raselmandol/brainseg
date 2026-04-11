import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import psutil
from PyQt6 import QtCore, QtGui, QtWidgets

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "opencv-python",
    "numpy",
    "scipy",
    "matplotlib",
    "psutil",
    "PyQt6",
    "nibabel",
]


class SystemInfoWindow(QtWidgets.QWidget):
    closeRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("EnvironmentReportPanel")
        self.setMinimumWidth(280)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setObjectName("EnvironmentHeader")
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(10)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(2)
        title = QtWidgets.QLabel("BrainSeg Environment Report")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.subtitle = QtWidgets.QLabel("Runtime and dependency health summary")
        self.subtitle.setObjectName("EnvironmentSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(self.subtitle)

        self.health_chip = QtWidgets.QLabel("Status: Checking...")
        self.health_chip.setObjectName("StatusChip")
        self.health_chip.setMinimumWidth(120)
        self.health_chip.setMaximumWidth(360)
        self.health_chip.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.health_chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        title_col.addWidget(self.health_chip, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        header_layout.addLayout(title_col, 1)

        root.addWidget(header)

        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        self._build_overview_tab()
        self._build_gpu_tab()
        self._build_packages_tab()

        self._report_cache = ""

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_information)
        actions.addWidget(self.refresh_btn)

        self.copy_btn = QtWidgets.QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        actions.addWidget(self.copy_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.closeRequested.emit)
        actions.addWidget(close_btn)

        root.addLayout(actions)

        self.apply_theme_palette(theme="light", accent_color="#1a73e8", theme_color=None)
        self._set_health_chip_colors("#e9ecef", "#333333")

        self.refresh_information()

    def refresh_information(self):
        system_info = self._collect_system_info()
        python_info = self._collect_python_info()
        gpu_info = self._collect_gpu_info()
        pkg_info = self._collect_package_info()

        self._fill_overview(system_info, python_info, pkg_info)
        self._fill_gpu(gpu_info)
        self._fill_packages(pkg_info)
        self._update_health(pkg_info)
        self._report_cache = self._build_report_text(system_info, python_info, gpu_info, pkg_info)

    def _copy_to_clipboard(self):
        QtWidgets.QApplication.clipboard().setText(self._report_cache)

    def _build_overview_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.system_table = self._create_kv_table()
        self.python_table = self._create_kv_table()

        sys_group = QtWidgets.QGroupBox("System")
        sys_layout = QtWidgets.QVBoxLayout(sys_group)
        sys_layout.addWidget(self.system_table)
        self.system_table.setMinimumHeight(240)

        py_group = QtWidgets.QGroupBox("Python")
        py_layout = QtWidgets.QVBoxLayout(py_group)
        py_layout.addWidget(self.python_table)
        self.python_table.setMinimumHeight(110)

        layout.addWidget(sys_group, 3)
        layout.addWidget(py_group, 2)

        self.tabs.addTab(page, "Overview")

    def _build_gpu_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.gpu_table = self._create_kv_table()
        layout.addWidget(self.gpu_table)

        self.tabs.addTab(page, "GPU")

    def _build_packages_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.packages_summary = QtWidgets.QLabel("Required packages")
        self.packages_summary.setObjectName("PackagesSummary")
        layout.addWidget(self.packages_summary)

        self.packages_table = QtWidgets.QTableWidget(0, 3)
        self.packages_table.setHorizontalHeaderLabels(["Package", "Status", "Version / Details"])
        self.packages_table.verticalHeader().setVisible(False)
        self.packages_table.setAlternatingRowColors(True)
        self.packages_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.packages_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.packages_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.packages_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.packages_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.packages_table, 1)

        self.tabs.addTab(page, "Packages")

    def _create_kv_table(self):
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Item", "Value"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        return table

    def _fill_overview(self, system_info, python_info, package_info):
        self._fill_kv_table(self.system_table, system_info)
        self._fill_kv_table(self.python_table, python_info)
        checked = len(package_info["found"]) + len(package_info["missing"])
        self.subtitle.setText(
            f"Runtime and dependency health summary - checked {checked} required packages"
        )

    def _fill_gpu(self, gpu_info):
        self._fill_kv_table(self.gpu_table, gpu_info)

    def _fill_packages(self, package_info):
        rows = []
        for name, ver in package_info["found"]:
            rows.append((name, "Installed", ver))
        for name in package_info["missing"]:
            rows.append((name, "Missing", "Not installed"))

        self.packages_table.setRowCount(len(rows))
        for i, (pkg, status, details) in enumerate(rows):
            pkg_item = QtWidgets.QTableWidgetItem(pkg)
            status_item = QtWidgets.QTableWidgetItem(status)
            details_item = QtWidgets.QTableWidgetItem(details)

            if status == "Missing":
                status_item.setForeground(QtGui.QColor("#b42318"))
            else:
                status_item.setForeground(QtGui.QColor("#027a48"))

            self.packages_table.setItem(i, 0, pkg_item)
            self.packages_table.setItem(i, 1, status_item)
            self.packages_table.setItem(i, 2, details_item)

        missing_count = len(package_info["missing"])
        installed_count = len(package_info["found"])
        self.packages_summary.setText(
            f"Installed: {installed_count} | Missing: {missing_count} | Total required: {installed_count + missing_count}"
        )

    def _fill_kv_table(self, table, data):
        table.setRowCount(len(data))
        for i, (key, value) in enumerate(data):
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(key)))
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(value)))

    def _update_health(self, package_info):
        missing_count = len(package_info["missing"])
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        if missing_count == 0:
            self.health_chip.setText(f"Status: Healthy | {timestamp}")
            self._set_health_chip_colors("#ecfdf3", "#027a48")
        else:
            self.health_chip.setText(f"Status: Attention ({missing_count} missing) | {timestamp}")
            self._set_health_chip_colors("#fff4ed", "#b42318")

    def _set_health_chip_colors(self, bg, fg):
        self.health_chip.setStyleSheet(
            "padding: 2px 6px; border-radius: 8px; font-size: 10px; font-weight: 600;"
            f"background: {bg}; color: {fg};"
        )

    def apply_theme_palette(self, theme: str, accent_color: str, theme_color=None):
        if theme_color:
            panel_bg = theme_color
            header_bg = self._shade_color(theme_color, 0.92)
            table_bg = self._shade_color(theme_color, 0.96)
            button_bg = self._shade_color(theme_color, 0.9)
            button_hover = self._shade_color(theme_color, 1.06)
            border = self._shade_color(theme_color, 0.78)
            text = self._ideal_text_color(theme_color)
            muted = self._shade_color(text, 1.3 if text == "#111111" else 0.75)
        else:
            if theme == "dark":
                panel_bg = "#1f2630"
                header_bg = "#252f3b"
                table_bg = "#1b222b"
                button_bg = "#2a3442"
                button_hover = "#334154"
                border = "#3e4a5a"
                text = "#e8edf4"
                muted = "#b6c0cf"
            else:
                panel_bg = "#f8fafc"
                header_bg = "#eef3f8"
                table_bg = "#ffffff"
                button_bg = "#f4f6f8"
                button_hover = "#e9edf2"
                border = "#d6dde5"
                text = "#1f2937"
                muted = "#5b6574"

        self.setStyleSheet(
            "QWidget#EnvironmentReportPanel {"
            f"background: {panel_bg}; color: {text};"
            f"border: 1px solid {border}; border-radius: 10px;"
            "}"
            "QFrame#EnvironmentHeader {"
            f"background: {header_bg}; border: 1px solid {border}; border-radius: 8px;"
            "}"
            "QLabel#EnvironmentSubtitle {"
            f"color: {muted};"
            "}"
            "QLabel#PackagesSummary {"
            f"color: {muted};"
            "}"
            "QGroupBox { font-weight: 600; border: 1px solid transparent; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 4px; padding: 0 3px; }"
            "QTableWidget {"
            f"background: {table_bg}; border: 1px solid {border}; border-radius: 6px;"
            "gridline-color: #d9d9d9;"
            "}"
            "QHeaderView::section { font-weight: 600; }"
            "QPushButton {"
            f"background: {button_bg}; color: {text}; border: 1px solid {border};"
            "padding: 5px 12px; border-radius: 6px;"
            "}"
            "QPushButton:hover {"
            f"background: {button_hover}; border-color: {accent_color};"
            "}"
            "QTabWidget::pane {"
            f"border: 1px solid {border}; border-radius: 8px;"
            "top: -1px;"
            "}"
            "QTabBar::tab {"
            f"background: {button_bg}; border: 1px solid {border};"
            "padding: 6px 12px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px;"
            "}"
            "QTabBar::tab:selected {"
            f"background: {table_bg}; border-color: {accent_color};"
            "}"
            "QLabel { font-size: 12px; }"
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

    def _build_report_text(self, system_info, python_info, gpu_info, package_info):
        lines = ["BrainSeg Environment Report", "=" * 28, ""]

        lines.append("System")
        lines.append("------")
        lines.extend([f"{k}: {v}" for k, v in system_info])
        lines.append("")

        lines.append("Python")
        lines.append("------")
        lines.extend([f"{k}: {v}" for k, v in python_info])
        lines.append("")

        lines.append("GPU")
        lines.append("---")
        lines.extend([f"{k}: {v}" for k, v in gpu_info])
        lines.append("")

        lines.append("Packages")
        lines.append("--------")
        for name, ver in package_info["found"]:
            lines.append(f"Installed - {name}: {ver}")
        for name in package_info["missing"]:
            lines.append(f"Missing - {name}")

        return "\n".join(lines)

    def _collect_system_info(self):
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        return [
            ("OS", f"{platform.system()} {platform.release()}"),
            ("OS version", platform.version()),
            ("Platform", platform.platform()),
            ("Machine", platform.machine()),
            ("CPU", platform.processor() or "Unknown"),
            ("CPU cores", f"physical={physical_cores}, logical={logical_cores}"),
            ("Total RAM", f"{total_ram_gb:.2f} GB"),
        ]

    def _collect_python_info(self):
        return [
            ("Version", platform.python_version()),
            ("Executable", sys.executable),
            ("Prefix", sys.prefix),
        ]

    def _collect_gpu_info(self):
        details = []

        try:
            import torch

            details.append(("PyTorch", torch.__version__))
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                details.append(("CUDA available", "yes"))
                details.append(("CUDA devices", str(count)))
                for idx in range(count):
                    name = torch.cuda.get_device_name(idx)
                    details.append((f"GPU {idx}", name))
            else:
                details.append(("CUDA available", "no"))

            mps = getattr(torch.backends, "mps", None)
            if mps is not None:
                details.append(("Apple MPS available", "yes" if mps.is_available() else "no"))
        except Exception as exc:
            details.append(("PyTorch GPU probe", f"failed: {exc}"))

        smi = self._query_nvidia_smi()
        if smi:
            for i, line in enumerate(smi, start=1):
                details.append((f"nvidia-smi #{i}", line))

        if not details:
            details.append(("GPU", "No GPU information detected."))

        return details

    def _query_nvidia_smi(self):
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return []

        cmd = [
            executable,
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []

        if result.returncode != 0:
            stderr_line = (result.stderr or "").strip()
            return [f"command returned exit code {result.returncode}", stderr_line] if stderr_line else []

        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        return lines

    def _collect_package_info(self):
        found = []
        missing = []

        for pkg in REQUIRED_PACKAGES:
            try:
                found.append((pkg, version(pkg)))
            except PackageNotFoundError:
                missing.append(pkg)
            except Exception as exc:
                found.append((pkg, f"error: {exc}"))

        return {"found": found, "missing": missing}
