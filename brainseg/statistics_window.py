from PyQt6 import QtWidgets
from PyQt6 import QtCore
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .statistics_tracker import statistics_tracker

class StatisticsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Segmentation Statistics")
        self.setMinimumSize(700, 500)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Scrollable content area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(12, 8, 12, 8)
        content_layout.setSpacing(12)
        self._add_all_content(content_layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Single close button at bottom
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setFixedWidth(100)
        btn_close.clicked.connect(self.close)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

    def _add_all_content(self, layout):
        metrics = statistics_tracker.get_metrics()
        stats = statistics_tracker.run_statistical_tests()
        def fmt(val, unit=None, places=3):
            if val is None:
                return "N/A"
            if isinstance(val, float):
                s = f"{val:.{places}f}"
            else:
                s = str(val)
            return f"{s} {unit}" if unit else s

        # --- Metrics Section ---
        metrics_box = QtWidgets.QGroupBox("Session Metrics")
        metrics_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        grid = QtWidgets.QFormLayout(metrics_box)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        grid.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        labels = [
            ("Latency p50", fmt(metrics['latency_p50'], 's')),
            ("Latency p95", fmt(metrics['latency_p95'], 's')),
            ("Memory Peak", fmt(metrics['memory_peak'], 'MB', 1)),
            ("Accuracy Mean", fmt(metrics['accuracy_mean'])),
            ("Accuracy Delta", fmt(metrics['accuracy_delta'])),
            ("Model Load Time", fmt(metrics['model_load_time'], 's')),
        ]
        for name, value in labels:
            lbl_name = QtWidgets.QLabel(name + ":")
            lbl_name.setStyleSheet("font-weight: 500;")
            lbl_val = QtWidgets.QLabel(value)
            lbl_val.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 13px;")
            grid.addRow(lbl_name, lbl_val)
        layout.addWidget(metrics_box)

        # --- Graphs Section ---
        graphs_box = QtWidgets.QGroupBox("Performance Graphs")
        graphs_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        v = QtWidgets.QVBoxLayout(graphs_box)
        n_graphs = 2
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            n_graphs += 1
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            n_graphs += 1
        fig, axs = plt.subplots(1, n_graphs, figsize=(4*n_graphs, 2.8), constrained_layout=True)
        if n_graphs == 1:
            axs = [axs]
        axs[0].plot(metrics['latencies'], label='Latency (s)', color='#1976d2')
        axs[0].set_title('Latency', fontsize=11)
        axs[0].grid(True, linestyle='--', alpha=0.3)
        axs[0].legend()
        axs[1].plot(metrics['memory_peaks'], label='Memory (MB)', color='#ffa726')
        axs[1].set_title('Memory Peak', fontsize=11)
        axs[1].grid(True, linestyle='--', alpha=0.3)
        axs[1].legend()
        idx = 2
        if metrics['accuracies'] and len(metrics['accuracies']) > 0:
            axs[idx].plot(metrics['accuracies'], label='Accuracy', color='#43a047')
            axs[idx].set_title('Accuracy', fontsize=11)
            axs[idx].grid(True, linestyle='--', alpha=0.3)
            axs[idx].legend()
            idx += 1
        if metrics['model_load_times'] and len(metrics['model_load_times']) > 0:
            axs[idx].plot(metrics['model_load_times'], label='Model Load Time (s)', color='#8e24aa')
            axs[idx].set_title('Model Load Time', fontsize=11)
            axs[idx].grid(True, linestyle='--', alpha=0.3)
            axs[idx].legend()
        canvas = FigureCanvas(fig)
        v.addWidget(canvas)
        layout.addWidget(graphs_box)

        # --- Comparison Section ---
        comp_box = QtWidgets.QGroupBox("Model Comparison")
        comp_box.setStyleSheet("font-weight: 600; font-size: 13px;")
        comp_layout = QtWidgets.QHBoxLayout(comp_box)
        if stats:
            comp_text = f"t-statistic: <b>{fmt(stats['t_stat'])}</b>   |   p-value: <b>{fmt(stats['p_value'])}</b>"
        else:
            comp_text = "Comparison: N/A (no candidate model metrics set or no accuracy data)"
        comp_label = QtWidgets.QLabel(comp_text)
        comp_label.setStyleSheet("font-size: 13px; font-family: Consolas, 'Courier New', monospace;")
        comp_layout.addWidget(comp_label)
        layout.addWidget(comp_box)
